"""
06_sft_stage1_projector.py
Stage 1 SFT: Train vision encoder (SigLIP) + multimodal projector.
Language model is fully frozen. No LoRA.

Goal: Adapt visual embeddings to BLV scene types before Stage 2 LM fine-tuning.

Usage:
    python 06_sft_stage1_projector.py --max_samples 50 --epochs 1   # smoke test
    python 06_sft_stage1_projector.py --max_samples 9000 --epochs 1  # full run
"""

import os, json, logging, argparse
from pathlib import Path
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    SmolVLMForConditionalGeneration,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            Path.home() / "p16_blv/logs/training_logs/sft_stage1.log"
        ),
    ],
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path.home() / "p16_blv"


def load_config():
    with open(PROJECT_ROOT / "config" / "paths_config.yaml") as f:
        return yaml.safe_load(f)


# ── Dataset (same as fixed 06_sft_train.py) ────────────────────────────────────
class BLVSFTDataset(Dataset):
    def __init__(self, captions_data, processor, max_samples=None):
        self.data = captions_data[:max_samples]
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        description = entry.get("blv_description", "").strip()
        kf_paths = entry.get("keyframe_paths", [])

        MAX_SIDE = 364
        images = []
        for p in kf_paths:
            if Path(p).exists():
                try:
                    img = Image.open(p).convert("RGB")
                    if max(img.size) > MAX_SIDE:
                        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
                    images.append(img)
                except Exception:
                    pass

        if not images:
            return None

        user_messages = [
            {
                "role": "user",
                "content": (
                    [{"type": "image", "image": img} for img in images]
                    + [{"type": "text", "text": "Describe this video for a blind user."}]
                ),
            }
        ]

        prompt_text = self.processor.apply_chat_template(
            user_messages, tokenize=False, add_generation_prompt=True
        )
        full_text = prompt_text + " " + description + "<end_of_utterance>\n"

        prompt_inputs = self.processor(
            text=[prompt_text], images=images, return_tensors="pt"
        )
        prompt_len = prompt_inputs["input_ids"].shape[1]

        full_inputs = self.processor(
            text=[full_text], images=images, return_tensors="pt"
        )

        input_ids = full_inputs["input_ids"].squeeze(0)
        attention_mask = full_inputs["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": full_inputs.get(
                "pixel_values", torch.zeros(1)
            ).squeeze(0),
        }


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    
    # 1. Handle text padding
    max_len = max(b["input_ids"].shape[0] for b in batch)
    pad_id = 2  # SmolVLM2 pad_token_id
    input_ids, attention_masks, labels = [], [], []
    for b in batch:
        seq_len = b["input_ids"].shape[0]
        pad_len = max_len - seq_len
        input_ids.append(
            torch.cat([b["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)])
        )
        attention_masks.append(
            torch.cat([b["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
        )
        labels.append(
            torch.cat([b["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
        )
        
    # 2. Handle vision/pixel_values dynamic padding
    max_frames = max(b["pixel_values"].shape[0] for b in batch)
    padded_pixel_values = []
    
    for b in batch:
        pv = b["pixel_values"]
        frames_to_add = max_frames - pv.shape[0]
        
        if frames_to_add > 0:
            # Pad with zeros: [frames_to_add, Channels, Height, Width]
            padding = torch.zeros(
                (frames_to_add, pv.shape[1], pv.shape[2], pv.shape[3]), 
                dtype=pv.dtype, 
                device=pv.device
            )
            pv = torch.cat([pv, padding], dim=0)
            
        padded_pixel_values.append(pv)

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        "labels": torch.stack(labels),
        "pixel_values": torch.stack(padded_pixel_values),
    }


# ── Loss monitor callback ──────────────────────────────────────────────────────
class LossMonitorCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            step = state.global_step
            train_loss = logs.get("loss", None)
            eval_loss = logs.get("eval_loss", None)
            if train_loss is not None:
                log.info("Step %d | train_loss: %.4f", step, train_loss)
            if eval_loss is not None:
                log.info("Step %d | eval_loss: %.4f", step, eval_loss)
            if train_loss and train_loss < 0.05:
                log.warning(
                    "Loss collapsed below 0.05 -- stopping training"
                )
                control.should_training_stop = True


# ── Model loading (Stage 1: bf16, no quantization) ─────────────────────────────
def load_model_stage1(cfg):
    model_id = cfg["models"]["student_id"]
    hf_cache = cfg["cache"]["hf_cache"]
    os.environ["HF_HOME"] = hf_cache

    log.info("Loading model (bf16, no quantization): %s", model_id)

    model = SmolVLMForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        cache_dir=hf_cache,
    )
    processor = AutoProcessor.from_pretrained(
        model_id, cache_dir=hf_cache, trust_remote_code=True
    )

    # ── Freeze everything ──
    for param in model.parameters():
        param.requires_grad = False

    # ── Unfreeze vision encoder + projector ──
    unfreeze_keywords = [
        "vision_model",
        "connector",
        "image_proj",
        "visual_projection",
        "mm_projector",
    ]
    for name, param in model.named_parameters():
        if any(kw in name for kw in unfreeze_keywords):
            param.requires_grad = True

    # ── Report ──
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info("Trainable: %s / %s (%.1f%%)", f"{trainable:,}", f"{total:,}", 100 * trainable / total)

    log.info("Trainable modules:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            log.info("  %s: %s", name, list(param.shape))

    return model, processor


# ── Main ────────────────────────────────────────────────────────────────────────
def run_stage1(cfg, args):
    captions_path = Path(cfg["data"]["qwen_captions"])
    if not captions_path.exists():
        log.error("Captions not found: %s", captions_path)
        raise SystemExit(1)

    with open(captions_path) as f:
        all_captions = json.load(f)
    log.info("Loaded %d captions", len(all_captions))

    all_captions = [
        c for c in all_captions if len(c.get("blv_description", "").strip()) > 30
    ]
    log.info("After quality filter: %d", len(all_captions))

    train_data = all_captions[: args.max_samples]
    val_data = all_captions[9000:]  # held-out eval
    if not val_data:
        val_data = all_captions[args.max_samples : args.max_samples + 100]

    model, processor = load_model_stage1(cfg)

    train_ds = BLVSFTDataset(train_data, processor, args.max_samples)
    val_ds = BLVSFTDataset(val_data, processor, 200)

    log.info("Train samples: %d | Val samples: %d", len(train_ds), len(val_ds))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name=args.run_name,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        callbacks=[LossMonitorCallback()],
    )

    log.info("Starting Stage 1 training (vision encoder + projector)...")
    trainer.train()

    log.info("Saving model to %s", args.output_dir)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    log.info("Stage 1 complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 1 SFT: vision encoder + projector"
    )
    parser.add_argument("--max_samples", type=int, default=9000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/student/sft_stage1_checkpoint",
    )
    parser.add_argument("--run_name", type=str, default="sft_stage1")
    parser.add_argument("--gpu", type=int, default=6)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    cfg = load_config()
    run_stage1(cfg, args)