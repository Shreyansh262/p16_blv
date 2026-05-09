"""
06_sft_train.py
Supervised Fine-Tuning of SmolVLM2-500M using LoRA.
Reads from qwen_captions.json produced by 03_generate_teacher_captions.py
Usage:
    python 06_sft_train.py --max_samples 500 --epochs 1
    python 06_sft_train.py --max_samples 10000 --epochs 3
"""

import os, json, logging, argparse
from pathlib import Path
from typing import Optional
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    SmolVLMForConditionalGeneration,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            Path.home() /
            "p16_blv/logs/training_logs/sft_training.log"
        )
    ]
)
log = logging.getLogger(__name__)


def load_config():
    config_path = (Path(__file__).parent.parent.parent
                   / "config" / "paths_config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Dataset ────────────────────────────────────────────────────────────────────
class BLVSFTDataset(Dataset):
    def __init__(self, captions_data, processor, max_samples=None):
        self.data      = captions_data[:max_samples]
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry       = self.data[idx]
        description = entry.get("blv_description", "").strip()
        kf_paths    = entry.get("keyframe_paths", [])

        MAX_SIDE = 364  # cap resolution to reduce token count
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

        # Build prompt (user turn only) with generation prompt
        user_messages = [
            {
                "role": "user",
                "content": (
                    [{"type": "image", "image": img}
                     for img in images]
                    + [{
                        "type": "text",
                        "text": "Describe this video for a blind user."
                    }]
                )
            }
        ]

        prompt_text = self.processor.apply_chat_template(
            user_messages, tokenize=False, add_generation_prompt=True
        )
        # Append description + end-of-utterance (matches SmolVLM2 template)
        full_text = prompt_text + " " + description + "<end_of_utterance>\n"



        # Tokenize prompt-only to find where response starts
        prompt_inputs = self.processor(
            text=[prompt_text],
            images=images,
            return_tensors="pt",
        )
        prompt_len = prompt_inputs["input_ids"].shape[1]

        # Tokenize full sequence (prompt + response)
        full_inputs = self.processor(
            text=[full_text],
            images=images,
            return_tensors="pt",
        )

        input_ids      = full_inputs["input_ids"].squeeze(0)
        attention_mask = full_inputs["attention_mask"].squeeze(0)
        labels         = input_ids.clone()

        # Mask prompt tokens — only train on the description
        labels[:prompt_len] = -100

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
            "pixel_values":   full_inputs.get("pixel_values",
                                              torch.zeros(1)).squeeze(0),
        }


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    # Dynamic padding to longest sequence in batch
    max_len = max(b["input_ids"].shape[0] for b in batch)
    pad_id  = 2  # SmolVLM2 pad_token_id (<|im_end|>)
    input_ids, attention_masks, labels = [], [], []
    for b in batch:
        seq_len = b["input_ids"].shape[0]
        pad_len = max_len - seq_len
        input_ids.append(torch.cat([b["input_ids"],      torch.full((pad_len,), pad_id,  dtype=torch.long)]))
        attention_masks.append(torch.cat([b["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
        labels.append(torch.cat([b["labels"],          torch.full((pad_len,), -100, dtype=torch.long)]))
        
    # --- START OF FIX: Dynamic padding for pixel_values ---
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
    # --- END OF FIX ---

    return {
        "input_ids":      torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        "labels":         torch.stack(labels),
        "pixel_values":   torch.stack(padded_pixel_values),
    }


# ── Model loading ──────────────────────────────────────────────────────────────
def load_student_model(cfg, gpu_id=0):
    model_id = cfg["models"]["student_id"]
    hf_cache = cfg["cache"]["hf_cache"]
    os.environ["HF_HOME"] = hf_cache

    log.info(f"Loading student model: {model_id}")

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = SmolVLMForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_cfg,
        device_map={"":0},
        cache_dir=hf_cache,
    )
    processor = AutoProcessor.from_pretrained(
        model_id, cache_dir=hf_cache, trust_remote_code=True
    )
    log.info("Student model loaded.")
    return model, processor


def apply_lora(model):
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=[
            "q_proj", "v_proj",
            "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


# ── Training ───────────────────────────────────────────────────────────────────
def run_sft(cfg, max_samples, epochs, batch_size, gpu_id):
    captions_path = Path(cfg["data"]["qwen_captions"])
    out_dir       = Path(cfg["models"]["student_sft"])
    log_dir       = Path(cfg["logs"]["training"])

    if not captions_path.exists():
        log.error(
            f"Captions not found: {captions_path}\n"
            "Run 03_generate_teacher_captions.py first."
        )
        raise SystemExit(1)

    with open(captions_path) as f:
        all_captions = json.load(f)
    log.info(f"Loaded {len(all_captions)} captions")

    # Filter out entries with no description
    all_captions = [
        c for c in all_captions
        if len(c.get("blv_description", "").strip()) > 30
    ]
    log.info(f"After quality filter: {len(all_captions)}")

    split      = int(len(all_captions) * 0.9)
    train_data = all_captions[:split]
    val_data   = all_captions[split:]

    model, processor = load_student_model(cfg, gpu_id)
    model            = apply_lora(model)

    train_ds = BLVSFTDataset(train_data, processor, max_samples)
    val_ds   = BLVSFTDataset(val_data,   processor,
                              max(50, max_samples // 10))

    log.info(f"Train samples: {len(train_ds)} | "
             f"Val samples: {len(val_ds)}")

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=max(1, 16 // batch_size),
        learning_rate=2e-4,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=True,
        logging_dir=str(log_dir),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="tensorboard",
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
    )

    log.info("Starting SFT training...")
    trainer.train()

    model.save_pretrained(str(out_dir))
    processor.save_pretrained(str(out_dir))
    log.info(f"SFT complete. Model saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int,  default=100)
    parser.add_argument("--epochs",      type=int,  default=1)
    parser.add_argument("--batch_size",  type=int,  default=2)
    parser.add_argument("--gpu",         type=int,  default=0)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    cfg = load_config()
    run_sft(cfg, args.max_samples, args.epochs,
            args.batch_size, args.gpu)