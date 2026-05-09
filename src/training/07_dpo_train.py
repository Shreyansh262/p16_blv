"""
07_dpo_train.py
DPO training on SmolVLM2-500M starting from SFT checkpoint.
Reads dpo_pairs.json from 04_build_dpo_pairs.py

Strategy:
  SmolVLM is not in transformers MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES,
  so TRL DPOTrainer cannot auto-detect it as a vision model. We:
  1. Use SmolVLMForConditionalGeneration (not AutoModelForVision2Seq)
  2. Pre-tokenise with the vision processor ourselves
  3. Use a PyTorch Dataset + custom collate_fn with dynamic padding
     for pixel_values (same pattern as 06_sft_train.py)
  4. Subclass DPOTrainer to skip its internal tokenisation

Usage:
    python 07_dpo_train.py --max_samples 100
    python 07_dpo_train.py --max_samples 5000
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
    BitsAndBytesConfig,
)
from peft import (
    PeftModel, LoraConfig, get_peft_model,
    TaskType, prepare_model_for_kbit_training,
)
from trl import DPOConfig, DPOTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            Path.home() /
            "p16_blv/logs/training_logs/dpo_training.log"
        )
    ]
)
log = logging.getLogger(__name__)

MAX_SIDE = 364  # cap resolution to reduce token count


# ── DPOTrainer subclass ──────────────────────────────────────────────────────
class VisionDPOTrainer(DPOTrainer):
    """DPOTrainer that accepts pre-tokenised vision datasets.

    SmolVLM is absent from MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES so
    DPOTrainer falls into the text-only tokenize path and crashes.
    We pre-tokenise everything ourselves and skip _prepare_dataset.
    """

    def _prepare_dataset(self, dataset, processing_class, args, dataset_name):
        return dataset


# ── Config ───────────────────────────────────────────────────────────────────
def load_config():
    config_path = (Path(__file__).parent.parent.parent
                   / "config" / "paths_config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Model loading ────────────────────────────────────────────────────────────
def load_model_and_processor(cfg, from_sft=True):
    hf_cache = cfg["cache"]["hf_cache"]
    os.environ["HF_HOME"] = hf_cache

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    base_id = cfg["models"]["student_id"]

    if from_sft:
        sft_path = cfg["models"]["student_sft"]
        log.info("Loading base model: %s", base_id)
        model = SmolVLMForConditionalGeneration.from_pretrained(
            base_id,
            quantization_config=bnb_cfg,
            device_map={"": 0},
            cache_dir=hf_cache,
        )

        log.info("Loading SFT LoRA adapter: %s", sft_path)
        model = PeftModel.from_pretrained(model, sft_path)
        # Merge SFT LoRA into base weights so DPO LoRA trains on top.
        # 4-bit merge is slightly lossy (unavoidable with quantisation).
        model = model.merge_and_unload()

        # Clean stale peft metadata left after merge to prevent
        # "multiple adapters" warning when adding DPO LoRA.
        if hasattr(model, "peft_config"):
            delattr(model, "peft_config")

        processor = AutoProcessor.from_pretrained(
            sft_path, cache_dir=hf_cache
        )
    else:
        log.info("Loading base model: %s", base_id)
        model = SmolVLMForConditionalGeneration.from_pretrained(
            base_id,
            quantization_config=bnb_cfg,
            device_map={"": 0},
            cache_dir=hf_cache,
        )
        processor = AutoProcessor.from_pretrained(
            base_id, cache_dir=hf_cache
        )

    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    log.info("Model ready for DPO.")
    return model, processor


# ── Image loading ────────────────────────────────────────────────────────────
def load_images(kf_paths):
    images = []
    for kp in kf_paths:
        if Path(kp).exists():
            try:
                img = Image.open(kp).convert("RGB")
                if max(img.size) > MAX_SIDE:
                    img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
                images.append(img)
            except Exception:
                pass
    return images


# ── PyTorch Dataset (mirrors SFT pattern) ────────────────────────────────────
class BLVDPODataset(Dataset):
    """Pre-tokenised DPO dataset returning tensors per sample."""

    def __init__(self, pairs, processor, max_samples=None):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.eos_id = self.tokenizer.eos_token_id
        self.data = pairs[:max_samples] if max_samples else pairs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        p = self.data[idx]
        kf_paths = p.get("keyframe_paths", [])
        images = load_images(kf_paths)

        if not images:
            return None

        # Build user message with images (same format as SFT)
        user_messages = [
            {
                "role": "user",
                "content": (
                    [{"type": "image", "image": img} for img in images]
                    + [{"type": "text",
                        "text": "Describe this video for a blind user."}]
                )
            }
        ]

        prompt_text = self.processor.apply_chat_template(
            user_messages, tokenize=False, add_generation_prompt=True
        )

        # Process prompt + images together
        processed = self.processor(
            text=[prompt_text],
            images=images,
            return_tensors="pt",
            add_special_tokens=False,
        )

        prompt_input_ids = processed["input_ids"].squeeze(0)
        pixel_values = processed["pixel_values"].squeeze(0)

        chosen_input_ids = torch.tensor(
            self.tokenizer(p["chosen"], add_special_tokens=False)["input_ids"]
            + [self.eos_id],
            dtype=torch.long,
        )
        rejected_input_ids = torch.tensor(
            self.tokenizer(p["rejected"], add_special_tokens=False)["input_ids"]
            + [self.eos_id],
            dtype=torch.long,
        )

        result = {
            "prompt_input_ids": prompt_input_ids,
            "chosen_input_ids": chosen_input_ids,
            "rejected_input_ids": rejected_input_ids,
            "pixel_values": pixel_values,
        }

        if "pixel_attention_mask" in processed:
            result["pixel_attention_mask"] = processed["pixel_attention_mask"].squeeze(0)

        return result


# ── Custom collate with dynamic padding (mirrors SFT collate_fn) ─────────────
def dpo_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    pad_id = 2  # SmolVLM2 pad_token_id

    def pad_1d(tensors, pad_value, padding_side="left"):
        max_len = max(t.shape[0] for t in tensors)
        padded, masks = [], []
        for t in tensors:
            pad_len = max_len - t.shape[0]
            if padding_side == "left":
                padded.append(torch.cat([torch.full((pad_len,), pad_value, dtype=t.dtype), t]))
                masks.append(torch.cat([torch.zeros(pad_len, dtype=torch.long), torch.ones(t.shape[0], dtype=torch.long)]))
            else:
                padded.append(torch.cat([t, torch.full((pad_len,), pad_value, dtype=t.dtype)]))
                masks.append(torch.cat([torch.ones(t.shape[0], dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)]))
        return torch.stack(padded), torch.stack(masks)

    # Pad prompt (left-padded, as DPOTrainer expects)
    prompt_ids, prompt_mask = pad_1d(
        [b["prompt_input_ids"] for b in batch], pad_id, "left"
    )
    # Pad chosen/rejected (right-padded)
    chosen_ids, chosen_mask = pad_1d(
        [b["chosen_input_ids"] for b in batch], pad_id, "right"
    )
    rejected_ids, rejected_mask = pad_1d(
        [b["rejected_input_ids"] for b in batch], pad_id, "right"
    )

    # Dynamic padding for pixel_values (same as SFT collate_fn)
    max_frames = max(b["pixel_values"].shape[0] for b in batch)
    padded_pixel_values = []
    for b in batch:
        pv = b["pixel_values"]
        frames_to_add = max_frames - pv.shape[0]
        if frames_to_add > 0:
            padding = torch.zeros(
                (frames_to_add, pv.shape[1], pv.shape[2], pv.shape[3]),
                dtype=pv.dtype,
            )
            pv = torch.cat([pv, padding], dim=0)
        padded_pixel_values.append(pv)

    output = {
        "prompt_input_ids": prompt_ids,
        "prompt_attention_mask": prompt_mask,
        "chosen_input_ids": chosen_ids,
        "chosen_attention_mask": chosen_mask,
        "rejected_input_ids": rejected_ids,
        "rejected_attention_mask": rejected_mask,
        "pixel_values": torch.stack(padded_pixel_values),
    }

    # Dynamic padding for pixel_attention_mask if present
    if "pixel_attention_mask" in batch[0]:
        max_frames_pam = max(b["pixel_attention_mask"].shape[0] for b in batch)
        padded_pam = []
        for b in batch:
            pam = b["pixel_attention_mask"]
            frames_to_add = max_frames_pam - pam.shape[0]
            if frames_to_add > 0:
                padding = torch.zeros(
                    (frames_to_add, pam.shape[1], pam.shape[2]),
                    dtype=pam.dtype,
                )
                pam = torch.cat([pam, padding], dim=0)
            padded_pam.append(pam)
        output["pixel_attention_mask"] = torch.stack(padded_pam)

    return output


# ── Training ─────────────────────────────────────────────────────────────────
def run_dpo(cfg, max_samples, from_sft):
    pairs_path = Path(cfg["data"]["dpo_pairs"])
    out_dir = Path(cfg["models"]["student_dpo"])
    log_dir = Path(cfg["logs"]["training"])

    if not pairs_path.exists():
        log.error(
            "DPO pairs not found: %s\nRun 04_build_dpo_pairs.py first.",
            pairs_path,
        )
        raise SystemExit(1)

    model, processor = load_model_and_processor(cfg, from_sft=from_sft)

    with open(pairs_path) as f:
        data = json.load(f)

    train_pairs = data.get("train", [])[:max_samples]
    val_pairs = data.get("val", [])[:max(20, max_samples // 10)]

    train_ds = BLVDPODataset(train_pairs, processor)
    val_ds = BLVDPODataset(val_pairs, processor)

    log.info("DPO dataset -- train: %d | val: %d", len(train_ds), len(val_ds))

    dpo_config = DPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=5e-7,
        beta=0.1,
        bf16=True,
        logging_dir=str(log_dir),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=50,
        save_total_limit=2,
        max_length=None,
        max_prompt_length=None,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        dataloader_num_workers=2,
        report_to="tensorboard",
    )

    trainer = VisionDPOTrainer(
        model=model,
        ref_model=None,  # implicit reference via LoRA base weights
        args=dpo_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=dpo_collate_fn,
        processing_class=processor.tokenizer,
    )

    log.info("Starting DPO training...")
    trainer.train()

    model.save_pretrained(str(out_dir))
    processor.save_pretrained(str(out_dir))
    log.info("DPO complete. Model saved to: %s", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--from_sft", action="store_true", default=True,
                        help="Load from SFT checkpoint (recommended)")
    parser.add_argument("--from_base", action="store_true",
                        help="Load from base model instead of SFT")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    cfg = load_config()
    from_sft = not args.from_base
    run_dpo(cfg, args.max_samples, from_sft)
