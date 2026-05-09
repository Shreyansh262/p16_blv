#!/usr/bin/env python3
"""
08_sft_v2.py  --  Single-stage SFT of SmolVLM2-500M-Video-Instruct
from ORIGINAL HuggingFace base weights (NOT from any checkpoint).

Phase 1 (steps 0 -> TRANSITION_STEP):  vision encoder + connector trainable, LoRA FROZEN
Phase 2 (TRANSITION_STEP -> end):       LoRA UNFROZEN, vision LR reduced to 5e-6

All 8 bug fixes from 06_sft_train.py are preserved.
"""

import os, sys, json, logging, argparse, math, shutil
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    SmolVLMForConditionalGeneration,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType

# ── Logging ───────────────────────────────────────────────────────────────────
log = logging.getLogger(__name__)

def setup_logging(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "training.log"),
        ],
    )

MAX_SIDE = 364  # Bug fix #7: cap image resolution

AD_SYSTEM_PROMPT = (
    "You are a professional audio describer for blind and low-vision (BLV) audiences, "
    "following ITC and Netflix Audio Description standards. "
    "STRICT RULES — every description MUST follow all of these: "
    "1. FIRST SENTENCE: Name the environment type and any immediate hazard within 2 meters "
    "(hot surfaces, steps, wet floor, moving people). If a hazard exists, say CAUTION explicitly. "
    "2. SPATIAL POSITIONS: Every object and person must have a direction "
    "(left/right/center/ahead/behind) AND an approximate distance in meters. "
    "3. PEOPLE: Describe by clothing color, position, and direction of movement only. "
    "4. HAZARDS: Explicitly flag steps, ramps, wet floors, hot surfaces, obstacles at "
    "foot level, and moving people or vehicles. "
    "5. DISTANCES: Use approximate meters — approximately 2 meters ahead, 1 meter to your left. "
    "6. Present tense, active voice only. "
    "7. Maximum 4 sentences. Every word must serve navigation. "
    "Write ONE single unified description across all provided images. "
    "Output only one paragraph of maximum 4 sentences. "
    "Describe this video scene for a blind user."
)


# ── Dataset ───────────────────────────────────────────────────────────────────
class BLVSFTDataset(Dataset):
    def __init__(self, captions_data, processor):
        self.data = captions_data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        description = entry.get("blv_description", "").strip()
        kf_paths = entry.get("keyframe_paths", [])

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
                    + [{"type": "text", "text": AD_SYSTEM_PROMPT}]
                ),
            }
        ]

        # Bug fix #1: apply_chat_template on user turn only, then append assistant content manually
        prompt_text = self.processor.apply_chat_template(
            user_messages, tokenize=False, add_generation_prompt=True
        )
        full_text = prompt_text + " " + description + "<end_of_utterance>\n"

        # Bug fix #2: label masking via prompt_len split (NOT token ID search)
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
        labels[:prompt_len] = -100  # mask prompt; train only on assistant response

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
            "pixel_values":   full_inputs.get("pixel_values", torch.zeros(1)).squeeze(0),
        }


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    max_len = max(b["input_ids"].shape[0] for b in batch)
    pad_id  = 2  # Bug fix #3: SmolVLM2 pad_token_id = 2, NOT 0

    input_ids, attention_masks, labels = [], [], []
    for b in batch:
        pad_len = max_len - b["input_ids"].shape[0]
        input_ids.append(
            torch.cat([b["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)])
        )
        attention_masks.append(
            torch.cat([b["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
        )
        labels.append(
            torch.cat([b["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
        )

    # Bug fix #4: dynamic zero-padding for variable frame counts
    max_frames = max(b["pixel_values"].shape[0] for b in batch)
    padded_pixel_values = []
    for b in batch:
        pv = b["pixel_values"]
        frames_to_add = max_frames - pv.shape[0]
        if frames_to_add > 0:
            padding = torch.zeros(
                (frames_to_add, pv.shape[1], pv.shape[2], pv.shape[3]),
                dtype=pv.dtype, device=pv.device,
            )
            pv = torch.cat([pv, padding], dim=0)
        padded_pixel_values.append(pv)

    return {
        "input_ids":      torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        "labels":         torch.stack(labels),
        "pixel_values":   torch.stack(padded_pixel_values),
    }


# ── Callbacks ─────────────────────────────────────────────────────────────────
class FreezeUnfreezeCallback(TrainerCallback):
    """
    Phase 1: sets vision_model + connector trainable, LoRA frozen.
    Phase 2: at TRANSITION_STEP, unfreezes LoRA and halves vision LR to 5e-6.
    """

    def __init__(self, transition_step: int):
        self.transition_step = transition_step
        self.unfrozen = False

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = False
            elif "vision_model" in name or "connector" in name:
                param.requires_grad = True

        v = sum(p.numel() for n, p in model.named_parameters()
                if "vision_model" in n and p.requires_grad)
        c = sum(p.numel() for n, p in model.named_parameters()
                if "connector" in n and p.requires_grad)
        l = sum(p.numel() for n, p in model.named_parameters()
                if "lora_" in n and p.requires_grad)
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)

        log.info("=== PHASE 1: vision + connector trainable, LoRA FROZEN ===")
        log.info(f"  steps 0 -> {self.transition_step}")
        log.info(f"  vision_model trainable: {v:,}")
        log.info(f"  connector trainable:    {c:,}")
        log.info(f"  LoRA trainable:         {l:,}  (FROZEN)")
        log.info(f"  TOTAL trainable:        {total:,}")

    def on_step_end(self, args, state, control, model=None, optimizer=None, **kwargs):
        if not self.unfrozen and state.global_step == self.transition_step:
            # Unfreeze LoRA
            for name, param in model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True

            # Reduce vision encoder LR from 1e-5 to 5e-6
            for pg in optimizer.param_groups:
                if pg.get("name") == "vision":
                    pg["lr"] = 5e-6

            l = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
            total = sum(p.numel() for p in model.parameters() if p.requires_grad)
            log.info(f"=== STEP {self.transition_step}: PHASE 2 BEGINS ===")
            log.info(f"  LoRA UNFROZEN: {l:,} params now trainable")
            log.info(f"  vision LR reduced to 5e-6")
            log.info(f"  TOTAL trainable now: {total:,}")
            self.unfrozen = True


class EpochCheckpointCallback(TrainerCallback):
    """Renames checkpoint-{step} -> checkpoint-epoch{N} after each epoch save.
    Updates state.best_model_checkpoint so load_best_model_at_end still works."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.epoch_num  = 0

    def on_epoch_end(self, args, state, control, **kwargs):
        self.epoch_num += 1

    def on_save(self, args, state, control, **kwargs):
        ckpt_src = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if ckpt_src.exists() and self.epoch_num > 0:
            ckpt_dst = Path(args.output_dir) / f"checkpoint-epoch{self.epoch_num}"
            if not ckpt_dst.exists():
                ckpt_src.rename(ckpt_dst)
                if (state.best_model_checkpoint and
                        str(ckpt_src) in state.best_model_checkpoint):
                    state.best_model_checkpoint = str(ckpt_dst)
                log.info(
                    f"Checkpoint renamed: checkpoint-{state.global_step} "
                    f"-> checkpoint-epoch{self.epoch_num}"
                )


# ── Custom Trainer: differential LR via 3 param groups ────────────────────────
class DifferentialLRTrainer(Trainer):

    def __init__(self, *args, vision_lr=1e-5, connector_lr=1e-5, lora_lr=2e-5, **kwargs):
        self.vision_lr    = vision_lr
        self.connector_lr = connector_lr
        self.lora_lr      = lora_lr
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        vision_params, connector_params, lora_params = [], [], []

        for name, param in self.model.named_parameters():
            if "vision_model" in name:
                vision_params.append(param)
            elif "connector" in name:
                connector_params.append(param)
            elif "lora_" in name:
                lora_params.append(param)
            # Non-LoRA base LM params remain frozen via PEFT; excluded from optimizer

        param_groups = [
            {"params": vision_params,    "lr": self.vision_lr,    "name": "vision"},
            {"params": connector_params, "lr": self.connector_lr, "name": "connector"},
            {"params": lora_params,      "lr": self.lora_lr,      "name": "lora"},
        ]
        param_groups = [pg for pg in param_groups if pg["params"]]

        self.optimizer = torch.optim.AdamW(
            param_groups, weight_decay=0.01, eps=1e-8
        )
        log.info(
            f"Optimizer: AdamW | "
            f"vision_lr={self.vision_lr} | connector_lr={self.connector_lr} | lora_lr={self.lora_lr}"
        )
        return self.optimizer


# ── Model helpers ──────────────────────────────────────────────────────────────
def load_model_and_processor(model_id: str, hf_cache: str):
    os.environ["HF_HOME"] = hf_cache
    log.info(f"Loading base model (bf16, no quantization): {model_id}")
    log.info(f"HF cache: {hf_cache}")

    model = SmolVLMForConditionalGeneration.from_pretrained(  # Bug fix #5
        model_id,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        cache_dir=hf_cache,
    )
    processor = AutoProcessor.from_pretrained(
        model_id, cache_dir=hf_cache, trust_remote_code=True
    )
    log.info("Base model loaded.")
    return model, processor


def apply_lora(model):
    # Bug fix #6: enable_input_require_grads() before get_peft_model so gradients
    # flow through frozen base layers into LoRA adapters.
    # Note: prepare_model_for_kbit_training is NOT used here because we are NOT
    # quantizing — calling it on a bf16 model promotes embeddings to float32 and
    # causes a dtype mismatch in SmolVLM's inputs_merger (Float vs BFloat16).
    model.enable_input_require_grads()
    lora_cfg = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=[
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Single-stage SFT of SmolVLM2-500M")
    parser.add_argument("--max_samples", type=int, default=9646,
                        help="Cap on total samples (train+val, default: full dataset)")
    parser.add_argument("--epochs",      type=int, default=4)
    parser.add_argument("--output_dir",  type=str, default="models/student/sft_v3")
    parser.add_argument("--gpu",         type=int, default=2)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"]    = str(args.gpu)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Bug fix #8

    base_dir = Path(__file__).resolve().parent.parent.parent
    log_dir  = base_dir / "logs" / "sft_v3"
    setup_logging(log_dir)

    log.info("=" * 60)
    log.info("08_sft_v2.py  --  Single-stage SFT from base HF weights")
    log.info(f"  GPU={args.gpu} | epochs={args.epochs} | max_samples={args.max_samples}")
    log.info("=" * 60)

    # ── Load and filter captions ───────────────────────────────────────────────
    captions_path = base_dir / "data" / "generated" / "all_captions_gemma.json"
    if not captions_path.exists():
        log.error(f"Captions not found: {captions_path}")
        sys.exit(1)

    with open(captions_path) as f:
        all_captions = json.load(f)
    log.info(f"Loaded {len(all_captions)} captions from {captions_path.name}")

    all_captions = [
        c for c in all_captions
        if len(c.get("blv_description", "").strip()) > 30
    ]
    log.info(f"After quality filter (len>30): {len(all_captions)}")
    all_captions = all_captions[: args.max_samples]

    split      = int(len(all_captions) * 0.9)
    train_data = all_captions[:split]
    val_data   = all_captions[split:]
    log.info(f"Train: {len(train_data)} | Val: {len(val_data)}")

    # ── Compute TRANSITION_STEP at 25% of total steps ─────────────────────────
    PER_DEVICE_BS   = 4    # fall back to 6+grad_accum=3 if OOM detected on smoke test
    GRAD_ACCUM      = 4
    effective_bs    = PER_DEVICE_BS * GRAD_ACCUM
    steps_per_epoch = math.ceil(len(train_data) / effective_bs)
    total_steps     = steps_per_epoch * args.epochs
    transition_step = max(1, round(total_steps * 0.25))

    log.info(
        f"per_device_bs={PER_DEVICE_BS} | grad_accum={GRAD_ACCUM} | "
        f"effective_bs={effective_bs}"
    )
    log.info(f"steps/epoch={steps_per_epoch} | total_steps={total_steps}")
    log.info(f"TRANSITION_STEP={transition_step}  (25% of {total_steps} total steps)")
    log.info(f"  Phase 1 (vision+connector only):          steps 0 -> {transition_step}")
    log.info(f"  Phase 2 (all trainable, vision LR=5e-6):  steps {transition_step} -> {total_steps}")

    # ── Load model from HF base weights ───────────────────────────────────────
    model_id = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    hf_cache = str(base_dir / "models" / ".hf_cache")
    model, processor = load_model_and_processor(model_id, hf_cache)
    model = apply_lora(model)

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = BLVSFTDataset(train_data, processor)
    val_ds   = BLVSFTDataset(val_data,   processor)

    # ── Output dir ────────────────────────────────────────────────────────────
    if Path(args.output_dir).is_absolute():
        output_dir = Path(args.output_dir)
    else:
        output_dir = base_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── TrainingArguments ─────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=PER_DEVICE_BS,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_steps=250,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_dir=str(log_dir / "tensorboard"),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=4,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    freeze_cb     = FreezeUnfreezeCallback(transition_step)
    epoch_ckpt_cb = EpochCheckpointCallback(output_dir)

    trainer = DifferentialLRTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        callbacks=[freeze_cb, epoch_ckpt_cb],
        vision_lr=1e-5,
        connector_lr=1e-5,
        lora_lr=2e-5,
    )

    log.info("Starting training...")
    trainer.train()

    # ── Save best model to output_dir/best/ ───────────────────────────────────
    best_dir = output_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(best_dir))
    processor.save_pretrained(str(best_dir))
    log.info(f"Best model saved to: {best_dir}")
    log.info("Training complete.")


if __name__ == "__main__":
    main()
