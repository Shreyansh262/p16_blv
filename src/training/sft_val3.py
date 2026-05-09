#!/usr/bin/env python3
"""
sft_val3 -- BLV SFT Training (corrected: BLV-scored data, 2 epochs, BLV callback) — P16 BLV Project
Matches 06_sft_train.py exactly (same model, LoRA, collator, Trainer).
Samples 15% of all_captions.json (seed=42), holds out 5 for inference.
Run inside: tmux new -s sft_validation
            conda activate blv && cd ~/p16_blv
            export CUDA_VISIBLE_DEVICES=6
            python src/training/sft_validation.py
"""

import os, sys, json, random, logging, re as _re
from pathlib import Path

import yaml
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    SmolVLMForConditionalGeneration,
    TrainingArguments,
    Trainer,
    TrainerCallback, EarlyStoppingCallback,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument('--gpu', type=int, default=0)
_ap.add_argument('--max_samples', type=int, default=3000)
_ap.add_argument('--epochs', type=int, default=2)
_args = _ap.parse_args()


# ── Logging ────────────────────────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = str(_args.gpu)
PROJECT_ROOT = Path.home() / "p16_blv"
log_file     = PROJECT_ROOT / "logs/training_logs/sft_val3.log"
log_file.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
)
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────
def load_config():
    p = PROJECT_ROOT / "config" / "paths_config.yaml"
    with open(p) as f:
        return yaml.safe_load(f)

cfg = load_config()

# ── GPU verify ─────────────────────────────────────────────────────────────
log.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
free_bytes, _ = torch.cuda.mem_get_info(0)
free_gb = free_bytes / 1e9
log.info(f"VRAM free: {free_gb:.1f} GB")
if free_gb < 20.0:
    log.error(f"Only {free_gb:.1f} GB free — need ≥20 GB. Re-run get_free_gpu.sh.")
    sys.exit(1)

# ── Data loading ───────────────────────────────────────────────────────────
captions_path = Path(cfg["data"]["qwen_captions"])
log.info(f"Loading captions from: {captions_path}")
with open(captions_path) as f:
    all_captions = json.load(f)

# Same quality filter as 06_sft_train.py
all_captions = [
    c for c in all_captions
    if len(c.get("blv_description", "").strip()) > 30
]
log.info(f"After quality filter: {len(all_captions)} entries")

# 15% sample, seed=42
def _blv_score_fast(caption):
    sc = 0; t = caption.lower()
    rooms   = ["kitchen","bedroom","bathroom","living","hallway","corridor","office","dining","outdoor","street","cafeteria","gym"]
    spatial = ["left","right","ahead","behind","center","near","far","front","back","side","corner","adjacent","towards"]
    hazards = ["hazard","caution","careful","obstacle","trip","slip","wet","step","edge","slippery","cautious"]
    heights = ["height","waist","knee","chest","floor","ceiling","wall","low","high","above","below","overhead"]
    if any(r in t for r in rooms): sc += 20
    sc += min(20, sum(1 for s in spatial if s in t) * 4)
    if _re.search(r"[0-9]+\s*(meter|feet|foot|inch|cm|step)", t): sc += 20
    elif any(w in t for w in heights): sc += 10
    if any(h in t for h in hazards): sc += 20
    sents = t.count(".") + t.count("!") + t.count("?")
    words = len(t.split())
    if 2 <= sents <= 5 and words <= 100: sc += 20
    elif sents <= 8 and words <= 160:    sc += 10
    return sc

log.info("Scoring all samples by BLV quality...")
_scored = [(_blv_score_fast(c.get("blv_description", "")), c) for c in all_captions]
_scored.sort(key=lambda x: -x[0])
MAX_SAMPLES = _args.max_samples if _args.max_samples else 3000
val_sample = [c for sc, c in _scored[:MAX_SAMPLES] if sc >= 50]
log.info("Top-BLV-scored sample: {} (max={}, min_blv_score=50)".format(len(val_sample), MAX_SAMPLES))

# Hold-out positions (within val_sample list)
GOLDEN_IDS = {"YT43U","QXCUP","FS3SY","JJGEU","X1RBM","WL1DJ","LU82W","2525468588","13090056545","7040233679"}
HOLDOUT_POSITIONS = [0, min(99,len(val_sample)-1), min(249,len(val_sample)-1), min(399,len(val_sample)-1), min(499,len(val_sample)-1)]
holdout_set       = set(HOLDOUT_POSITIONS)
holdout_samples   = [val_sample[i] for i in HOLDOUT_POSITIONS if i < len(val_sample)]
training_samples  = [s for i, s in enumerate(val_sample) if i not in holdout_set and s.get("video_id") not in GOLDEN_IDS]

log.info(f"Hold-out: {len(holdout_samples)} | Training: {len(training_samples)}")
if _args.max_samples:
    training_samples = training_samples[:_args.max_samples]

# ── Dataset — identical to 06_sft_train.py ────────────────────────────────
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

        messages = [
            {
                "role": "user",
                "content": (
                    [{"type": "image", "image": img} for img in images]
                    + [{"type": "text", "text": "Describe this video for a blind user."}]
                ),
            },
            {"role": "assistant", "content": description},
        ]

        text   = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        inputs = self.processor(text=[text], images=images, return_tensors="pt")

        input_ids      = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        labels         = input_ids.clone()

        sep = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        assistant_start = (input_ids == sep).nonzero()
        if len(assistant_start) >= 2:
            labels[: assistant_start[-1].item()] = -100

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
            "pixel_values":   inputs.get("pixel_values", torch.zeros(1)).squeeze(0),
        }


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    max_len = max(b["input_ids"].shape[0] for b in batch)
    pad_id  = 0
    input_ids, attention_masks, labels = [], [], []
    for b in batch:
        pad_len = max_len - b["input_ids"].shape[0]
        input_ids.append(torch.cat([b["input_ids"],      torch.full((pad_len,), pad_id, dtype=torch.long)]))
        attention_masks.append(torch.cat([b["attention_mask"], torch.zeros(pad_len,               dtype=torch.long)]))
        labels.append(torch.cat([b["labels"],          torch.full((pad_len,), -100,  dtype=torch.long)]))
    # Pad pixel_values to max frame count across batch
    pv_list    = [b["pixel_values"] for b in batch]
    max_frames = max(pv.shape[0] for pv in pv_list)
    padded_pv  = []
    for pv in pv_list:
        if pv.shape[0] < max_frames:
            pad = torch.zeros(max_frames - pv.shape[0], *pv.shape[1:], dtype=pv.dtype)
            pv  = torch.cat([pv, pad], dim=0)
        padded_pv.append(pv)

    return {
        "input_ids":      torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        "labels":         torch.stack(labels),
        "pixel_values":   torch.stack(padded_pv),
    }


# ── Model — identical to 06_sft_train.py ──────────────────────────────────
model_id = cfg["models"]["student_id"]
hf_cache = cfg["cache"]["hf_cache"]
os.environ["HF_HOME"]                  = hf_cache
os.environ["PYTORCH_CUDA_ALLOC_CONF"]  = "expandable_segments:True"

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

log.info(f"Loading model: {model_id}")
processor = AutoProcessor.from_pretrained(model_id, cache_dir=hf_cache, trust_remote_code=True)
model     = SmolVLMForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=bnb_cfg,
    device_map={"": 0},
    cache_dir=hf_cache,
)
model = prepare_model_for_kbit_training(model)

lora_cfg = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()


# -- BLV Score Callback (new in sft_val3) -------------------------------------
BLV_LOG_PATH = PROJECT_ROOT / "logs/training_logs/sft_val3_blv_scores.jsonl"
GOLDEN_PATH  = Path("/tmp/golden_set_10.json")
_golden_set  = json.load(open(GOLDEN_PATH)) if GOLDEN_PATH.exists() else []
_blv_plateau = 0
_blv_best    = 0.0
log.info("BLV callback: {} golden samples loaded".format(len(_golden_set)))

class BLVScoreCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        global _blv_plateau, _blv_best
        if not _golden_set: return
        model.eval()
        scores = []
        for g in _golden_set:
            images = []
            for p in g.get("keyframe_paths", []):
                if Path(p).exists():
                    try:
                        img = Image.open(p).convert("RGB")
                        if max(img.size) > 364:
                            img.thumbnail((364, 364), Image.LANCZOS)
                        images.append(img)
                    except Exception:
                        pass
            if not images:
                continue
            msgs = [{"role": "user", "content":
                     [{"type": "image", "image": im} for im in images] +
                     [{"type": "text", "text": "Describe this video for a blind user."}]}]
            prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp    = processor(text=[prompt], images=images, return_tensors="pt")
            inp    = {k: v.to("cuda") for k, v in inp.items()}
            if "pixel_values" in inp:
                inp["pixel_values"] = inp["pixel_values"].to(torch.float16)
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=128, do_sample=True,
                                     temperature=0.7, repetition_penalty=1.3,
                                     no_repeat_ngram_size=3)
            text = processor.tokenizer.decode(
                out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            scores.append(_blv_score_fast(text))
        if not scores:
            return
        avg = sum(scores) / len(scores)
        entry = {"step": state.global_step, "mean_blv_score": round(avg, 2), "n": len(scores)}
        with open(BLV_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
        log.info("Step {} | BLV Score: {:.1f}/100".format(state.global_step, avg))
        if avg <= _blv_best + 1.0:
            _blv_plateau += 1
            if _blv_plateau >= 3:
                log.warning("BLV score plateaued for 3 evals -- stopping training")
                control.should_training_stop = True
        else:
            _blv_best = avg
            _blv_plateau = 0

# ── Callback to capture per-epoch loss ────────────────────────────────────
epoch_losses = {}

class EpochLossCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        for entry in reversed(state.log_history):
            if "loss" in entry:
                epoch_num = round(state.epoch)
                epoch_losses[f"epoch{epoch_num}"] = entry["loss"]
                log.info(f">>> Epoch {epoch_num} loss: {entry['loss']:.4f}")
                break

# ── Datasets ───────────────────────────────────────────────────────────────
CKPT_DIR = PROJECT_ROOT / "models/student/sft_val3_checkpoint"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

eval_size = max(50, len(training_samples) // 10)
train_ds  = BLVSFTDataset(training_samples[:-eval_size], processor)
eval_ds   = BLVSFTDataset(training_samples[-eval_size:],  processor)

log.info(f"Train: {len(train_ds)} | Eval: {len(eval_ds)}")

# ── TrainingArguments — same as 06_sft_train.py ───────────────────────────
training_args = TrainingArguments(
    output_dir=str(CKPT_DIR),
    num_train_epochs=_args.epochs,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    warmup_steps=50,
    lr_scheduler_type="cosine",
    bf16=True,
    gradient_checkpointing=True,
    logging_dir=str(PROJECT_ROOT / "logs/training_logs"),
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_steps=200,
    save_total_limit=2,
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
    eval_dataset=eval_ds,
    data_collator=collate_fn,
    callbacks=[EpochLossCallback(), BLVScoreCallback(), EarlyStoppingCallback(early_stopping_patience=3)],
)

log.info("=" * 60)
log.info("STARTING SFT VALIDATION TRAINING")
log.info(f"Train: {len(train_ds)} | Eval: {len(eval_ds)} | Epochs: 3 | GA steps: 16")
log.info("=" * 60)

trainer.train()
model.save_pretrained(str(CKPT_DIR))
processor.save_pretrained(str(CKPT_DIR))
log.info(f"Checkpoint saved: {CKPT_DIR}")

# ── Post-training inference on held-out samples ────────────────────────────
log.info("\nRunning inference on held-out samples...")
model.eval()

inference_results = []
MAX_SIDE = 364

with torch.no_grad():
    for sample in holdout_samples:
        video_id  = sample.get("video_id", "unknown")
        reference = sample.get("blv_description", "N/A").strip()
        kf_paths  = sample.get("keyframe_paths", [])

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
            inference_results.append({"video_id": video_id,
                                       "generated": "[no keyframes found]",
                                       "reference": reference})
            continue

        messages = [
            {
                "role": "user",
                "content": (
                    [{"type": "image", "image": img} for img in images]
                    + [{"type": "text", "text": "Describe this video for a blind user."}]
                ),
            }
        ]
        prompt_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs     = processor(text=[prompt_text], images=images, return_tensors="pt")
        inputs     = {k: v.to("cuda") for k, v in inputs.items()}
        output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7, repetition_penalty=1.3, no_repeat_ngram_size=3)
        generated  = processor.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        inference_results.append({"video_id": video_id,
                                   "generated": generated,
                                   "reference": reference})

# ── Validation Report ──────────────────────────────────────────────────────
e1   = epoch_losses.get("epoch1", float("nan"))
e2   = epoch_losses.get("epoch2", float("nan"))
e3   = epoch_losses.get("epoch3", float("nan"))
drop = (e1 - e3) if (e1 == e1 and e3 == e3) else 0.0

print("\n" + "=" * 60)
print("sft_val3 FINAL REPORT")
print("=" * 60)

print("\n[1] LOSS CURVE")
print(f"  Epoch 1 loss: {e1:.4f}")
print(f"  Epoch 2 loss: {e2:.4f}")
print(f"  Epoch 3 loss: {e3:.4f}")
if drop > 2.0:
    print(f"  Drop e1→e3:  {drop:.4f}  →  GOOD ✅")
else:
    print(f"  Drop e1→e3:  {drop:.4f}  →  WEAK ⚠️  (check LR or data)")
if e3 < 0.5:
    print(f"  🔴 epoch3_loss < 0.5 → OVERFITTING — check data diversity")

print("\n[2] SAMPLE OUTPUTS (first 120 chars, 3 of 5 held-outs)")
for i, r in enumerate(inference_results[:3]):
    print(f"\n  Sample {i+1} | video_id: {r['video_id']}")
    print(f"  GEN: {r['generated'][:120]}")
    print(f"  REF: {r['reference'][:120]}")

print("\n[3] GO / NO-GO VERDICT")
if 2.0 <= e3 <= 4.0:
    print(f"  epoch3_loss = {e3:.4f}  →  🟢 PROCEED to full 10k training")
elif 4.0 < e3 <= 8.0:
    print(f"  epoch3_loss = {e3:.4f}  →  ⚠️  WEAK — check learning rate before scaling up")
elif e3 > 8.0:
    print(f"  epoch3_loss = {e3:.4f}  →  🔴 STOP — debug data format or tokenizer")
elif e3 < 1.0:
    print(f"  epoch3_loss = {e3:.4f}  →  🔴 STOP — overfitting on ~1500 samples")

print("\n⚠️  REMINDER: Switch back to GPU 6 for future runs (already on GPU 6 ✅)")
print("=" * 60)
