import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["PYTORCH_ALLOC_CONF"]   = "expandable_segments:True"
os.chdir("/usershome/cs671_user2/p16_blv")

import torch, json, glob
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel, LoraConfig, get_peft_model
import torch.nn as nn

BASE_MODEL  = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
GRPO_PATH   = "models/student/grpo/best"
DATA_PATH   = "data/sft_patch_50.json"
SAVE_DIR    = "models/student/sft_patch_grpo"

ZERO_SHOT_PROMPT = (
    "You are an audio description system for blind and low vision users. "
    "Follow these rules:\n"
    "1. First sentence: name the environment type. Write CAUTION if any hazard is within 2 meters.\n"
    "2. Every person and object must have a direction (left/right/center/ahead) and distance in meters.\n"
    "3. Describe people by clothing color and movement direction only.\n"
    "4. Present tense. Maximum 4 sentences. Every word must serve navigation.\n"
    "Describe this scene:"
)

EPOCHS     = 3
LR         = 5e-5
BATCH_SIZE = 2   # gradient-accumulated micro-batches
MAX_FRAMES = 4

LORA_CFG = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.1,
    target_modules=["up_proj", "o_proj", "q_proj", "v_proj",
                    "gate_proj", "k_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

def p(msg=""):
    print(msg, flush=True)

SEP  = "=" * 70
THIN = "-" * 50

p(SEP)
p("SFT PATCH — GRPO_best + 50 GEMMA captions — GPU 2")
p(f"Base    : {BASE_MODEL}")
p(f"Adapter : {GRPO_PATH}")
p(f"Data    : {DATA_PATH}")
p(f"Save    : {SAVE_DIR}")
p(f"Epochs  : {EPOCHS}  |  LR: {LR}  |  Batch: {BATCH_SIZE}")
p(SEP)

# ── Processor (limit image tiles to control memory) ────────────────────────────
p("\nLoading processor ...")
proc = AutoProcessor.from_pretrained(BASE_MODEL)
# SmolVLM2 dynamic tiling: 4 frames × up to 9 tiles × ~729 tokens = OOM.
# Limit to 1 tile per image (still sees the full image, just at lower resolution).
if hasattr(proc, "image_processor") and hasattr(proc.image_processor, "max_image_tiles"):
    proc.image_processor.max_image_tiles = 1
    p(f"Set max_image_tiles=1 on image_processor")
p("Processor ready.")

# ── Load data ──────────────────────────────────────────────────────────────────
with open(DATA_PATH) as f:
    raw = json.load(f)
p(f"Loaded {len(raw)} training entries.")

# ── Build flat sample list: (frames, full_text, prompt_len) ───────────────────
# prompt_len is computed ONCE here on CPU so we don't need a second proc() call
# on GPU during training.
p("Pre-computing prompt lengths (CPU) ...")
samples = []
for e in raw:
    paths  = e["keyframe_paths"][:MAX_FRAMES]
    frames = [Image.open(pp).convert("RGB") for pp in paths]
    label  = e["blv_description"]

    n = len(frames)
    messages = [{"role": "user", "content":
                 [{"type": "image"}] * n + [{"type": "text", "text": ZERO_SHOT_PROMPT}]}]
    text_in  = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_text = text_in + label + proc.tokenizer.eos_token

    # encode prompt+images once on CPU to measure prompt_len
    enc_p = proc(text=text_in, images=frames, return_tensors="pt")
    prompt_len = enc_p["input_ids"].shape[1]
    del enc_p   # free CPU tensors

    samples.append((frames, full_text, prompt_len))

p(f"Pre-computation done. Example prompt_len={samples[0][2]}")

# ── Dataset / DataLoader ───────────────────────────────────────────────────────
class BLVDataset(Dataset):
    def __init__(self, s): self.s = s
    def __len__(self):     return len(self.s)
    def __getitem__(self, i): return self.s[i]

def identity_collate(batch): return batch

dataset    = BLVDataset(samples)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=identity_collate, drop_last=False)

# ── Load GRPO best → merge → new LoRA ─────────────────────────────────────────
p("\nLoading GRPO best and merging ...")
base   = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL, dtype=torch.bfloat16, device_map={"": "cuda:0"})
merged = PeftModel.from_pretrained(base, GRPO_PATH).merge_and_unload()
p("Merged. Attaching fresh LoRA ...")
model  = get_peft_model(merged, LORA_CFG)
model.gradient_checkpointing_enable()
model.print_trainable_parameters()
model.train()

optimizer = torch.optim.AdamW(
    [pp for pp in model.parameters() if pp.requires_grad], lr=LR)

Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
Path("logs").mkdir(exist_ok=True)

# ── Training loop ──────────────────────────────────────────────────────────────
global_step = 0
for epoch in range(1, EPOCHS + 1):
    epoch_loss = 0.0
    n_batches  = 0

    for batch in dataloader:
        optimizer.zero_grad()
        batch_loss = torch.tensor(0.0, device="cuda:0", dtype=torch.float32)

        for (frames, full_text, prompt_len) in batch:
            # Single proc() call — images processed once
            enc = proc(text=full_text, images=frames, return_tensors="pt")
            enc = {k: v.to("cuda:0") for k, v in enc.items()}

            labels_ids = enc["input_ids"].clone()
            labels_ids[:, :prompt_len] = -100   # mask prompt tokens

            outputs = model(**enc, labels=labels_ids)
            # accumulate as float32 to avoid bfloat16 rounding issues
            batch_loss = batch_loss + outputs.loss.float()

            del enc, labels_ids, outputs
            torch.cuda.empty_cache()

        batch_loss = batch_loss / len(batch)
        batch_loss.backward()

        nn.utils.clip_grad_norm_(
            [pp for pp in model.parameters() if pp.requires_grad], 1.0)
        optimizer.step()

        loss_val    = batch_loss.item()
        epoch_loss += loss_val
        n_batches  += 1
        global_step += 1

        if global_step % 5 == 0:
            p(f"  [E{epoch} step {global_step}] loss={loss_val:.4f}")

    avg = epoch_loss / max(n_batches, 1)
    p(f"\nEpoch {epoch}/{EPOCHS} done — avg loss: {avg:.4f}\n")

# ── Save ───────────────────────────────────────────────────────────────────────
model.save_pretrained(SAVE_DIR)
proc.save_pretrained(SAVE_DIR)
p(f"\nSaved → {SAVE_DIR}")
p("=== TRAINING DONE ===")

# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE on 5 target videos
# ══════════════════════════════════════════════════════════════════════════════
p(f"\n{SEP}")
p("POST-TRAINING INFERENCE — 5 videos")
p(SEP)

TARGET_IDS = ["46GP8", "N11GT", "0IH69", "IA2O6", "PFOD8"]
KF_DIR     = "data/keyframes/charades"

video_frames = {}
for vid in TARGET_IDS:
    paths = sorted(glob.glob(f"{KF_DIR}/{vid}_kf*.jpg"))[:4]
    assert len(paths) == 4, f"Missing keyframes for {vid}"
    video_frames[vid] = paths

with open("data/generated/all_captions_gemma.json") as f_ref:
    all_data = json.load(f_ref)
ref_map = {e["video_id"]: e.get("blv_description", "") for e in all_data}

model.eval()
infer_results = {}

for vid in TARGET_IDS:
    frames = [Image.open(path).convert("RGB") for path in video_frames[vid]]
    n = len(frames)

    messages   = [{"role": "user", "content":
                   [{"type": "image"}] * n + [{"type": "text", "text": ZERO_SHOT_PROMPT}]}]
    text_in    = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs     = proc(text=text_in, images=frames, return_tensors="pt")
    inputs     = {k: v.to("cuda:0") for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=300, do_sample=False)

    caption = proc.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
    infer_results[vid] = {"gemma_ref": ref_map.get(vid, ""), "sft_patch": caption}

    p(f"\n{THIN}")
    p(f"VIDEO: {vid}")
    p(THIN)
    p(f"[GEMMA REF]\n  {ref_map.get(vid, '')}")
    p(f"\n[SFT_PATCH]\n  {caption}")

# ── Save inference JSON ────────────────────────────────────────────────────────
Path("outputs").mkdir(exist_ok=True)
output = {
    "model": SAVE_DIR,
    "base_adapter": GRPO_PATH,
    "prompt_type": "zero_shot",
    "prompt": ZERO_SHOT_PROMPT,
    "videos": TARGET_IDS,
    "results": {
        vid: {
            "keyframes": video_frames[vid],
            "gemma_ref": infer_results[vid]["gemma_ref"],
            "sft_patch": infer_results[vid]["sft_patch"],
        }
        for vid in TARGET_IDS
    }
}

with open("results/inference/comparisons/raw_captions_sft_patch.json", "w") as fout:
    json.dump(output, fout, indent=2)

p("\nSaved → outputs/raw_captions_sft_patch.json")
p("=== ALL DONE ===")
