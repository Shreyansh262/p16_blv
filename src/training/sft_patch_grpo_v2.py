import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["PYTORCH_ALLOC_CONF"]   = "expandable_segments:True"
os.chdir("/usershome/cs671_user2/p16_blv")

import gc, glob, json, time
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel, LoraConfig, get_peft_model

BASE_MODEL  = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
GRPO_PATH   = "models/student/grpo/best"
DATA_PATH   = "data/sft_patch_50.json"
SAVE_DIR    = "models/student/sft_patch_grpo_v2"
TARGET_IDS  = ["46GP8", "N11GT", "0IH69", "IA2O6", "PFOD8"]
KF_DIR      = "data/keyframes/charades"

INSTRUCTION = (
    "You are a BLV navigation assistant. Describe the scene for a blind user. "
    "Rules: environment type first, CAUTION only if real hazard present, "
    "meter distances for all objects and people, directional words, "
    "4 sentences max, present tense."
)

EPOCHS     = 3
LR         = 5e-5
BATCH_SIZE = 2
MAX_FRAMES = 4

LORA_CFG = LoraConfig(
    r=64, lora_alpha=128, lora_dropout=0.1,
    target_modules=["up_proj","o_proj","q_proj","v_proj","gate_proj","k_proj","down_proj"],
    bias="none", task_type="CAUSAL_LM",
)

SEP  = "=" * 70
THIN = "-" * 50

def p(msg=""):
    print(msg, flush=True)

p(SEP)
p("SFT PATCH v2 — GRPO_best + 50 GEMMA captions — GPU 2")
p(f"Base adapter : {GRPO_PATH}")
p(f"Data         : {DATA_PATH}")
p(f"Save         : {SAVE_DIR}")
p(f"Instruction  : {INSTRUCTION[:80]}...")
p(f"Epochs {EPOCHS} | LR {LR} | Batch {BATCH_SIZE} | LoRA r=64 α=128")
p(SEP)

# ── Load data ──────────────────────────────────────────────────────────────────
with open(DATA_PATH) as f:
    raw = json.load(f)
p(f"Loaded {len(raw)} training entries.")

# ── Processor (limit tiles for memory) ────────────────────────────────────────
p("\nLoading processor ...")
proc = AutoProcessor.from_pretrained(BASE_MODEL)
if hasattr(proc, "image_processor") and hasattr(proc.image_processor, "max_image_tiles"):
    proc.image_processor.max_image_tiles = 1
    p("  max_image_tiles=1")

# ── Pre-compute samples on CPU ─────────────────────────────────────────────────
p("Pre-computing samples (CPU) ...")
t_pre = time.time()
samples = []
for e in raw:
    paths  = e["keyframe_paths"][:MAX_FRAMES]
    frames = [Image.open(pp).convert("RGB") for pp in paths]
    label  = e["blv_description"]
    n      = len(frames)

    messages = [{"role": "user", "content":
                 [{"type": "image"}] * n + [{"type": "text", "text": INSTRUCTION}]}]
    text_in   = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_text = text_in + label + proc.tokenizer.eos_token

    enc_p      = proc(text=text_in, images=frames, return_tensors="pt")
    prompt_len = enc_p["input_ids"].shape[1]
    del enc_p

    samples.append((frames, full_text, prompt_len))

p(f"Pre-computation done in {time.time()-t_pre:.1f}s | "
  f"example prompt_len={samples[0][2]} | {len(samples)} samples total")

# ── Dataset ───────────────────────────────────────────────────────────────────
class BLVDataset(Dataset):
    def __init__(self, s): self.s = s
    def __len__(self):     return len(self.s)
    def __getitem__(self, i): return self.s[i]

dataloader = DataLoader(
    BLVDataset(samples), batch_size=BATCH_SIZE, shuffle=True,
    collate_fn=lambda b: b, drop_last=False
)

# ── Load GRPO best → merge → fresh LoRA ───────────────────────────────────────
p(f"\nLoading base {BASE_MODEL} ...")
base   = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL, dtype=torch.bfloat16, device_map={"": "cuda:0"})
p(f"Merging GRPO adapter {GRPO_PATH} ...")
merged = PeftModel.from_pretrained(base, GRPO_PATH).merge_and_unload()
p("Attaching fresh LoRA ...")
model  = get_peft_model(merged, LORA_CFG)
model.gradient_checkpointing_enable()
model.print_trainable_parameters()
model.train()

optimizer = torch.optim.AdamW(
    [pp for pp in model.parameters() if pp.requires_grad], lr=LR)

Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
Path("logs").mkdir(exist_ok=True)

# ── Training ───────────────────────────────────────────────────────────────────
p(f"\n{SEP}")
p("TRAINING START")
p(SEP)

total_steps = len(dataloader) * EPOCHS
p(f"Steps per epoch: {len(dataloader)} | Total steps: {total_steps}")
p()

global_step  = 0
all_losses   = []
epoch_losses = []

for epoch in range(1, EPOCHS + 1):
    epoch_loss  = 0.0
    n_batches   = 0
    epoch_start = time.time()

    for batch in dataloader:
        optimizer.zero_grad()
        batch_loss = torch.tensor(0.0, device="cuda:0", dtype=torch.float32)

        for (frames, full_text, prompt_len) in batch:
            enc = proc(text=full_text, images=frames, return_tensors="pt")
            enc = {k: v.to("cuda:0") for k, v in enc.items()}

            lbl = enc["input_ids"].clone()
            lbl[:, :prompt_len] = -100

            out = model(**enc, labels=lbl)
            batch_loss = batch_loss + out.loss.float()

            del enc, lbl, out
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
        all_losses.append((global_step, loss_val))

        # Print every step so we can monitor
        p(f"  [E{epoch} step {global_step:>3}/{total_steps}] loss={loss_val:.4f}")

    avg = epoch_loss / max(n_batches, 1)
    epoch_losses.append(avg)
    elapsed = time.time() - epoch_start
    p(f"\nEpoch {epoch}/{EPOCHS} done | avg_loss={avg:.4f} | {elapsed:.0f}s\n")

p(SEP)
p("LOSS CURVE (all steps):")
for step, loss in all_losses:
    p(f"  step {step:>3}: {loss:.4f}")
p()
p(f"Epoch averages: " + " | ".join(f"E{i+1}={v:.4f}" for i, v in enumerate(epoch_losses)))
p(SEP)

# ── Save ───────────────────────────────────────────────────────────────────────
p(f"\nSaving adapter to {SAVE_DIR} ...")
model.save_pretrained(SAVE_DIR)
proc.save_pretrained(SAVE_DIR)
p(f"Saved. Files:")
for f_ in sorted(Path(SAVE_DIR).iterdir()):
    p(f"  {f_.stat().st_size/1e6:.1f} MB  {f_.name}")
p("=== TRAINING DONE ===")

# ── Inference on 5 target videos ──────────────────────────────────────────────
p(f"\n{SEP}")
p("POST-TRAINING INFERENCE — 5 videos (zero-shot, same instruction)")
p(SEP)

# reload as merged model for clean inference
del model
gc.collect()
torch.cuda.empty_cache()

base2   = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL, dtype=torch.bfloat16, device_map={"": "cuda:0"})
merged2 = PeftModel.from_pretrained(base2, GRPO_PATH).merge_and_unload()
from peft import PeftModel as _P
infer_model = _P.from_pretrained(merged2, SAVE_DIR).merge_and_unload()
infer_model.eval()
p("Inference model ready (GRPO_best → SFT_patch_v2 merged).\n")

with open("data/generated/all_captions_gemma.json") as f_ref:
    all_data = json.load(f_ref)
ref_map = {e["video_id"]: e.get("blv_description", "") for e in all_data}

video_frames = {}
for vid in TARGET_IDS:
    paths = sorted(glob.glob(f"{KF_DIR}/{vid}_kf*.jpg"))[:4]
    assert len(paths) == 4, f"Missing keyframes for {vid}"
    video_frames[vid] = paths

infer_results = {}
for vid in TARGET_IDS:
    frames = [Image.open(path).convert("RGB") for path in video_frames[vid]]
    n = len(frames)
    messages  = [{"role": "user", "content":
                  [{"type": "image"}] * n + [{"type": "text", "text": INSTRUCTION}]}]
    text_in   = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs    = proc(text=text_in, images=frames, return_tensors="pt")
    inputs    = {k: v.to("cuda:0") for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = infer_model.generate(**inputs, max_new_tokens=200,
                                   do_sample=False, repetition_penalty=1.2)
    caption = proc.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
    infer_results[vid] = {"gemma_ref": ref_map.get(vid, ""), "sft_patch_v2": caption}

    p(f"\n{THIN}\nVIDEO: {vid}\n{THIN}")
    p(f"[GEMMA REF]\n  {ref_map.get(vid, '')}")
    p(f"\n[SFT_PATCH_V2]\n  {caption}")

Path("outputs").mkdir(exist_ok=True)
output = {
    "model": SAVE_DIR, "base_adapter": GRPO_PATH,
    "instruction": INSTRUCTION,
    "videos": TARGET_IDS,
    "results": {
        vid: {
            "keyframes": video_frames[vid],
            "gemma_ref": infer_results[vid]["gemma_ref"],
            "sft_patch_v2": infer_results[vid]["sft_patch_v2"],
        } for vid in TARGET_IDS
    }
}
with open("results/inference/comparisons/raw_captions_sft_patch_v2.json", "w") as fout:
    json.dump(output, fout, indent=2)

p(f"\nSaved → outputs/raw_captions_sft_patch_v2.json")
p("=== ALL DONE ===")
