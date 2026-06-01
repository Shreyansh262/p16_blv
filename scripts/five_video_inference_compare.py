#!/usr/bin/env python3
"""
five_video_inference_compare.py
Pick 5 random videos (seed=77), run inference with 5 models, compare outputs.
Usage: python scripts/five_video_inference_compare.py --gpu 2
"""

# ── GPU must be set before any torch/transformers import ─────────────────────
import argparse as _ap
_p = _ap.ArgumentParser(add_help=False)
_p.add_argument("--gpu", type=int, default=2)
_args, _ = _p.parse_known_args()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(_args.gpu)
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# ── torchvision mock (exact pattern from run_inference_all_conditions.py) ─────
import sys as _sys, importlib.machinery as _imm, importlib.util as _imu, enum as _enum
class _InterpolationMode(_enum.Enum):
    NEAREST="nearest"; BILINEAR="bilinear"; BICUBIC="bicubic"
    BOX="box"; HAMMING="hamming"; LANCZOS="lanczos"; NEAREST_EXACT="nearest-exact"
def _mod(n):
    s=_imm.ModuleSpec(n,None,is_package=True); m=_imu.module_from_spec(s); return m
_tv=_mod("torchvision"); _tr=_mod("torchvision.transforms")
_tr.InterpolationMode=_InterpolationMode; _tv.transforms=_tr
for _s in ["_meta_registrations","datasets","models","ops","utils","extension","io",
           "transforms.v2","transforms.v2.functional"]:
    _sys.modules["torchvision."+_s]=_mod("torchvision."+_s)
_sys.modules["torchvision"]=_tv; _sys.modules["torchvision.transforms"]=_tr
del _sys, _imm, _imu, _enum, _mod, _tv, _tr, _s, _InterpolationMode

import gc, json, random, shutil, sys, time, zipfile
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, SmolVLMForConditionalGeneration
from peft import PeftModel

PROJECT_ROOT = Path("/usershome/cs671_user2/p16_blv")
os.chdir(PROJECT_ROOT)

BASE_MODEL       = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
HF_CACHE         = str(PROJECT_ROOT / "models/.hf_cache")
SEED             = 77
N_VIDEOS         = 5
MAX_NEW_TOKENS   = 256

ALL_CAPTIONS_PATH   = PROJECT_ROOT / "data/generated/all_captions.json"
GEMMA_CAPTIONS_PATH = PROJECT_ROOT / "data/generated/all_captions_gemma.json"
OUTPUT_DIR          = PROJECT_ROOT / "outputs"
KF_OUT_DIR          = OUTPUT_DIR / "keyframes_seed77"
RESULT_FILE         = OUTPUT_DIR / "five_video_comparison_seed77.txt"

EXCL_FILES = [
    PROJECT_ROOT / "results/analysis/held_out_test_set.json",
    PROJECT_ROOT / "data/augmented/balanced_splits/balanced_eval.json",
]

# SFT_PATCH: prefer fully merged model, fall back to adapter-on-GRPO
_sft_patch_merged  = PROJECT_ROOT / "models/student/sft_patch_final_merged"
_sft_patch_adapter = PROJECT_ROOT / "models/student/sft_patch_grpo_v2"
_sft_patch_grpo    = PROJECT_ROOT / "models/student/grpo/best"

PROMPT = (
    "You are an assistive AI for blind and low-vision users. Describe this scene for safe navigation. "
    "Include: environment type, key obstacles and furniture with approximate distances and heights, "
    "any hazards (label as CAUTION or hazard), and a clear safe path. Be concise, factual, and "
    "spatially precise. Use present tense. Maximum 4 sentences."
)

SEP  = "=" * 42

# ── Tee output to file + stdout ───────────────────────────────────────────────
OUTPUT_DIR.mkdir(exist_ok=True)
KF_OUT_DIR.mkdir(exist_ok=True)
_out_file = open(RESULT_FILE, "w")

def p(msg="", end="\n"):
    print(msg, end=end, flush=True)
    _out_file.write(msg + end)
    _out_file.flush()

# ─────────────────────────────────────────────────────────────────────────────
p(SEP)
p("FIVE VIDEO INFERENCE COMPARE  seed=77  GPU=" + str(_args.gpu))
p(SEP)
p()

# ── STEP 1 — Select 5 videos (seed=77) ───────────────────────────────────────
p("Loading all_captions.json ...")
with open(ALL_CAPTIONS_PATH) as f:
    all_captions = json.load(f)
caption_map = {e["video_id"]: e for e in all_captions}
all_ids = list(caption_map.keys())
p(f"  Total videos: {len(all_ids)}")

excl_ids = set()
for efp in EXCL_FILES:
    if efp.exists():
        with open(efp) as f:
            data = json.load(f)
        ids_before = len(excl_ids)
        if isinstance(data, list):
            for e in data:
                if "video_id" in e:
                    excl_ids.add(e["video_id"])
        p(f"  Exclusion from {efp.name}: +{len(excl_ids)-ids_before} IDs")

pool = [vid for vid in all_ids if vid not in excl_ids]
p(f"  Pool after exclusion: {len(pool)} videos")
p()

random.seed(SEED)
selected = random.sample(pool, N_VIDEOS)

p("SELECTED VIDEO IDs (seed=77):")
for i, vid in enumerate(selected, 1):
    entry = caption_map[vid]
    p(f"  [{i}/5] {vid}  dataset={entry.get('dataset','?')}")
p()

# ── STEP 2 — Keyframes (copy existing LUV-extracted frames to output dir) ─────
p(SEP)
p("STEP 2 — Keyframes (LUV-extracted, n=4 per video)")
p(SEP)

video_frames = {}
for vid in selected:
    entry = caption_map[vid]
    src_paths = entry.get("keyframe_paths", [])
    out_vid_dir = KF_OUT_DIR / f"video_{vid}"
    out_vid_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for i, src_path in enumerate(src_paths, 1):
        src = Path(src_path)
        if not src.exists():
            p(f"  WARNING: missing keyframe {src_path}")
            continue
        dst = out_vid_dir / f"frame_{i:03d}.jpg"
        shutil.copy2(src, dst)
        frames.append(dst)

    video_frames[vid] = frames
    p(f"  {vid}: {len(frames)} frames → {out_vid_dir}")

p()

# ── Load Gemma teacher captions ───────────────────────────────────────────────
p("Loading Gemma teacher captions ...")
with open(GEMMA_CAPTIONS_PATH) as f:
    gemma_data = json.load(f)
gemma_map = {e["video_id"]: e.get("blv_description", "").strip() for e in gemma_data}
p(f"  Loaded {len(gemma_map)} entries.")
p()

# ── Inference helper ──────────────────────────────────────────────────────────
def run_inference(model, processor, frame_paths):
    images = [Image.open(fp).convert("RGB") for fp in frame_paths]
    messages = [{"role": "user", "content":
        [{"type": "image"}] * len(images) + [{"type": "text", "text": PROMPT}]
    }]
    text_in = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=text_in, images=images, return_tensors="pt")
    inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    return processor.decode(out[0][prompt_len:], skip_special_tokens=True).strip()

def unload(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()

# ── STEP 3 — Sequential model inference ──────────────────────────────────────
results = {vid: {} for vid in selected}

# ── [B] Base model ────────────────────────────────────────────────────────────
p(SEP)
p("[B] BASE MODEL (SmolVLM2-500M, no training)")
p(SEP)
p(f"  Loading from {BASE_MODEL} ...")
proc_b = AutoProcessor.from_pretrained(BASE_MODEL, cache_dir=HF_CACHE)
model_b = SmolVLMForConditionalGeneration.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16,
    device_map={"": "cuda:0"}, cache_dir=HF_CACHE
)
model_b.eval()
for vid in selected:
    t0 = time.time()
    results[vid]["B"] = run_inference(model_b, proc_b, video_frames[vid])
    p(f"  {vid}: done ({time.time()-t0:.1f}s)")
unload(model_b)
del proc_b
p("  [B] unloaded.")
p()

# ── [C] SFT_SINGLE_STAGE ─────────────────────────────────────────────────────
p(SEP)
p("[C] SFT_SINGLE_STAGE")
p(SEP)
ckpt_c = str(PROJECT_ROOT / "models/student/sft_single_stage/best")
p(f"  Loading base + adapter: {ckpt_c}")
proc_c = AutoProcessor.from_pretrained(ckpt_c, cache_dir=HF_CACHE)
model_base_c = SmolVLMForConditionalGeneration.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16,
    device_map={"": "cuda:0"}, cache_dir=HF_CACHE
)
model_c = PeftModel.from_pretrained(model_base_c, ckpt_c).merge_and_unload()
model_c.eval()
for vid in selected:
    t0 = time.time()
    results[vid]["C"] = run_inference(model_c, proc_c, video_frames[vid])
    p(f"  {vid}: done ({time.time()-t0:.1f}s)")
unload(model_c)
del proc_c
p("  [C] unloaded.")
p()

# ── [D] GRPO ──────────────────────────────────────────────────────────────────
p(SEP)
p("[D] GRPO")
p(SEP)
ckpt_d = str(PROJECT_ROOT / "models/student/grpo/best")
p(f"  Loading base + adapter: {ckpt_d}")
proc_d = AutoProcessor.from_pretrained(ckpt_d, cache_dir=HF_CACHE)
model_base_d = SmolVLMForConditionalGeneration.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16,
    device_map={"": "cuda:0"}, cache_dir=HF_CACHE
)
model_d = PeftModel.from_pretrained(model_base_d, ckpt_d).merge_and_unload()
model_d.eval()
for vid in selected:
    t0 = time.time()
    results[vid]["D"] = run_inference(model_d, proc_d, video_frames[vid])
    p(f"  {vid}: done ({time.time()-t0:.1f}s)")
unload(model_d)
del proc_d
p("  [D] unloaded.")
p()

# ── [E] SFT_PATCH ─────────────────────────────────────────────────────────────
p(SEP)
if _sft_patch_merged.exists():
    sft_patch_display = str(_sft_patch_merged)
    p(f"[E] SFT_PATCH (path used: {sft_patch_display})")
    p(SEP)
    p("  Loading merged model ...")
    proc_e  = AutoProcessor.from_pretrained(sft_patch_display, cache_dir=HF_CACHE)
    model_e = SmolVLMForConditionalGeneration.from_pretrained(
        sft_patch_display, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"}
    )
else:
    sft_patch_display = str(_sft_patch_adapter)
    p(f"[E] SFT_PATCH (path used: {sft_patch_display})")
    p(SEP)
    p("  Loading base + GRPO adapter + sft_patch adapter ...")
    proc_e      = AutoProcessor.from_pretrained(sft_patch_display, cache_dir=HF_CACHE)
    model_base_e = SmolVLMForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"}, cache_dir=HF_CACHE
    )
    model_grpo_e = PeftModel.from_pretrained(
        model_base_e, str(_sft_patch_grpo)
    ).merge_and_unload()
    model_e = PeftModel.from_pretrained(
        model_grpo_e, sft_patch_display
    ).merge_and_unload()

model_e.eval()
for vid in selected:
    t0 = time.time()
    results[vid]["E"] = run_inference(model_e, proc_e, video_frames[vid])
    p(f"  {vid}: done ({time.time()-t0:.1f}s)")
unload(model_e)
del proc_e
p("  [E] unloaded.")
p()

# ── STEP 5 — Final formatted output ───────────────────────────────────────────
p()
p(SEP)
p("FINAL COMPARISON OUTPUT")
p(SEP)
p()

for n, vid in enumerate(selected, 1):
    entry = caption_map[vid]
    video_path = entry.get("keyframe_paths", ["<unknown>"])[0]
    p("==========================================")
    p(f"VIDEO [{n}/5]: {vid}  |  Path: {video_path}")
    p("==========================================")
    p("[A] TEACHER (Gemma 31B):")
    p(gemma_map.get(vid, "<not found in gemma captions>"))
    p()
    p("[B] BASE MODEL (SmolVLM2-500M, no training):")
    p(results[vid]["B"])
    p()
    p("[C] SFT_SINGLE_STAGE:")
    p(results[vid]["C"])
    p()
    p("[D] GRPO:")
    p(results[vid]["D"])
    p()
    p(f"[E] SFT_PATCH (path used: {sft_patch_display}):")
    p(results[vid]["E"])
    p()

# ── STEP 6 — ZIP keyframes ────────────────────────────────────────────────────
p(SEP)
p("STEP 6 — Creating keyframe ZIP archives")
p(SEP)

combined_zip_path = OUTPUT_DIR / "keyframes_seed77_all.zip"
with zipfile.ZipFile(combined_zip_path, "w", zipfile.ZIP_DEFLATED) as zout:
    for vid in selected:
        vid_dir = KF_OUT_DIR / f"video_{vid}"
        frames_added = 0
        for frame_file in sorted(vid_dir.glob("*.jpg")):
            arcname = f"video_{vid}/{frame_file.name}"
            zout.write(frame_file, arcname)
            frames_added += 1
        p(f"  video_{vid}: {frames_added} frames added")

p()
p(f"Combined ZIP: {combined_zip_path}")
p(f"Results file: {RESULT_FILE}")
p()
p("=== ALL DONE ===")
_out_file.close()
