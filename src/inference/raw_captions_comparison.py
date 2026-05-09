import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"   # must precede torch import
os.chdir("/usershome/cs671_user2/p16_blv")

import torch, json, glob, sys
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

# ── Config ─────────────────────────────────────────────────────────────────────
BASE_MODEL   = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
SFT_PATH     = "models/student/sft_v2/best"
GRPO_PATH    = "models/student/grpo/best"
TEACHER_PATH = "/usershome/cs671_user2/.npm-cache-sys/GPT4Scene-and-VLN-R1/ckpts/Qwen2-VL-7B-Instruct"
OUTPUT_JSON  = "results/inference/comparisons/raw_captions_comparison.json"
LOG_SEP      = "=" * 80
LOG_THIN     = "-" * 60

FEW_SHOT_PROMPT = (
    "You are an audio description system for blind and low vision users.\n\n"
    "Examples of correct output:\n"
    "Example 1: CAUTION: Kitchen environment, wet floor 1 meter ahead. "
    "Man in blue shirt moves left at 2 meters. Child stands center at 1.5 meters. "
    "Path right is clear.\n"
    "Example 2: Living room. Person in white shirt seated 2 meters ahead center. "
    "Coffee table at 1 meter right — potential trip hazard. Path left is clear.\n\n"
    "Now describe this scene following the exact same format:"
)

# ── Target videos ──────────────────────────────────────────────────────────────
TARGET_IDS = ["46GP8", "N11GT", "0IH69", "IA2O6", "PFOD8"]

# Build keyframe paths from confirmed on-disk pattern
KF_DIR = "data/keyframes/charades"
video_frames = {}
for vid in TARGET_IDS:
    paths = sorted(glob.glob(f"{KF_DIR}/{vid}_kf*.jpg"))[:4]
    assert len(paths) == 4, f"Expected 4 keyframes for {vid}, got {len(paths)}: {paths}"
    video_frames[vid] = paths

# Load reference descriptions
with open("data/generated/all_captions_gemma.json") as f:
    all_data = json.load(f)
ref_map = {e["video_id"]: e.get("blv_description", "") for e in all_data}

def p(msg="", flush=True):
    print(msg, flush=flush)

Path("outputs").mkdir(exist_ok=True)

p(LOG_SEP)
p("RAW CAPTIONS COMPARISON — 5 videos × 3 models")
p(f"Videos : {TARGET_IDS}")
p(f"GPU    : CUDA_VISIBLE_DEVICES=6  (visible as cuda:0)")
p(f"Models : SFT_v2 | GRPO_best | Qwen2-VL-7B-Teacher")
p(LOG_SEP)

results = {vid: {"ref": ref_map.get(vid, ""), "sft": "", "grpo": "", "teacher": ""}
           for vid in TARGET_IDS}

def load_frames(vid):
    return [Image.open(p).convert("RGB") for p in video_frames[vid]]

# ── SmolVLM2 inference ─────────────────────────────────────────────────────────
def smolvlm_infer(model, processor, frames):
    n = len(frames)
    messages = [{"role": "user", "content":
                 [{"type": "image"}] * n + [{"type": "text", "text": FEW_SHOT_PROMPT}]}]
    text_in = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs  = processor(text=text_in, images=frames, return_tensors="pt")
    inputs  = {k: v.to("cuda:0") for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=300, do_sample=False)
    return processor.decode(out[0][prompt_len:], skip_special_tokens=True).strip()

# ── Qwen2-VL inference ─────────────────────────────────────────────────────────
def qwen_infer(model, processor, frames):
    content = [{"type": "image", "image": img} for img in frames]
    content.append({"type": "text", "text": FEW_SHOT_PROMPT})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    imgs = [item["image"] for item in content if item["type"] == "image"]
    inputs = processor(text=[text], images=imgs, padding=True, return_tensors="pt")
    inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=300, do_sample=False)
    return processor.decode(out[0][prompt_len:], skip_special_tokens=True).strip()

# ══════════════════════════════════════════════════════════════════════════════
# PASS 1 — SFT v2
# ══════════════════════════════════════════════════════════════════════════════
p(f"\n{'>'*3} PASS 1/3 — SFT v2  ({SFT_PATH})")
proc_smol = AutoProcessor.from_pretrained(BASE_MODEL)
base = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL, dtype=torch.bfloat16, device_map={"": "cuda:0"})
sft  = PeftModel.from_pretrained(base, SFT_PATH).merge_and_unload()
sft.eval()
p("SFT v2 ready.")

for vid in TARGET_IDS:
    frames = load_frames(vid)
    cap = smolvlm_infer(sft, proc_smol, frames)
    results[vid]["sft"] = cap
    p(f"\n  {LOG_THIN}")
    p(f"  VIDEO {vid} | SFT_V2")
    p(f"  {LOG_THIN}")
    p(f"  {cap}")

del sft, base
torch.cuda.empty_cache()
p("\nSFT pass done — GPU freed.")

# ══════════════════════════════════════════════════════════════════════════════
# PASS 2 — GRPO best
# ══════════════════════════════════════════════════════════════════════════════
p(f"\n{'>'*3} PASS 2/3 — GRPO best  ({GRPO_PATH})")
base2 = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL, dtype=torch.bfloat16, device_map={"": "cuda:0"})
grpo  = PeftModel.from_pretrained(base2, GRPO_PATH).merge_and_unload()
grpo.eval()
p("GRPO best ready.")

for vid in TARGET_IDS:
    frames = load_frames(vid)
    cap = smolvlm_infer(grpo, proc_smol, frames)
    results[vid]["grpo"] = cap
    p(f"\n  {LOG_THIN}")
    p(f"  VIDEO {vid} | GRPO_BEST")
    p(f"  {LOG_THIN}")
    p(f"  {cap}")

del grpo, base2, proc_smol
torch.cuda.empty_cache()
p("\nGRPO pass done — GPU freed.")

# ══════════════════════════════════════════════════════════════════════════════
# PASS 3 — Teacher (Qwen2-VL-7B)
# ══════════════════════════════════════════════════════════════════════════════
p(f"\n{'>'*3} PASS 3/3 — Teacher  ({TEACHER_PATH})")
try:
    from transformers import Qwen2VLForConditionalGeneration
    proc_qwen = AutoProcessor.from_pretrained(TEACHER_PATH)
    teacher   = Qwen2VLForConditionalGeneration.from_pretrained(
        TEACHER_PATH, dtype=torch.bfloat16, device_map={"": "cuda:0"})
    teacher.eval()
    p("Teacher ready.")

    for vid in TARGET_IDS:
        frames = load_frames(vid)
        cap = qwen_infer(teacher, proc_qwen, frames)
        results[vid]["teacher"] = cap
        p(f"\n  {LOG_THIN}")
        p(f"  VIDEO {vid} | TEACHER_QWEN2VL7B")
        p(f"  {LOG_THIN}")
        p(f"  {cap}")

    del teacher, proc_qwen
    torch.cuda.empty_cache()
    p("\nTeacher pass done — GPU freed.")

except Exception as exc:
    p(f"\nTeacher FAILED: {exc}")
    for vid in TARGET_IDS:
        results[vid]["teacher"] = f"ERROR: {exc}"

# ══════════════════════════════════════════════════════════════════════════════
# FINAL SIDE-BY-SIDE PRINT
# ══════════════════════════════════════════════════════════════════════════════
p(f"\n\n{LOG_SEP}")
p("FINAL SIDE-BY-SIDE COMPARISON")
p(LOG_SEP)

for vid in TARGET_IDS:
    r = results[vid]
    p(f"\n{LOG_SEP}")
    p(f"VIDEO: {vid}")
    p(f"Frames: {video_frames[vid]}")
    p(LOG_THIN)
    p(f"[GEMMA REF]\n  {r['ref']}")
    p(f"\n[SFT_V2]\n  {r['sft']}")
    p(f"\n[GRPO_BEST]\n  {r['grpo']}")
    p(f"\n[TEACHER_QWEN2VL7B]\n  {r['teacher']}")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE JSON
# ══════════════════════════════════════════════════════════════════════════════
output = {
    "videos": TARGET_IDS,
    "models": {
        "sft": SFT_PATH,
        "grpo": GRPO_PATH,
        "teacher": TEACHER_PATH,
    },
    "prompt": FEW_SHOT_PROMPT,
    "results": {
        vid: {
            "keyframes": video_frames[vid],
            "gemma_ref": results[vid]["ref"],
            "sft_v2":    results[vid]["sft"],
            "grpo_best": results[vid]["grpo"],
            "teacher":   results[vid]["teacher"],
        }
        for vid in TARGET_IDS
    }
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(output, f, indent=2)

p(f"\nSaved JSON → {OUTPUT_JSON}")
p("=== ALL DONE ===")
