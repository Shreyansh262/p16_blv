import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"   # must be set before torch
os.chdir("/usershome/cs671_user2/p16_blv")

import torch, json, random, sys
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_MODEL   = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
SFT_PATH     = "models/student/sft_v2/best"
GRPO_PATH    = "models/student/grpo/best"
TEACHER_PATH = "/usershome/cs671_user2/.npm-cache-sys/GPT4Scene-and-VLN-R1/ckpts/Qwen2-VL-7B-Instruct"
OUTPUT_FILE  = "results/inference/comparisons/inference_comparison_seed42.txt"

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

EXCLUDE_IDS = {"46GP8", "N11GT", "0IH69"}   # seed-99 eval set

# ── Sample 5 videos ────────────────────────────────────────────────────────────
random.seed(42)
with open("data/generated/all_captions_gemma.json") as f:
    all_data = json.load(f)

valid = [
    e for e in all_data
    if e["video_id"] not in EXCLUDE_IDS
    and e.get("keyframe_paths")
    and all(os.path.exists(p) for p in e["keyframe_paths"][:4])
]
random.shuffle(valid)
selected = valid[:5]

print("=" * 80, flush=True)
print(f"3-WAY INFERENCE COMPARISON — seed=42, GPU=6", flush=True)
print(f"Selected video IDs : {[e['video_id'] for e in selected]}", flush=True)
print(f"Excluded (seed-99) : {sorted(EXCLUDE_IDS)}", flush=True)
print("=" * 80, flush=True)

Path("outputs").mkdir(exist_ok=True)

results = {}
for e in selected:
    results[e["video_id"]] = {"ref": e["blv_description"], "dataset": e["dataset"],
                               "paths": e["keyframe_paths"][:4]}

def load_frames(entry, n=4):
    return [Image.open(p).convert("RGB") for p in entry["keyframe_paths"][:n]]

def log(msg):
    print(msg, flush=True)

# ── SmolVLM2 inference helper ──────────────────────────────────────────────────
def smolvlm_infer(model, processor, frames):
    n = len(frames)
    messages = [{"role": "user", "content":
                 [{"type": "image"}] * n + [{"type": "text", "text": FEW_SHOT_PROMPT}]}]
    text_in = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text_in, images=frames, return_tensors="pt")
    inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=300, do_sample=False)
    return processor.decode(out[0][prompt_len:], skip_special_tokens=True).strip()

# ── Qwen2-VL inference helper ──────────────────────────────────────────────────
def qwen_infer(model, processor, frames):
    content = [{"type": "image", "image": img} for img in frames]
    content.append({"type": "text", "text": FEW_SHOT_PROMPT})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs = [item["image"] for item in content if item["type"] == "image"]

    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=300, do_sample=False)
    return processor.decode(out[0][prompt_len:], skip_special_tokens=True).strip()

# ══════════════════════════════════════════════════════════════════════════════
# PASS 1 — SFT v2
# ══════════════════════════════════════════════════════════════════════════════
log("\n>>> PASS 1/3 — SFT v2")
proc_smol = AutoProcessor.from_pretrained(BASE_MODEL)
base = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL, dtype=torch.bfloat16, device_map={"": "cuda:0"})
sft_model = PeftModel.from_pretrained(base, SFT_PATH).merge_and_unload()
sft_model.eval()
log("SFT v2 loaded.")

for entry in selected:
    vid = entry["video_id"]
    frames = load_frames(entry)
    cap = smolvlm_infer(sft_model, proc_smol, frames)
    results[vid]["sft"] = cap
    log(f"  SFT [{vid}]: {cap}")

del sft_model, base
torch.cuda.empty_cache()
log("SFT pass done — GPU memory freed.")

# ══════════════════════════════════════════════════════════════════════════════
# PASS 2 — GRPO best
# ══════════════════════════════════════════════════════════════════════════════
log("\n>>> PASS 2/3 — GRPO best")
base2 = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL, dtype=torch.bfloat16, device_map={"": "cuda:0"})
grpo_model = PeftModel.from_pretrained(base2, GRPO_PATH).merge_and_unload()
grpo_model.eval()
log("GRPO best loaded.")

for entry in selected:
    vid = entry["video_id"]
    frames = load_frames(entry)
    cap = smolvlm_infer(grpo_model, proc_smol, frames)
    results[vid]["grpo"] = cap
    log(f"  GRPO [{vid}]: {cap}")

del grpo_model, base2, proc_smol
torch.cuda.empty_cache()
log("GRPO pass done — GPU memory freed.")

# ══════════════════════════════════════════════════════════════════════════════
# PASS 3 — Teacher (Qwen2-VL-7B-Instruct)
# ══════════════════════════════════════════════════════════════════════════════
log("\n>>> PASS 3/3 — Teacher (Qwen2-VL-7B-Instruct)")
try:
    from transformers import Qwen2VLForConditionalGeneration
    proc_qwen = AutoProcessor.from_pretrained(TEACHER_PATH)
    teacher = Qwen2VLForConditionalGeneration.from_pretrained(
        TEACHER_PATH, dtype=torch.bfloat16, device_map={"": "cuda:0"})
    teacher.eval()
    log("Teacher loaded.")

    for entry in selected:
        vid = entry["video_id"]
        frames = load_frames(entry)
        cap = qwen_infer(teacher, proc_qwen, frames)
        results[vid]["teacher"] = cap
        log(f"  TEACHER [{vid}]: {cap}")

    del teacher, proc_qwen
    torch.cuda.empty_cache()
    log("Teacher pass done.")

except Exception as exc:
    log(f"  Teacher inference failed: {exc}")
    for vid in results:
        results[vid].setdefault("teacher", f"ERROR: {exc}")

# ══════════════════════════════════════════════════════════════════════════════
# BUILD + PRINT + SAVE
# ══════════════════════════════════════════════════════════════════════════════
SEP  = "=" * 80
THIN = "-" * 80

lines = [
    SEP,
    "3-WAY INFERENCE COMPARISON",
    f"Seed=42  |  GPU=6  |  Models: SFT_v2 / GRPO_best / Qwen2-VL-7B-Teacher",
    f"Selected IDs : {[e['video_id'] for e in selected]}",
    f"Excluded     : {sorted(EXCLUDE_IDS)}",
    SEP, "",
]

for i, entry in enumerate(selected, 1):
    vid = entry["video_id"]
    r   = results[vid]
    lines += [
        f"VIDEO {i}/5  |  ID: {vid}  |  Dataset: {r['dataset']}",
        f"Keyframes: {r['paths']}",
        THIN,
        "",
        "[GEMMA REFERENCE]",
        r.get("ref", "N/A"),
        "",
        "[SFT_V2]",
        r.get("sft", "N/A"),
        "",
        "[GRPO_BEST]",
        r.get("grpo", "N/A"),
        "",
        "[TEACHER — Qwen2-VL-7B-Instruct]",
        r.get("teacher", "N/A"),
        "",
        SEP,
        "",
    ]

full_output = "\n".join(lines)

print("\n\n" + full_output, flush=True)

with open(OUTPUT_FILE, "w") as f:
    f.write(full_output)

log(f"\nSaved to {OUTPUT_FILE}")
log("=== ALL DONE ===")
