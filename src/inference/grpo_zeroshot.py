import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"   # must precede torch import
os.chdir("/usershome/cs671_user2/p16_blv")

import torch, json, glob
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

# ── Config ─────────────────────────────────────────────────────────────────────
BASE_MODEL  = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
GRPO_PATH   = "models/student/grpo/best"
OUTPUT_JSON = "results/inference/comparisons/raw_captions_zeroshot.json"

ZERO_SHOT_PROMPT = (
    "You are an audio description system for blind and low vision users. "
    "Follow these rules:\n"
    "1. First sentence: name the environment type. Write CAUTION if any hazard is within 2 meters.\n"
    "2. Every person and object must have a direction (left/right/center/ahead) and distance in meters.\n"
    "3. Describe people by clothing color and movement direction only.\n"
    "4. Present tense. Maximum 4 sentences. Every word must serve navigation.\n"
    "Describe this scene:"
)

TARGET_IDS = ["46GP8", "N11GT", "0IH69", "IA2O6", "PFOD8"]
KF_DIR     = "data/keyframes/charades"

video_frames = {}
for vid in TARGET_IDS:
    paths = sorted(glob.glob(f"{KF_DIR}/{vid}_kf*.jpg"))[:4]
    assert len(paths) == 4, f"Missing keyframes for {vid}: {paths}"
    video_frames[vid] = paths

with open("data/generated/all_captions_gemma.json") as f:
    all_data = json.load(f)
ref_map = {e["video_id"]: e.get("blv_description", "") for e in all_data}

Path("outputs").mkdir(exist_ok=True)

SEP  = "=" * 70
THIN = "-" * 50

def p(msg=""):
    print(msg, flush=True)

p(SEP)
p("GRPO_best — ZERO-SHOT PROMPT — 5 videos")
p(f"Videos : {TARGET_IDS}")
p(f"Prompt :\n{ZERO_SHOT_PROMPT}")
p(SEP)

# ── Load GRPO best ─────────────────────────────────────────────────────────────
p(f"\nLoading GRPO best ({GRPO_PATH}) ...")
proc = AutoProcessor.from_pretrained(BASE_MODEL)
base = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL, dtype=torch.bfloat16, device_map={"": "cuda:0"})
grpo = PeftModel.from_pretrained(base, GRPO_PATH).merge_and_unload()
grpo.eval()
p("GRPO best ready.\n")

results = {}

for vid in TARGET_IDS:
    frames = [Image.open(path).convert("RGB") for path in video_frames[vid]]
    n = len(frames)

    messages = [{"role": "user", "content":
                 [{"type": "image"}] * n + [{"type": "text", "text": ZERO_SHOT_PROMPT}]}]
    text_in    = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs     = proc(text=text_in, images=frames, return_tensors="pt")
    inputs     = {k: v.to("cuda:0") for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = grpo.generate(**inputs, max_new_tokens=300, do_sample=False)

    caption = proc.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
    results[vid] = {"gemma_ref": ref_map.get(vid, ""), "grpo_zeroshot": caption}

    p(f"\n{THIN}")
    p(f"VIDEO: {vid}")
    p(THIN)
    p(f"[GEMMA REF]\n  {ref_map.get(vid, '')}")
    p(f"\n[GRPO_BEST — zero-shot]\n  {caption}")

# ── Save JSON ──────────────────────────────────────────────────────────────────
output = {
    "prompt_type": "zero_shot",
    "prompt": ZERO_SHOT_PROMPT,
    "model": GRPO_PATH,
    "videos": TARGET_IDS,
    "results": {
        vid: {
            "keyframes":     video_frames[vid],
            "gemma_ref":     results[vid]["gemma_ref"],
            "grpo_zeroshot": results[vid]["grpo_zeroshot"],
        }
        for vid in TARGET_IDS
    }
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(output, f, indent=2)

p(f"\n\nSaved → {OUTPUT_JSON}")
p("=== DONE ===")
