import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.chdir("/usershome/cs671_user2/p16_blv")

import torch, json, glob
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

BASE_MODEL  = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
PATCH_PATH  = "models/student/sft_patch_grpo"
OUTPUT_JSON = "results/inference/comparisons/raw_captions_sft_patch_fixed.json"

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

SEP  = "=" * 70
THIN = "-" * 50

def p(msg=""):
    print(msg, flush=True)

p(SEP)
p("SFT_PATCH_GRPO — ZERO-SHOT FIXED — 5 videos")
p(f"Model              : {PATCH_PATH}")
p(f"Videos             : {TARGET_IDS}")
p(f"GPU                : CUDA_VISIBLE_DEVICES=2 (visible as cuda:0)")
p(f"repetition_penalty : 1.3")
p(f"max_new_tokens     : 150")
p(SEP)

video_frames = {}
for vid in TARGET_IDS:
    paths = sorted(glob.glob(f"{KF_DIR}/{vid}_kf*.jpg"))[:4]
    assert len(paths) == 4, f"Missing keyframes for {vid}: {paths}"
    video_frames[vid] = paths

with open("data/generated/all_captions_gemma.json") as f:
    all_data = json.load(f)
ref_map = {e["video_id"]: e.get("blv_description", "") for e in all_data}

p(f"\nLoading model ...")
proc = AutoProcessor.from_pretrained(BASE_MODEL)
if hasattr(proc, "image_processor") and hasattr(proc.image_processor, "max_image_tiles"):
    proc.image_processor.max_image_tiles = 1

base  = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL, dtype=torch.bfloat16, device_map={"": "cuda:0"})
model = PeftModel.from_pretrained(base, PATCH_PATH).merge_and_unload()
model.eval()
p("Model ready.\n")

results = {}

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
        out = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            repetition_penalty=1.3,
        )

    caption = proc.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
    ref     = ref_map.get(vid, "")
    results[vid] = {"gemma_ref": ref, "sft_patch_fixed": caption}

    p(f"\n{THIN}")
    p(f"VIDEO: {vid}")
    p(THIN)
    p(f"[GEMMA REF]")
    p(f"  {ref}")
    p(f"\n[SFT_PATCH_GRPO — zero-shot fixed]")
    p(f"  {caption}")

p(f"\n{SEP}")

Path("outputs").mkdir(exist_ok=True)
output = {
    "model": PATCH_PATH,
    "base_model": BASE_MODEL,
    "prompt_type": "zero_shot",
    "prompt": ZERO_SHOT_PROMPT,
    "repetition_penalty": 1.3,
    "max_new_tokens": 150,
    "videos": TARGET_IDS,
    "results": {
        vid: {
            "keyframes":       video_frames[vid],
            "gemma_ref":       results[vid]["gemma_ref"],
            "sft_patch_fixed": results[vid]["sft_patch_fixed"],
        }
        for vid in TARGET_IDS
    }
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(output, f, indent=2)

p(f"Saved → {OUTPUT_JSON}")
p("=== DONE ===")
