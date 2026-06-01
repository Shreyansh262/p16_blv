#!/usr/bin/env python3
"""
ten_video_inference.py — runs sft_patch_grpo_v2 on 10 videos using pre-extracted keyframes.
Output: ~/p16_blv/outputs/ten_video_eval.json
"""
import os, json, random
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

import torch
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, SmolVLMForConditionalGeneration
from peft import PeftModel

PROJECT  = Path.home() / "p16_blv"
MANIFEST = PROJECT / "data/keyframes/charades_luv_manifest.json"
CKPT     = PROJECT / "models/student/sft_patch_grpo_v2"
BASE_ID  = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
OUT_FILE = PROJECT / "outputs/ten_video_eval.json"
MAX_SIDE = 364

PROMPT = (
    "You are a professional audio describer for blind and low-vision (BLV) audiences, "
    "following ITC and Netflix Audio Description standards. "
    "STRICT RULES: "
    "1. First sentence: name the environment and say CAUTION if any hazard is within 2 meters. "
    "2. Every object and person must have a direction (left/right/center/ahead) and distance in meters. "
    "3. People: describe by clothing color, position, and movement direction only. "
    "4. Hazards: flag steps, ramps, wet floors, hot surfaces, obstacles, moving people/vehicles. "
    "5. Present tense, active voice. Maximum 4 sentences. Every word must serve navigation. "
    "Describe this video scene for a blind user."
)

print(f"Loading manifest ...")
with open(MANIFEST) as f:
    manifest = json.load(f)
print(f"  {len(manifest)} entries")

random.seed(42)
selected = random.sample(manifest, 10)

print(f"Loading processor + base model ...")
processor = AutoProcessor.from_pretrained(BASE_ID)
base = SmolVLMForConditionalGeneration.from_pretrained(
    BASE_ID, torch_dtype=torch.bfloat16, device_map={"": 0}
)
print(f"Applying LoRA from {CKPT} ...")
model = PeftModel.from_pretrained(base, str(CKPT))
model.eval()
print(f"Ready.\n" + "=" * 60)

results = []
for i, entry in enumerate(selected):
    vid_id = entry["video_id"]

    # Load keyframes directly from manifest paths
    frames = []
    for p in entry.get("keyframe_paths", []):
        path = Path(p)
        if path.exists():
            img = Image.open(path).convert("RGB")
            if max(img.size) > MAX_SIDE:
                img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
            frames.append(img)

    if not frames:
        print(f"[{i+1}/10] {vid_id}: no keyframes found, skipping")
        continue

    messages = [{"role": "user", "content":
        [{"type": "image", "image": img} for img in frames]
        + [{"type": "text", "text": PROMPT}]
    }]
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[prompt_text], images=frames, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs, max_new_tokens=200, do_sample=False, repetition_penalty=1.1
        )

    caption = processor.decode(
        out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()

    print(f"[{i+1}/10] {vid_id}  ({len(frames)} frames)")
    print(f"  {caption}\n")
    results.append({"video_id": vid_id, "dataset": entry.get("dataset", "charades"),
                    "n_frames": len(frames), "caption": caption})

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print("=" * 60)
print("FINAL OUTPUT")
print("=" * 60)
for r in results:
    print(f"\nVIDEO ID : {r['video_id']}")
    print(f"CAPTION  : {r['caption']}")
print(f"\nSaved {len(results)} results → {OUT_FILE}")
