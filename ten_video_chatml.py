import os, random, torch
from pathlib import Path
from PIL import Image
from collections import defaultdict

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

BASE_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
CKPT    = "/usershome/cs671_user2/p16_blv/models/student/sft_patch_grpo_v2"
FRAMES  = Path("/usershome/cs671_user2/p16_blv/data/keyframes/charades")

# Group flat keyframe files by video_id (prefix before _kf)
groups = defaultdict(list)
for p in FRAMES.glob("*.jpg"):
    vid_id = p.name.split("_kf")[0]
    groups[vid_id].append(p)

video_ids = sorted(groups.keys())
print(f"{len(video_ids)} unique video IDs found in {FRAMES}")

random.seed(42)
selected_ids = random.sample(video_ids, 10)

print(f"Loading processor from {BASE_ID}...")
processor = AutoProcessor.from_pretrained(BASE_ID)

print(f"Loading base model...")
base = AutoModelForImageTextToText.from_pretrained(
    BASE_ID, torch_dtype=torch.bfloat16, device_map="cuda"
)
print(f"Applying LoRA adapter from {CKPT}...")
model = PeftModel.from_pretrained(base, CKPT)
model.eval()
print("Model loaded.\n")

for idx, vid_id in enumerate(selected_ids):
    frames = sorted(groups[vid_id])[:4]

    if not frames:
        print(f"[{idx+1}] {vid_id} — no frames found, skipping")
        continue

    print(f"{'='*60}")
    print(f"[{idx+1}/10] Video: {vid_id}")
    print(f"Frames : {[f.name for f in frames]}")

    images = [Image.open(f).convert("RGB") for f in frames]

    img_tokens = "<image>" * len(images)
    prompt = (
        f"<|im_start|>user\n"
        f"{img_tokens}\n"
        f"Describe this video scene for a blind or low-vision user. "
        f"Mention spatial layout, directions, distances, hazards, and "
        f"any people or objects relevant to navigation."
        f"<|im_end|>\n<|im_start|>assistant\n"
    )

    inputs = processor(
        text=prompt,
        images=images,
        return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        ids = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            temperature=1.0
        )

    caption = processor.decode(
        ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    print(f"Caption: {caption}")
    print()

print("Done — 10 videos complete.")
