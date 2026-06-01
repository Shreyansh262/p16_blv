#!/usr/bin/env python3
"""TextVQA val inference: SmolVLM2-500M sft_patch_grpo_v2"""
import json, os, torch
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
CKPT    = "/usershome/cs671_user2/p16_blv/models/student/sft_patch_grpo_v2"
ANN     = Path("TextVQA_0.5.1_val.json")
IMG_DIR = Path("train_images")
OUT     = Path("textvqa_preds_v2.json")

if not Path(CKPT).exists(): raise FileNotFoundError(f"Checkpoint not found: {CKPT}")
if not ANN.exists(): raise FileNotFoundError("Annotations not found. Run download_textvqa.sh first.")
if not IMG_DIR.exists(): raise FileNotFoundError("Images not found. Run download_textvqa.sh first.")

processor = AutoProcessor.from_pretrained(CKPT)
model = AutoModelForImageTextToText.from_pretrained(
    CKPT, torch_dtype=torch.bfloat16, device_map="cuda"
).eval()

data = json.loads(ANN.read_text())["data"]
results, skipped = [], 0
for i, item in enumerate(data):
    img_path = None
    for ext in [".jpg", ".png", ""]:
        c = IMG_DIR / (item["image_id"] + ext)
        if c.exists(): img_path = c; break
    if img_path is None: skipped += 1; continue
    img   = Image.open(img_path).convert("RGB")
    quest = item["question"]
    gts   = item["answers"]
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": quest + " Answer in as few words as possible. One word if sufficient. No punctuation."}
    ]}]
    text_in = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs  = processor(text=text_in, images=img, return_tensors="pt").to("cuda")
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    pred = processor.decode(ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    results.append({"question_id": item["question_id"], "question": quest, "pred": pred, "gts": gts})
    if i % 200 == 0: print(f"[{i}/{len(data)}]  pred={pred[:50]}", flush=True)

OUT.write_text(json.dumps(results, indent=2))
print(f"Saved {len(results)} predictions (skipped {skipped} missing images) -> {OUT}")
