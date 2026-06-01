#!/usr/bin/env python3
"""OCR-Bench inference: SmolVLM2-500M sft_patch_grpo_v2"""
import json, os, torch
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoProcessor, AutoModelForImageTextToText

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
CKPT = "/usershome/cs671_user2/p16_blv/models/student/sft_patch_grpo_v2"
OUT  = Path("ocr_bench_preds.json")

if not Path(CKPT).exists():
    raise FileNotFoundError(f"Checkpoint not found: {CKPT}")

processor = AutoProcessor.from_pretrained(CKPT)
model = AutoModelForImageTextToText.from_pretrained(
    CKPT, torch_dtype=torch.bfloat16, device_map="cuda"
).eval()

ds = load_from_disk("./ocr_bench_test")
results = []
for i, sample in enumerate(ds):
    img   = sample["image"]
    quest = sample["question"]
    # answer is a list — take first element as ground truth
    gt    = sample["answer"][0] if isinstance(sample["answer"], list) else sample["answer"]
    qtype = sample.get("question_type", "")

    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": quest + " Answer concisely."}
    ]}]
    text_in = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs  = processor(text=text_in, images=img, return_tensors="pt").to("cuda")
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    pred = processor.decode(ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    results.append({"id": i, "question_type": qtype, "question": quest, "gt": gt, "pred": pred})
    if i % 100 == 0:
        print(f"[{i}/{len(ds)}] {qtype} | pred={pred[:60]}", flush=True)

OUT.write_text(json.dumps(results, indent=2))
print(f"Saved {len(results)} predictions -> {OUT}")
