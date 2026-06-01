#!/usr/bin/env python3
"""
Inference script for Qwen2-VL and Qwen2.5-VL family.
Auto-detects model type from config.json.
Requires: pip install qwen-vl-utils
"""
import json, os, torch, argparse
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor
from datasets import load_from_disk

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--task", required=True, choices=["ocr", "vqa"])
parser.add_argument("--gpu", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "qwen-vl-utils", "--quiet"])
    from qwen_vl_utils import process_vision_info

if not Path(args.model).exists():
    raise FileNotFoundError(f"Model not found: {args.model}")

# Auto-detect model type from config
cfg = json.loads((Path(args.model) / "config.json").read_text())
model_type = cfg.get("model_type", "qwen2_vl")
print(f"Detected model_type: {model_type}")

if model_type == "qwen2_5_vl":
    from transformers import Qwen2_5_VLForConditionalGeneration as QwenCls
else:
    from transformers import Qwen2VLForConditionalGeneration as QwenCls

print(f"Loading {model_type} on GPU {args.gpu}...")
model = QwenCls.from_pretrained(
    args.model, torch_dtype=torch.bfloat16, device_map="cuda"
).eval()
processor = AutoProcessor.from_pretrained(args.model)
print("Model loaded.")

OUT = Path(args.out)
OUT.parent.mkdir(parents=True, exist_ok=True)

def run_qwen(image: Image.Image, question: str, max_new_tokens: int = 128) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": question}
        ]
    }]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs,
        videos=video_inputs, return_tensors="pt"
    ).to("cuda")
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False)
    trimmed = ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True
    )[0].strip()

if args.task == "ocr":
    ds = load_from_disk(
        "/usershome/cs671_user2/p16_blv/eval/ocr_bench/ocr_bench_test"
    )
    results = []
    for i, sample in enumerate(ds):
        gt = sample["answer"]
        if isinstance(gt, list): gt = gt[0]
        pred = run_qwen(sample["image"], sample["question"] +
                        " Answer concisely.")
        results.append({"id": i, "question": sample["question"],
                        "gt": gt, "pred": pred})
        if i % 100 == 0:
            print(f"[{i}/{len(ds)}] {pred[:60]}", flush=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"Saved {len(results)} -> {OUT}")

elif args.task == "vqa":
    ANN     = Path("/usershome/cs671_user2/p16_blv/eval/textvqa/TextVQA_0.5.1_val.json")
    IMG_DIR = Path("/usershome/cs671_user2/p16_blv/eval/textvqa/train_images")
    data    = json.loads(ANN.read_text())["data"]
    results = []
    skipped = 0
    for i, item in enumerate(data):
        img_path = None
        for ext in [".jpg", ".png", ""]:
            c = Path(str(IMG_DIR / item["image_id"]) + ext)
            if c.exists(): img_path = c; break
        if img_path is None:
            skipped += 1; continue
        img  = Image.open(img_path).convert("RGB")
        pred = run_qwen(img, item["question"] +
                        " Answer with a short phrase or single word.", 32)
        results.append({"question_id": item["question_id"],
                        "question": item["question"],
                        "pred": pred, "gts": item["answers"]})
        if i % 200 == 0:
            print(f"[{i}/{len(data)}] {pred[:50]}", flush=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"Saved {len(results)} (skipped {skipped}) -> {OUT}")
