#!/usr/bin/env python3
"""
Inference script for MiniCPM-V-2.6.
Uses AutoModel with trust_remote_code=True and model.chat() interface.
"""
import json, os, torch, argparse
from pathlib import Path
from PIL import Image
from transformers import AutoTokenizer, AutoModel
from datasets import load_from_disk

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--task", required=True, choices=["ocr", "vqa"])
parser.add_argument("--gpu", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

if not Path(args.model).exists():
    raise FileNotFoundError(f"Model not found: {args.model}")

print(f"Loading MiniCPM-V-2.6 on GPU {args.gpu}...")
tokenizer = AutoTokenizer.from_pretrained(
    args.model, trust_remote_code=True
)
model = AutoModel.from_pretrained(
    args.model, torch_dtype=torch.bfloat16,
    device_map="cuda", trust_remote_code=True
).eval()
print("Model loaded.")

OUT = Path(args.out)
OUT.parent.mkdir(parents=True, exist_ok=True)

def run_minicpm(image: Image.Image, question: str,
                max_new_tokens: int = 128) -> str:
    msgs = [{"role": "user", "content": [image, question]}]
    res = model.chat(
        image=None,
        msgs=msgs,
        tokenizer=tokenizer,
        sampling=False,
        max_new_tokens=max_new_tokens,
    )
    return res.strip() if isinstance(res, str) else str(res).strip()

if args.task == "ocr":
    ds = load_from_disk(
        "/usershome/cs671_user2/p16_blv/eval/ocr_bench/ocr_bench_test"
    )
    results = []
    for i, sample in enumerate(ds):
        gt = sample["answer"]
        if isinstance(gt, list): gt = gt[0]
        pred = run_minicpm(sample["image"], sample["question"] + " Answer concisely.")
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
        pred = run_minicpm(img, item["question"] +
                           " Answer with a short phrase or single word.", 32)
        results.append({"question_id": item["question_id"],
                        "question": item["question"],
                        "pred": pred, "gts": item["answers"]})
        if i % 200 == 0:
            print(f"[{i}/{len(data)}] {pred[:50]}", flush=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"Saved {len(results)} (skipped {skipped}) -> {OUT}")
