#!/usr/bin/env python3
"""
Inference script for InternVL2.5-2B.
"""
import json, os, torch, argparse
from pathlib import Path
from PIL import Image
from transformers import AutoTokenizer, AutoModel
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from datasets import load_from_disk

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--task", required=True, choices=["ocr", "vqa"])
parser.add_argument("--gpu", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB")),
        T.Resize((input_size, input_size),
                 interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def load_image(image, input_size=448):
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    transform = build_transform(input_size)
    pixel_values = transform(image).unsqueeze(0)
    return pixel_values

if not Path(args.model).exists():
    raise FileNotFoundError(f"Model not found: {args.model}")

print(f"Loading InternVL2.5-2B on GPU {args.gpu}...")
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

def run_internvl(image: Image.Image, question: str,
                 max_new_tokens: int = 128) -> str:
    pixel_values = load_image(image).to(torch.bfloat16).cuda()
    prompt = f"<image>\n{question}"
    gc = dict(max_new_tokens=max_new_tokens, do_sample=False)
    response = model.chat(tokenizer, pixel_values, prompt, gc)
    return response.strip()

if args.task == "ocr":
    ds = load_from_disk(
        "/usershome/cs671_user2/p16_blv/eval/ocr_bench/ocr_bench_test"
    )
    results = []
    for i, sample in enumerate(ds):
        gt = sample["answer"]
        if isinstance(gt, list): gt = gt[0]
        pred = run_internvl(sample["image"],
                            sample["question"] + " Answer concisely.")
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
        pred = run_internvl(img, item["question"] +
                            " Answer with a short phrase or single word.", 32)
        results.append({"question_id": item["question_id"],
                        "question": item["question"],
                        "pred": pred, "gts": item["answers"]})
        if i % 200 == 0:
            print(f"[{i}/{len(data)}] {pred[:50]}", flush=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"Saved {len(results)} (skipped {skipped}) -> {OUT}")
