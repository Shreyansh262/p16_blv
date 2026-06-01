#!/usr/bin/env python3
"""
Inference script for Phi-3.5-vision-instruct.
Uses AutoModelForCausalLM with trust_remote_code=True.
Flash attention disabled for compatibility.
"""
import json, os, torch, argparse
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from datasets import load_from_disk

# Compatibility shims for DynamicCache API changes in transformers 5.x
try:
    from transformers.cache_utils import DynamicCache
    if not hasattr(DynamicCache, "get_usable_length"):
        def _get_usable_length(self, new_seq_len, layer_idx=None):
            try:
                return self.get_seq_length() if layer_idx is None else self.get_seq_length(layer_idx)
            except Exception:
                return self.get_seq_length()
        DynamicCache.get_usable_length = _get_usable_length
    if not hasattr(DynamicCache, "get_max_length"):
        DynamicCache.get_max_length = lambda self: None
    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
except Exception:
    pass

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--task", required=True, choices=["ocr", "vqa"])
parser.add_argument("--gpu", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

if not Path(args.model).exists():
    raise FileNotFoundError(f"Model not found: {args.model}")

print(f"Loading Phi-3.5-vision on GPU {args.gpu}...")
processor = AutoProcessor.from_pretrained(
    args.model, trust_remote_code=True, num_crops=4
)
model = AutoModelForCausalLM.from_pretrained(
    args.model, torch_dtype=torch.bfloat16,
    device_map="cuda", trust_remote_code=True,
    _attn_implementation="eager"
).eval()
print("Model loaded.")

OUT = Path(args.out)
OUT.parent.mkdir(parents=True, exist_ok=True)

def run_phi(image: Image.Image, question: str,
            max_new_tokens: int = 128) -> str:
    placeholder = "<|image_1|>\n"
    messages = [{"role": "user", "content": placeholder + question}]
    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(prompt, [image], return_tensors="pt").to("cuda")
    gen_cfg = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=processor.tokenizer.eos_token_id,
    )
    with torch.no_grad():
        ids = model.generate(**inputs, **gen_cfg)
    trimmed = ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

if args.task == "ocr":
    ds = load_from_disk(
        "/usershome/cs671_user2/p16_blv/eval/ocr_bench/ocr_bench_test"
    )
    results = []
    for i, sample in enumerate(ds):
        gt = sample["answer"]
        if isinstance(gt, list): gt = gt[0]
        pred = run_phi(sample["image"], sample["question"] + " Answer concisely.")
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
        pred = run_phi(img, item["question"] +
                       " Answer with a short phrase or single word.", 32)
        results.append({"question_id": item["question_id"],
                        "question": item["question"],
                        "pred": pred, "gts": item["answers"]})
        if i % 200 == 0:
            print(f"[{i}/{len(data)}] {pred[:50]}", flush=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"Saved {len(results)} (skipped {skipped}) -> {OUT}")
