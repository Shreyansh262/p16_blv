import json, random, sys, os, re, pathlib
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

BASE_DIR   = pathlib.Path("/usershome/cs671_user2/p16_blv")
BASE_MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
HF_CACHE   = str(BASE_DIR / "models" / ".hf_cache")
CHECKPOINT = str(BASE_DIR / "models" / "student" / "sft_single_stage" / "best")
DATA_PATH  = str(BASE_DIR / "data" / "generated" / "all_captions_gemma.json")
SEED       = 42
N_SAMPLES  = 10

FEW_SHOT_PROMPT = (
    "You are an audio description system for blind and low vision users.\n\n"
    "Examples of correct output:\n"
    "Example 1: CAUTION: Kitchen environment, wet floor 1 meter ahead. "
    "Man in blue shirt moves left at 2 meters. Child stands center at 1.5 meters. Path right is clear.\n"
    "Example 2: Living room. Person in white shirt seated 2 meters ahead center. "
    "Coffee table at 1 meter right -- potential trip hazard. Path left is clear.\n\n"
    "Now describe this scene following the exact same format:"
)

GEN_KWARGS = dict(
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7,
    repetition_penalty=1.3,
)

DIRECTION_WORDS = re.compile(r'\b(left|right|center|ahead|behind)\b', re.IGNORECASE)
METER_DIST      = re.compile(r'\d+(\.\d+)?\s*meter', re.IGNORECASE)

def score(text):
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    return {
        "CAUTION_present":  "CAUTION" in text.upper(),
        "direction_words":  len(DIRECTION_WORDS.findall(text)),
        "meter_distances":  bool(METER_DIST.search(text)),
        "sentence_count":   len(sentences),
    }

def load_images(keyframe_paths):
    imgs = []
    for p in keyframe_paths:
        try:
            imgs.append(Image.open(p).convert("RGB"))
        except Exception:
            pass
    return imgs

def run_inference(model, processor, images, prompt_text):
    messages = [{"role": "user", "content": (
        [{"type": "image", "image": img} for img in images] +
        [{"type": "text",  "text":  prompt_text}]
    )}]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=prompt, images=images, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, **GEN_KWARGS)
    n_in = inputs["input_ids"].shape[1]
    return processor.decode(out[0][n_in:], skip_special_tokens=True).strip()

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading processor ...", flush=True)
processor = AutoProcessor.from_pretrained(CHECKPOINT, cache_dir=HF_CACHE)

print("Loading base model ...", flush=True)
base = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL, dtype=torch.bfloat16, cache_dir=HF_CACHE,
)
print("Applying LoRA ...", flush=True)
model = PeftModel.from_pretrained(base, CHECKPOINT)
model = model.merge_and_unload()
model.eval().to("cuda")
print(f"Ready on {model.device} ({model.dtype})\n", flush=True)

# ── Data ──────────────────────────────────────────────────────────────────────
data = json.load(open(DATA_PATH))
test_split = data[7717:]
random.seed(SEED)
samples = random.sample(test_split, N_SAMPLES)
print(f"Sampled {N_SAMPLES} from test split ({len(test_split)} total, seed={SEED})\n", flush=True)

SEP = "=" * 72
scores_all = []

# ── Inference ─────────────────────────────────────────────────────────────────
for i, sample in enumerate(samples):
    vid    = sample["video_id"]
    ref    = sample.get("blv_description", sample.get("original_caption", "N/A"))
    images = load_images(sample.get("keyframe_paths", []))

    print(SEP, flush=True)
    print(f"SAMPLE {i+1:2d}/10  |  video_id: {vid}  |  frames: {len(images)}", flush=True)
    print(f"\n[GEMMA REFERENCE]\n{ref}", flush=True)

    if not images:
        print("[SKIP] No valid keyframes.\n", flush=True)
        scores_all.append({"vid": vid, **score("")})
        continue

    output = run_inference(model, processor, images, FEW_SHOT_PROMPT)
    s = score(output)
    scores_all.append({"vid": vid, **s})

    print(f"\n[FEW-SHOT OUTPUT]\n{output}", flush=True)
    print(f"\n[SCORES] CAUTION={s['CAUTION_present']}  "
          f"directions={s['direction_words']}  "
          f"meters={s['meter_distances']}  "
          f"sentences={s['sentence_count']}", flush=True)

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{SEP}", flush=True)
print("SUMMARY TABLE", flush=True)
print(f"{'#':<4} {'video_id':<14} {'CAUTION':<9} {'directions':<12} {'meters':<8} {'sentences'}", flush=True)
print("-" * 60, flush=True)
for i, s in enumerate(scores_all):
    print(f"{i+1:<4} {str(s['vid']):<14} {str(s['CAUTION_present']):<9} "
          f"{s['direction_words']:<12} {str(s['meter_distances']):<8} {s['sentence_count']}", flush=True)

# Aggregate
n = len(scores_all)
caution_rate  = sum(s["CAUTION_present"]  for s in scores_all) / n
dir_avg       = sum(s["direction_words"]  for s in scores_all) / n
meter_rate    = sum(s["meter_distances"]  for s in scores_all) / n
sent_avg      = sum(s["sentence_count"]   for s in scores_all) / n
print("-" * 60, flush=True)
print(f"{'AVG':<4} {'':14} {caution_rate:<9.0%} {dir_avg:<12.1f} {meter_rate:<8.0%} {sent_avg:.1f}", flush=True)
print(SEP, flush=True)
print("Done.", flush=True)
