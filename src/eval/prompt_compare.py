"""
Prompt comparison: BLV_PROMPT (old) vs BLV_PROMPT_V2 (new structured)
Checkpoint: models/student/sft_single_stage/best/
5 random test samples from all_captions_gemma.json[7717:]
GPU 6 only.
"""
import json, random, sys, os, pathlib
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

BASE_DIR      = pathlib.Path("/usershome/cs671_user2/p16_blv")
BASE_MODEL    = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
HF_CACHE      = str(BASE_DIR / "models" / ".hf_cache")
CHECKPOINT    = str(BASE_DIR / "models" / "student" / "sft_single_stage" / "best")
DATA_PATH     = str(BASE_DIR / "data" / "generated" / "all_captions_gemma.json")
SEED          = 99
N_SAMPLES     = 5

BLV_PROMPT = (
    "You are an audio description system for blind and low vision users. "
    "Follow these rules strictly: "
    "1. First sentence: name the environment and say CAUTION if any hazard is within 2 meters. "
    "2. Every object and person must have a direction (left/right/center/ahead) and distance in meters. "
    "3. People: describe by clothing color, position, and movement direction only. "
    "4. Present tense, active voice. Maximum 4 sentences. Every word must serve navigation. "
    "Describe this video scene for a blind user."
)

BLV_PROMPT_V2 = (
    "You are an audio description system for blind users. You MUST follow this exact structure:\n"
    "Sentence 1: State the environment. If ANY hazard exists within 2 meters, begin with 'CAUTION:' before anything else.\n"
    "Sentence 2-3: Every object/person MUST have direction (left/right/center/ahead) and distance in meters. "
    "People: clothing color + movement only.\n"
    "Sentence 4: State whether path ahead is clear or blocked.\n"
    "Present tense. Active voice. Maximum 4 sentences.\n\n"
    "Example output: 'CAUTION: Kitchen near stove. Hot surface at 1 meter ahead center. "
    "Person in blue shirt moving left at 2 meters. Path ahead is clear.'\n\n"
    "Now describe this video scene for a blind user:"
)

GEN_KWARGS = dict(
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7,
    repetition_penalty=1.3,
)

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
    n_input = inputs["input_ids"].shape[1]
    generated = out[0][n_input:]
    return processor.decode(generated, skip_special_tokens=True).strip()

# ── Load model ────────────────────────────────────────────────────────────────
print(f"Loading processor from {CHECKPOINT} ...", flush=True)
processor = AutoProcessor.from_pretrained(CHECKPOINT, cache_dir=HF_CACHE)

print(f"Loading base model {BASE_MODEL} ...", flush=True)
base = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    cache_dir=HF_CACHE,
)

print(f"Applying LoRA from {CHECKPOINT} ...", flush=True)
model = PeftModel.from_pretrained(base, CHECKPOINT)
model = model.merge_and_unload()
model.eval()
model = model.to("cuda")
print(f"Model on {model.device}, dtype={model.dtype}", flush=True)

# ── Load data ─────────────────────────────────────────────────────────────────
data = json.load(open(DATA_PATH))
test_split = data[7717:]
random.seed(SEED)
samples = random.sample(test_split, N_SAMPLES)
print(f"\nSampled {N_SAMPLES} items from test split ({len(test_split)} total, seed={SEED})\n", flush=True)

SEP = "=" * 80

# ── Run inference ─────────────────────────────────────────────────────────────
for i, sample in enumerate(samples):
    vid     = sample["video_id"]
    ref     = sample.get("blv_description", sample.get("original_caption", "N/A"))
    kfps    = sample.get("keyframe_paths", [])
    images  = load_images(kfps)

    print(SEP, flush=True)
    print(f"SAMPLE {i+1}/{N_SAMPLES}  |  video_id: {vid}", flush=True)
    print(f"Keyframes loaded: {len(images)}/{len(kfps)}", flush=True)
    print(f"\n[GEMMA REFERENCE]\n{ref}", flush=True)

    if not images:
        print("[SKIP] No valid keyframes.", flush=True)
        continue

    out_old = run_inference(model, processor, images, BLV_PROMPT)
    print(f"\n[OLD BLV_PROMPT output]\n{out_old}", flush=True)

    out_new = run_inference(model, processor, images, BLV_PROMPT_V2)
    print(f"\n[NEW BLV_PROMPT_V2 output]\n{out_new}", flush=True)

print(SEP, flush=True)
print("Done.", flush=True)
