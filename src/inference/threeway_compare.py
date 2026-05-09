#!/usr/bin/env python3
"""
threeway_compare.py  (v2 - fixed SFT checkpoint)
3-way comparison: BASE SmolVLM2-500M vs SFT checkpoint vs Reference (teacher) caption.

Key fix from v1: checkpoint-489 collapsed (loss->0.0004, generates EOS immediately
with 4-frame inputs). Using checkpoint-171 instead which produces coherent output.

Usage:
    python scripts/threeway_compare.py [--gpu 6] [--seed 42] [--n 3]
    python scripts/threeway_compare.py --gpu 6 --seed 42 --n 3 --checkpoint 489
"""

import os
import sys
import json
import random
import argparse
import textwrap
from pathlib import Path

import torch
from PIL import Image

# -- Parse args early so we can set CUDA_VISIBLE_DEVICES ----------------------
parser = argparse.ArgumentParser()
parser.add_argument("--gpu",        type=int, default=6)
parser.add_argument("--seed",       type=int, default=42)
parser.add_argument("--n",          type=int, default=3, help="Number of videos to compare")
parser.add_argument("--checkpoint", type=int, default=171,
                    help="SFT checkpoint step to use (171 recommended; 489 collapsed)")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ["HF_HOME"] = "/usershome/cs671_user2/p16_blv/models/.hf_cache"

from transformers import AutoProcessor, SmolVLMForConditionalGeneration
from peft import PeftModel

# -- Configuration -------------------------------------------------------------
DEVICE           = "cuda:0"
BASE_MODEL_ID    = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
HF_CACHE         = "/usershome/cs671_user2/p16_blv/models/.hf_cache"
SFT_CHECKPOINT   = (f"/usershome/cs671_user2/p16_blv/models/student/"
                    f"sft_validation_checkpoint/checkpoint-{args.checkpoint}")
CAPTIONS_FILE    = "/usershome/cs671_user2/p16_blv/data/generated/all_captions.json"
MAX_SIDE         = 364
MAX_NEW_TOKENS   = 256

PROMPT = "Describe this video for a blind user."

# -- Helpers -------------------------------------------------------------------

def load_images(keyframe_paths):
    """Load and resize keyframe images."""
    images = []
    for p in keyframe_paths:
        path = Path(p)
        if path.exists():
            try:
                img = Image.open(path).convert("RGB")
                if max(img.size) > MAX_SIDE:
                    img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
                images.append(img)
            except Exception as e:
                print(f"  [WARN] Could not load {p}: {e}")
    return images


def build_messages(images, prompt):
    """Build chat messages in SmolVLM2 format."""
    return [
        {
            "role": "user",
            "content": (
                [{"type": "image", "image": img} for img in images]
                + [{"type": "text", "text": prompt}]
            ),
        }
    ]


def run_inference(model, processor, images, prompt):
    """Run inference and return generated text."""
    messages = build_messages(images, prompt)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=images,
        return_tensors="pt",
    ).to(DEVICE)

    # Cast pixel_values to float16 to match model weights
    if "pixel_values" in inputs and inputs["pixel_values"] is not None:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    # Decode only the newly generated tokens
    n_input = inputs["input_ids"].shape[1]
    generated = output_ids[0][n_input:]
    result = processor.decode(generated, skip_special_tokens=True).strip()
    # Detect collapsed output (EOS immediately)
    if len(generated) <= 2:
        result = f"[COLLAPSED - generated only {len(generated)} token(s)]"
    return result


def wrap(text, width=90, indent="    "):
    return "\n".join(
        textwrap.fill(line, width=width, initial_indent=indent, subsequent_indent=indent)
        for line in text.splitlines()
    )


# -- Main ----------------------------------------------------------------------

def main():
    random.seed(args.seed)

    print("=" * 80)
    print("  3-WAY MODEL COMPARISON: BASE vs SFT vs REFERENCE")
    print("  SmolVLM2-500M -- BLV Pipeline (CS671)")
    print(f"  SFT checkpoint: checkpoint-{args.checkpoint}")
    print("=" * 80)
    print()

    # -- Load captions ---------------------------------------------------------
    print("[1/4] Loading captions JSON...")
    with open(CAPTIONS_FILE) as f:
        all_captions = json.load(f)
    print(f"      Total entries: {len(all_captions)}")

    # Filter to entries with existing keyframes
    valid = []
    for entry in all_captions:
        kf_paths = entry.get("keyframe_paths", [])
        existing = [p for p in kf_paths if Path(p).exists()]
        if existing and entry.get("blv_description", "").strip():
            entry["_existing_kf"] = existing
            valid.append(entry)
    print(f"      Entries with valid keyframes + reference: {len(valid)}")

    selected = random.sample(valid, min(args.n, len(valid)))
    print(f"      Selected {len(selected)} videos for comparison (seed={args.seed})")
    print()

    # -- Load processor (shared) -----------------------------------------------
    print("[2/4] Loading processor...")
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL_ID, cache_dir=HF_CACHE, trust_remote_code=True
    )
    print("      Processor ready.")
    print()

    # -- Load BASE model --------------------------------------------------------
    print("[3/4] Loading BASE model (no fine-tuning)...")
    base_model = SmolVLMForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map={"": 0},
        cache_dir=HF_CACHE,
    )
    base_model.eval()
    print("      BASE model ready.")
    print()

    # -- Run BASE inference ----------------------------------------------------
    print("[4/4] Running inference...")
    print()

    base_outputs = {}
    for entry in selected:
        vid = entry["video_id"]
        images = load_images(entry["_existing_kf"])
        if not images:
            base_outputs[vid] = "[ERROR: no images loaded]"
            continue
        print(f"  BASE -> {vid} ({len(images)} frames)...", end=" ", flush=True)
        out = run_inference(base_model, processor, images, PROMPT)
        base_outputs[vid] = out
        print("done")

    # -- Free BASE model, load SFT ---------------------------------------------
    print()
    print("  Freeing BASE model from GPU...")
    del base_model
    torch.cuda.empty_cache()

    print(f"  Loading SFT model (BASE + LoRA checkpoint-{args.checkpoint})...")
    sft_base = SmolVLMForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map={"": 0},
        cache_dir=HF_CACHE,
    )
    sft_model = PeftModel.from_pretrained(
        sft_base,
        SFT_CHECKPOINT,
        torch_dtype=torch.float16,
    )
    sft_model.eval()
    print("  SFT model ready.")
    print()

    sft_outputs = {}
    for entry in selected:
        vid = entry["video_id"]
        images = load_images(entry["_existing_kf"])
        if not images:
            sft_outputs[vid] = "[ERROR: no images loaded]"
            continue
        print(f"  SFT  -> {vid} ({len(images)} frames)...", end=" ", flush=True)
        out = run_inference(sft_model, processor, images, PROMPT)
        sft_outputs[vid] = out
        print("done")

    del sft_model
    torch.cuda.empty_cache()

    # -- Print results ---------------------------------------------------------
    print()
    print("=" * 80)
    print("  RESULTS")
    print("=" * 80)

    for i, entry in enumerate(selected, 1):
        vid = entry["video_id"]
        dataset = entry.get("dataset", "?")
        ref = entry.get("blv_description", "").strip()
        orig = entry.get("original_caption", "").strip()

        print()
        print("-" * 80)
        print(f"  Video {i}/{len(selected)}: {vid}  [{dataset}]")
        print(f"  Keyframes: {len(entry['_existing_kf'])} frames")
        print("-" * 80)
        print()

        print("  ORIGINAL CAPTION (human annotation):")
        print(wrap(orig))
        print()

        print("  REFERENCE (teacher Qwen2-VL-7B, blv_description):")
        print(wrap(ref))
        print()

        print("  BASE SmolVLM2-500M (no fine-tuning):")
        print(wrap(base_outputs.get(vid, "[missing]")))
        print()

        print(f"  SFT SmolVLM2-500M (LoRA checkpoint-{args.checkpoint}):")
        print(wrap(sft_outputs.get(vid, "[missing]")))
        print()

    print("=" * 80)
    print("  COMPARISON COMPLETE")
    print("=" * 80)
    print()
    print("  DIAGNOSIS NOTE:")
    print("  checkpoint-489 generates empty output with 4-frame inputs.")
    print("  Root cause: training loss collapsed to ~0.0004 (overfitting/mode collapse).")
    print("  checkpoint-171 (mid-training) produces coherent output.")
    print("  Recommendation: re-run SFT with lower LR or add early stopping.")


if __name__ == "__main__":
    main()
