#!/usr/bin/env python3
"""
Merge SFT v2 LoRA adapter into base model weights.
Reads:  models/student/sft_v2/best   (LoRA adapter -- NOT modified)
Writes: models/student/sft_v2_merged  (full merged HF model)
Runs on CPU -- no GPU required.
"""
import sys, torch
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent.parent.parent
BASE_MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
ADAPTER    = BASE_DIR / "models/student/sft_v2/best"
OUT        = BASE_DIR / "models/student/sft_v2_merged"

def main():
    from transformers import SmolVLMForConditionalGeneration, AutoProcessor
    from peft import PeftModel

    if OUT.exists() and any(OUT.glob("*.safetensors")):
        print(f"[skip] {OUT} already exists -- delete it to re-merge.")
        sys.exit(0)

    print(f"Loading base model from cache: {BASE_MODEL}")
    model = SmolVLMForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, device_map="cpu"
    )
    processor = AutoProcessor.from_pretrained(BASE_MODEL)

    print(f"Attaching LoRA adapter: {ADAPTER}")
    model = PeftModel.from_pretrained(model, str(ADAPTER))

    print("Merging LoRA weights into base ...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT, safe_serialization=True)
    processor.save_pretrained(OUT)

    size_gb = sum(f.stat().st_size for f in OUT.glob("*.safetensors")) / 1e9
    print(f"Done. Model size on disk: {size_gb:.2f} GB")
    print(f"Merged model ready at: {OUT}")

if __name__ == "__main__":
    main()
