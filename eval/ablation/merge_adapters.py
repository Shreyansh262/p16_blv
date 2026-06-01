#!/usr/bin/env python3
"""
Merge SFT_v2 and GRPO LoRA adapters into full models.
Loads base model once, merges both adapters sequentially.
"""
import os, sys, torch
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from pathlib import Path
from transformers import AutoProcessor
from peft import PeftModel

BASE_MODEL  = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
ADAPTERS = [
    (
        "/usershome/cs671_user2/p16_blv/models/student/sft_v2/best",
        "/usershome/cs671_user2/p16_blv/models/student/sft_v2_merged"
    ),
    (
        "/usershome/cs671_user2/p16_blv/models/student/grpo/best",
        "/usershome/cs671_user2/p16_blv/models/student/grpo_merged"
    ),
]

# Verify adapter paths exist
for adapter_path, out_path in ADAPTERS:
    if not Path(adapter_path).exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")
    print(f"Adapter OK: {adapter_path}")

# Load base model once
print(f"\nLoading base model: {BASE_MODEL}")
from transformers import AutoModelForImageTextToText
base_model = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)
processor = AutoProcessor.from_pretrained(BASE_MODEL)
print("Base model loaded.")

# Merge each adapter
for adapter_path, out_path in ADAPTERS:
    print(f"\n{'='*50}")
    print(f"Merging: {adapter_path}")
    print(f"Output : {out_path}")

    # Load adapter on top of base
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        torch_dtype=torch.bfloat16
    )

    # Merge and unload LoRA weights
    print("Merging LoRA weights...")
    merged = model.merge_and_unload()

    # Save merged model
    Path(out_path).mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to {out_path}...")
    merged.save_pretrained(out_path)
    processor.save_pretrained(out_path)
    print(f"Saved. Size: {sum(f.stat().st_size for f in Path(out_path).rglob('*') if f.is_file())/1e9:.2f} GB")

    # Reload base cleanly for next merge
    # (merge_and_unload modifies in place — reload from saved)
    print("Reloading clean base for next adapter...")
    del model, merged
    torch.cuda.empty_cache()
    base_model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )

print("\nAll adapters merged successfully.")
print("sft_v2_merged :", "/usershome/cs671_user2/p16_blv/models/student/sft_v2_merged")
print("grpo_merged   :", "/usershome/cs671_user2/p16_blv/models/student/grpo_merged")
