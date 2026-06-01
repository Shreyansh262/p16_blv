import torch
import subprocess
from transformers import AutoProcessor, SmolVLMForConditionalGeneration
from peft import PeftModel

BASE_MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
GRPO_ADAPTER = "models/student/grpo/best"
PATCH_ADAPTER = "models/student/sft_patch_grpo_v2"
OUTPUT_DIR = "models/student/final_merged"

print("Step 1: Loading base model...")
model = SmolVLMForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="cpu"
)
processor = AutoProcessor.from_pretrained(GRPO_ADAPTER)

print("Step 2: Applying GRPO adapter and merging...")
model = PeftModel.from_pretrained(model, GRPO_ADAPTER)
model = model.merge_and_unload()

print("Step 3: Applying SFT_PATCH_V2 adapter and merging...")
model = PeftModel.from_pretrained(model, PATCH_ADAPTER)
model = model.merge_and_unload()

print("Step 4: Saving merged model...")
model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
processor.save_pretrained(OUTPUT_DIR)

print(f"Done. Saved to {OUTPUT_DIR}")
print("Total size:")
subprocess.run(["du", "-sh", OUTPUT_DIR])
