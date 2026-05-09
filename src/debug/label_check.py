"""Diagnostic: inspect tokenization and label masking for SFT."""
import os, sys, json, yaml
from pathlib import Path
from PIL import Image
import torch

project_root = Path.home() / "p16_blv"
sys.path.insert(0, str(project_root / "src"))

# Load config
with open(project_root / "config" / "paths_config.yaml") as f:
    cfg = yaml.safe_load(f)

os.environ["HF_HOME"] = cfg["cache"]["hf_cache"]

from transformers import AutoProcessor, SmolVLMForConditionalGeneration

model_id = cfg["models"]["student_id"]
hf_cache = cfg["cache"]["hf_cache"]
processor = AutoProcessor.from_pretrained(model_id, cache_dir=hf_cache, trust_remote_code=True)
tokenizer = processor.tokenizer

print("=" * 60)
print("STEP 1: Check special tokens")
print("=" * 60)

for tok_name in ["<|im_start|>", "<|im_end|>", "<end_of_utterance>",
                 "assistant", "Assistant", "<|assistant|>", "<bos>", "<eos>"]:
    tok_id = tokenizer.convert_tokens_to_ids(tok_name)
    is_unk = (tok_id == tokenizer.unk_token_id)
    print(f"  {tok_name:25s} -> id={tok_id:6d} {'(UNK!)' if is_unk else ''}")

print(f"\n  unk_token_id = {tokenizer.unk_token_id}")
print(f"  pad_token_id = {tokenizer.pad_token_id}")
print(f"  eos_token_id = {tokenizer.eos_token_id}")
print(f"  bos_token_id = {tokenizer.bos_token_id}")

print("\n" + "=" * 60)
print("STEP 2: Chat template output (raw text)")
print("=" * 60)

# Create a dummy sample
dummy_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
messages = [
    {"role": "user", "content": [
        {"type": "image", "image": dummy_img},
        {"type": "text", "text": "Describe this video for a blind user."}
    ]},
    {"role": "assistant", "content": "A person walks across a bright room."}
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
print(repr(text))

print("\n" + "=" * 60)
print("STEP 3: Tokenize and check label masking")
print("=" * 60)

inputs = processor(text=[text], images=[dummy_img], return_tensors="pt")
input_ids = inputs["input_ids"].squeeze(0)
print(f"\n  Total tokens: {len(input_ids)}")

# Reproduce the masking logic from 06_sft_train.py
sep = tokenizer.convert_tokens_to_ids("<|im_start|>")
print(f"  sep token (<|im_start|>) id: {sep}")
print(f"  Is UNK? {sep == tokenizer.unk_token_id}")

matches = (input_ids == sep).nonzero()
print(f"  Occurrences of sep in input_ids: {len(matches)}")
for i, m in enumerate(matches):
    pos = m.item()
    context = tokenizer.decode(input_ids[max(0,pos-2):pos+5])
    print(f"    match {i}: position {pos}, context: {repr(context)}")

labels = input_ids.clone()
if len(matches) >= 2:
    mask_until = matches[-1].item()
    labels[:mask_until] = -100
    print(f"\n  Masking labels[:{ mask_until}] = -100")
else:
    print("\n  WARNING: fewer than 2 matches -- NO MASKING applied!")

# Show label distribution
n_masked = (labels == -100).sum().item()
n_total = len(labels)
n_train = n_total - n_masked
print(f"\n  Labels: {n_masked} masked (-100) + {n_train} trainable = {n_total} total")

# Decode what the model actually trains on
train_ids = labels[labels != -100]
train_text = tokenizer.decode(train_ids)
print(f"\n  Trainable token text:\n    {repr(train_text[:300])}")

# Also decode the FULL sequence for reference
full_text = tokenizer.decode(input_ids)
print(f"\n  Full decoded sequence:\n    {repr(full_text[:500])}")

print("\n" + "=" * 60)
print("STEP 4: Where is the assistant response in the token stream?")
print("=" * 60)

# Find the description text
desc = "A person walks across a bright room."
desc_ids = tokenizer.encode(desc, add_special_tokens=False)
print(f"  Description token ids: {desc_ids}")
print(f"  Description tokens: {[tokenizer.decode([t]) for t in desc_ids]}")

# Search for description in input_ids
input_list = input_ids.tolist()
for start_pos in range(len(input_list) - len(desc_ids) + 1):
    if input_list[start_pos:start_pos+len(desc_ids)] == desc_ids:
        print(f"  Found description at positions {start_pos}-{start_pos+len(desc_ids)-1}")
        corresponding_labels = labels[start_pos:start_pos+len(desc_ids)]
        all_masked = (corresponding_labels == -100).all().item()
        print(f"  Labels at those positions all -100? {all_masked}")
        if all_masked:
            print("  BUG CONFIRMED: description tokens are masked -- model cannot learn them!")
        break
else:
    print("  WARNING: description tokens not found as contiguous subsequence")
    # Try to find approximately
    desc_first_tok = desc_ids[0]
    positions = (input_ids == desc_first_tok).nonzero()
    print(f"  First description token ({tokenizer.decode([desc_first_tok])}) found at: {[p.item() for p in positions]}")
