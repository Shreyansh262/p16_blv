"""Verify the patched __getitem__ produces correct labels."""
import os, sys, json, yaml
from pathlib import Path
from PIL import Image
import torch

project_root = Path.home() / "p16_blv"
sys.path.insert(0, str(project_root / "src"))

with open(project_root / "config" / "paths_config.yaml") as f:
    cfg = yaml.safe_load(f)
os.environ["HF_HOME"] = cfg["cache"]["hf_cache"]

from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained(
    cfg["models"]["student_id"], cache_dir=cfg["cache"]["hf_cache"], trust_remote_code=True
)
tokenizer = processor.tokenizer

# Simulate the patched __getitem__
img = Image.new("RGB", (224, 224), color=(128, 128, 128))
desc = "A person walks across a bright room near a large window."

user_messages = [
    {"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": "Describe this video for a blind user."}
    ]}
]

prompt_text = processor.apply_chat_template(
    user_messages, tokenize=False, add_generation_prompt=True
)
full_text = prompt_text + " " + desc + "<end_of_utterance>\n"

prompt_inputs = processor(text=[prompt_text], images=[img], return_tensors="pt")
prompt_len = prompt_inputs["input_ids"].shape[1]

full_inputs = processor(text=[full_text], images=[img], return_tensors="pt")
input_ids = full_inputs["input_ids"].squeeze(0)
labels = input_ids.clone()
labels[:prompt_len] = -100

n_masked = (labels == -100).sum().item()
n_train = len(labels) - n_masked
print("=" * 60)
print("PATCHED LABEL CHECK")
print("=" * 60)
print(f"Total tokens:     {len(input_ids)}")
print(f"Prompt tokens:    {prompt_len} (masked)")
print(f"Response tokens:  {n_train} (trainable)")
print(f"Ratio:            {n_train/len(input_ids)*100:.1f}% trainable")

train_ids = labels[labels != -100]
train_text = tokenizer.decode(train_ids)
print(f"\nTrainable text: {repr(train_text)}")

masked_text = tokenizer.decode(input_ids[:prompt_len])
print(f"\nMasked text (last 80 chars): ...{repr(masked_text[-80:])}")

# Verify description is in trainable portion
assert desc in train_text, "BUG: description not in trainable tokens!"
print("\nVERIFIED: description is in trainable portion")

# Check that <end_of_utterance> is trainable (model must learn when to stop)
eou_id = tokenizer.convert_tokens_to_ids("<end_of_utterance>")
eou_positions = (input_ids == eou_id).nonzero().flatten().tolist()
eou_in_labels = [pos for pos in eou_positions if labels[pos].item() != -100]
print(f"<end_of_utterance> positions: {eou_positions}")
print(f"<end_of_utterance> trainable: {eou_in_labels}")
if eou_in_labels:
    print("VERIFIED: model will learn to produce <end_of_utterance> (stop token)")
else:
    print("WARNING: <end_of_utterance> is masked — model won't learn to stop!")
