"""Quick test: figure out how to get description into tokenized output."""
import os, yaml
from pathlib import Path
from PIL import Image

with open(Path.home() / "p16_blv/config/paths_config.yaml") as f:
    cfg = yaml.safe_load(f)
os.environ["HF_HOME"] = cfg["cache"]["hf_cache"]

from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained(
    cfg["models"]["student_id"], cache_dir=cfg["cache"]["hf_cache"], trust_remote_code=True
)

img = Image.new("RGB", (224, 224), color=(128, 128, 128))
desc = "A person walks across a bright room."

print("=" * 60)
print("TEST 1: assistant content as list-of-dicts")
print("=" * 60)
msgs1 = [
    {"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": "Describe this video for a blind user."}
    ]},
    {"role": "assistant", "content": [
        {"type": "text", "text": desc}
    ]}
]
try:
    t1 = processor.apply_chat_template(msgs1, tokenize=False, add_generation_prompt=False)
    print(repr(t1))
except Exception as e:
    print(f"ERROR: {e}")

print("\n" + "=" * 60)
print("TEST 2: user-only + add_generation_prompt=True, then manual append")
print("=" * 60)
msgs2 = [
    {"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": "Describe this video for a blind user."}
    ]}
]
t2 = processor.apply_chat_template(msgs2, tokenize=False, add_generation_prompt=True)
print(f"Prompt part:\n  {repr(t2)}")
full = t2 + desc + "<end_of_utterance>\n"
print(f"Full with appended desc:\n  {repr(full)}")

print("\n" + "=" * 60)
print("TEST 3: Tokenize TEST 2 and verify description is present")
print("=" * 60)
inputs = processor(text=[full], images=[img], return_tensors="pt")
input_ids = inputs["input_ids"].squeeze(0)
decoded = processor.tokenizer.decode(input_ids)
print(f"Total tokens: {len(input_ids)}")
# Find where the prompt ends
prompt_inputs = processor(text=[t2], images=[img], return_tensors="pt")
prompt_len = prompt_inputs["input_ids"].shape[1]
print(f"Prompt tokens: {prompt_len}")
print(f"Response tokens: {len(input_ids) - prompt_len}")
resp_text = processor.tokenizer.decode(input_ids[prompt_len:])
print(f"Response decoded: {repr(resp_text)}")
