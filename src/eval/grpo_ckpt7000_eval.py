import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # set before torch init
os.chdir("/usershome/cs671_user2/p16_blv")

import torch, json
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

FEW_SHOT_PROMPT = (
    "You are an audio description system for blind and low vision users.\n\n"
    "Examples of correct output:\n"
    "Example 1: CAUTION: Kitchen environment, wet floor 1 meter ahead. "
    "Man in blue shirt moves left at 2 meters. Child stands center at 1.5 meters. "
    "Path right is clear.\n"
    "Example 2: Living room. Person in white shirt seated 2 meters ahead center. "
    "Coffee table at 1 meter right — potential trip hazard. Path left is clear.\n\n"
    "Now describe this scene following the exact same format:"
)

BASE_MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
SFT_PATH   = "models/student/sft_v2/best"
GRPO_CKPT  = "models/student/grpo/checkpoint-7000"

print(f"GRPO checkpoint: {GRPO_CKPT}")
print(f"GPU: {os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)

import random
random.seed(99)
with open("data/generated/all_captions_gemma.json") as f:
    all_data = json.load(f)
valid = []
for entry in all_data:
    paths = entry.get("keyframe_paths", [])
    if paths and all(os.path.exists(p) for p in paths[:4]):
        valid.append(entry)
    if len(valid) == 3:
        break

print(f"Videos: {[e['video_id'] for e in valid]}", flush=True)

def load_images(paths):
    return [Image.open(p).convert("RGB") for p in paths[:4]]

def run_inference(model, processor, images, label):
    n = len(images)
    messages = [{"role": "user", "content":
                 [{"type": "image"}] * n +
                 [{"type": "text", "text": FEW_SHOT_PROMPT}]}]
    text_in = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs  = processor(text=text_in, images=images, return_tensors="pt")
    inputs  = {k: v.to("cuda:0") for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
        )
    decoded = processor.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
    print(f"\n--- {label} ---")
    print(decoded)

print("Loading processor ...", flush=True)
processor = AutoProcessor.from_pretrained(BASE_MODEL)

print("Loading SFT v2 ...", flush=True)
base = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL, dtype=torch.bfloat16, device_map={"": "cuda:0"}
)
sft_model = PeftModel.from_pretrained(base, SFT_PATH)
sft_model = sft_model.merge_and_unload()
sft_model.eval()
print("SFT v2 ready.", flush=True)

print(f"Loading GRPO {GRPO_CKPT} ...", flush=True)
base2 = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL, dtype=torch.bfloat16, device_map={"": "cuda:0"}
)
grpo_model = PeftModel.from_pretrained(base2, GRPO_CKPT)
grpo_model = grpo_model.merge_and_unload()
grpo_model.eval()
print("GRPO ready.", flush=True)

SEP = "=" * 80

for i, entry in enumerate(valid):
    vid    = entry["video_id"]
    ref    = entry["blv_description"]
    frames = load_images(entry["keyframe_paths"][:4])

    print(f"\n{SEP}")
    print(f"VIDEO {i+1} of 3  |  ID: {vid}")
    print(SEP)
    print("\n--- GEMMA REFERENCE ---")
    print(ref)

    run_inference(sft_model,  processor, frames, "SFT_V2")
    run_inference(grpo_model, processor, frames, "GRPO ckpt-7000")

    print(f"\n{SEP}\n")

print("=== ALL 3 VIDEOS DONE ===")
