import torch
from PIL import Image
from transformers import AutoProcessor, SmolVLMForConditionalGeneration, Qwen2_5_VLForConditionalGeneration

DEVICE = "cuda:6"
IMAGE_PATHS = [
    "./eval_images/image_1.jpeg",
    "./eval_images/image_2.png",
    "./eval_images/image_3.png",
    "./eval_images/image_4.png",
]
PROMPT = "Describe this scene for a blind user."

# ─── SmolVLM2-500M base ───
print("=" * 60)
print("Loading SmolVLM2-500M-Video-Instruct (BASE)...")
smol_id = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
smol_proc = AutoProcessor.from_pretrained(smol_id)
smol_model = SmolVLMForConditionalGeneration.from_pretrained(smol_id, torch_dtype=torch.bfloat16).to(DEVICE)
smol_model.eval()

smol_captions = {}
for img_path in IMAGE_PATHS:
    image = Image.open(img_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    text = smol_proc.apply_chat_template(messages, add_generation_prompt=True)
    inputs = smol_proc(text=text, images=[image], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = smol_model.generate(**inputs, max_new_tokens=256, do_sample=False)
    caption = smol_proc.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    smol_captions[img_path] = caption
    print(f"\n[SmolVLM Base] {img_path}:\n{caption}")

del smol_model
torch.cuda.empty_cache()

# ─── Qwen2.5-VL-3B-Instruct ───
print("\n" + "=" * 60)
print("Loading Qwen2.5-VL-3B-Instruct...")
qwen_id = "Qwen/Qwen2.5-VL-3B-Instruct"
qwen_proc = AutoProcessor.from_pretrained(qwen_id)
qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(qwen_id, torch_dtype=torch.bfloat16).to(DEVICE)
qwen_model.eval()

qwen_captions = {}
for img_path in IMAGE_PATHS:
    image = Image.open(img_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    text = qwen_proc.apply_chat_template(messages, add_generation_prompt=True)
    inputs = qwen_proc(text=text, images=[image], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = qwen_model.generate(**inputs, max_new_tokens=256, do_sample=False)
    caption = qwen_proc.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    qwen_captions[img_path] = caption
    print(f"\n[Qwen 3B Instruct] {img_path}:\n{caption}")

del qwen_model
torch.cuda.empty_cache()

# ─── Side-by-side summary ───
print("\n" + "=" * 60)
print("SUMMARY — SIDE BY SIDE")
print("=" * 60)
for img_path in IMAGE_PATHS:
    print(f"\n📷 {img_path}")
    print(f"\n  [SmolVLM Base]\n  {smol_captions[img_path]}")
    print(f"\n  [Qwen 3B Instruct]\n  {qwen_captions[img_path]}")
    print("-" * 60)
