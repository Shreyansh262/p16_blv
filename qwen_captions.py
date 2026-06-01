import os, torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

IMAGE_PATHS = [
    "./eval_images/image_1.jpeg",
    "./eval_images/image_2.png",
    "./eval_images/image_3.png",
    "./eval_images/image_4.png",
]
PROMPT = "Describe this scene for a blind user."

print("=" * 60)
print("Loading Qwen2.5-VL-3B-Instruct on GPU 7...")
qwen_id = "Qwen/Qwen2.5-VL-3B-Instruct"
proc  = AutoProcessor.from_pretrained(qwen_id)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    qwen_id, torch_dtype=torch.bfloat16, device_map="cuda"
).eval()
print("Model loaded.\n")

for img_path in IMAGE_PATHS:
    image = Image.open(img_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    text = proc.apply_chat_template(messages, add_generation_prompt=True)
    inputs = proc(text=text, images=[image], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    caption = proc.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\n[Qwen2.5-VL-3B] {img_path}:\n{caption}")

print("\nDone.")
