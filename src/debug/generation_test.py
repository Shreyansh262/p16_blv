import torch, json, sys, os
os.chdir('/usershome/cs671_user2/p16_blv')
sys.path.insert(0, '/usershome/cs671_user2/p16_blv')

from transformers import AutoProcessor, SmolVLMForConditionalGeneration
from peft import PeftModel
from PIL import Image
from pathlib import Path

MODEL_BASE = 'HuggingFaceTB/SmolVLM2-500M-Video-Instruct'
CHECKPOINT = 'models/student/sft_val5_checkpoint'
CACHE = 'models/.hf_cache'
DATA = 'data/generated/qwen_captions.json'

print("Loading base model...")
processor = AutoProcessor.from_pretrained(MODEL_BASE, cache_dir=CACHE)
base_model = SmolVLMForConditionalGeneration.from_pretrained(
    MODEL_BASE, cache_dir=CACHE,
    torch_dtype=torch.bfloat16, device_map='cuda'
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, CHECKPOINT)
model.eval()

with open(DATA) as f:
    samples = json.load(f)
test_samples = samples[200:205]

print(f"\n{'='*60}")
print("GENERATION TEST - 5 samples")
print(f"{'='*60}")

for i, sample in enumerate(test_samples):
    keyframe_paths = sample.get('keyframe_paths', [])
    images = []
    for kp in keyframe_paths[:2]:
        # Try multiple possible paths
        candidates = [
            kp,
            os.path.join('data/keyframes/charades', os.path.basename(kp)),
            os.path.join('data/keyframes', os.path.basename(kp)),
        ]
        for c in candidates:
            if os.path.exists(c):
                try:
                    images.append(Image.open(c).convert('RGB'))
                except:
                    pass
                break

    if not images:
        print(f"\nSample {i+1}: SKIPPED (no keyframes at {keyframe_paths[:1]})")
        continue

    content = [{"type": "image"} for _ in images]
    content.append({"type": "text", "text": "Describe this video scene for a blind user."})
    messages = [{"role": "user", "content": content}]

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=images, return_tensors='pt').to(device='cuda', dtype=torch.bfloat16)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            repetition_penalty=1.2,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    new_tokens = out[0][inputs['input_ids'].shape[1]:]
    generated = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    teacher_ref = sample.get('blv_description', '')[:120]

    print(f"\n--- Sample {i+1} ---")
    print(f"GENERATED : {generated if generated else '<<EMPTY>>'}")
    print(f"TEACHER   : {teacher_ref}")
    print(f"STATUS    : {'NON-EMPTY' if generated else 'EMPTY'}")

print(f"\n{'='*60}")
print("Done.")
