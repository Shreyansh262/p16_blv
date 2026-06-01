import os, json
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

import torch
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, SmolVLMForConditionalGeneration
from peft import PeftModel

PROJECT   = Path.home() / 'p16_blv'
MANIFEST  = PROJECT / 'data/keyframes/charades_luv_manifest.json'
PREV_OUT  = PROJECT / 'outputs/ten_video_eval.json'
CKPT      = PROJECT / 'models/student/sft_patch_grpo_v2'
BASE_ID   = 'HuggingFaceTB/SmolVLM2-500M-Video-Instruct'
OUT_FILE  = PROJECT / 'outputs/ten_video_eval_simpleprompt.json'
MAX_SIDE  = 364
PROMPT    = 'Describe this scene for a blind user'

# Get the exact 10 video IDs from previous run (preserves order)
with open(PREV_OUT) as f:
    prev = json.load(f)
video_ids = [e['video_id'] for e in prev]
print(f'Target video IDs: {video_ids}')

# Build keyframe_paths lookup from manifest
with open(MANIFEST) as f:
    manifest = json.load(f)
kf_lookup = {e['video_id']: e.get('keyframe_paths', []) for e in manifest}

def load_frames(video_id):
    frames = []
    for p in kf_lookup.get(video_id, []):
        path = Path(p)
        if path.exists():
            img = Image.open(path).convert('RGB')
            if max(img.size) > MAX_SIDE:
                img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
            frames.append(img)
    return frames

print('Loading processor + base model ...')
processor = AutoProcessor.from_pretrained(BASE_ID)
base = SmolVLMForConditionalGeneration.from_pretrained(
    BASE_ID, torch_dtype=torch.bfloat16, device_map={'': 0}
)
print(f'Applying LoRA from {CKPT} ...')
model = PeftModel.from_pretrained(base, str(CKPT))
model.eval()
print('Ready.\n' + '=' * 60)

results = []
for i, vid_id in enumerate(video_ids):
    frames = load_frames(vid_id)
    if not frames:
        print(f'[{i+1}/10] {vid_id}: NO KEYFRAMES — skipping')
        continue

    messages = [{'role': 'user', 'content':
        [{'type': 'image', 'image': img} for img in frames]
        + [{'type': 'text', 'text': PROMPT}]
    }]
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[prompt_text], images=frames, return_tensors='pt').to(model.device)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            repetition_penalty=1.3,
        )

    caption = processor.decode(
        out_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True
    ).strip()

    print(f'[{i+1}/10] {vid_id}')
    print(f'  {caption}\n')
    results.append({'video_id': vid_id, 'caption': caption})

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_FILE, 'w') as f:
    json.dump(results, f, indent=2)

print('=' * 60)
print('FINAL OUTPUT — 10 VIDEOS (simple prompt)')
print('=' * 60)
for r in results:
    print(f"\nVIDEO ID : {r['video_id']}")
    print(f"CAPTION  : {r['caption']}")
print(f"\nSaved {len(results)} results -> {OUT_FILE}")
