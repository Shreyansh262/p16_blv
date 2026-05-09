import torch, json, os
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, SmolVLMForConditionalGeneration
from peft import PeftModel

BASE_DIR = Path('/usershome/cs671_user2/p16_blv')
HF_CACHE = str(BASE_DIR / 'models' / '.hf_cache')
BASE_MODEL = 'HuggingFaceTB/SmolVLM2-500M-Video-Instruct'
TEST_VIDEOS = ['46GP8', 'N11GT', '0IH69']

AD_SYSTEM_PROMPT = (
    'You are a professional audio describer for blind and low-vision (BLV) audiences, '
    'following ITC and Netflix Audio Description standards. '
    'STRICT RULES — every description MUST follow all of these: '
    '1. FIRST SENTENCE: Name the environment type and any immediate hazard within 2 meters '
    '(hot surfaces, steps, wet floor, moving people). If a hazard exists, say CAUTION explicitly. '
    '2. SPATIAL POSITIONS: Every object and person must have a direction '
    '(left/right/center/ahead/behind) AND an approximate distance in meters. '
    '3. PEOPLE: Describe by clothing color, position, and direction of movement only. '
    '4. HAZARDS: Explicitly flag steps, ramps, wet floors, hot surfaces, obstacles at '
    'foot level, and moving people or vehicles. '
    '5. DISTANCES: Use approximate meters — approximately 2 meters ahead, 1 meter to your left. '
    '6. Present tense, active voice only. '
    '7. Maximum 4 sentences. Every word must serve navigation. '
    'Write ONE single unified description across all provided images. '
    'Output only one paragraph of maximum 4 sentences. '
    'Describe this video scene for a blind user.'
)

BLV_PROMPT = (
    'You are an audio description system for blind and low vision users. '
    'Follow these rules strictly: '
    '1. First sentence: name the environment and say CAUTION if any hazard is within 2 meters. '
    '2. Every object and person must have a direction (left/right/center/ahead) and distance in meters. '
    '3. People: describe by clothing color, position, and movement direction only. '
    '4. Present tense, active voice. Maximum 4 sentences. Every word must serve navigation. '
    'Describe this video scene for a blind user.'
)

os.chdir(BASE_DIR)

with open(BASE_DIR / 'data/generated/dpo_pairs_v2.json') as f:
    data = json.load(f)
all_pairs = data['train'] + data['val']
vid_to_kfps = {p['video_id']: p['keyframe_paths'] for p in all_pairs}

with open(BASE_DIR / 'data/generated/all_captions_gemma.json') as f:
    gemma_data = json.load(f)
gemma_map = {e['video_id']: e['blv_description'] for e in gemma_data}

def load_model(checkpoint_path, processor_path=None):
    proc_src = processor_path or checkpoint_path
    try:
        processor = AutoProcessor.from_pretrained(proc_src, cache_dir=HF_CACHE)
        print(f'  Processor loaded from: {proc_src}')
    except Exception as e:
        print(f'  Processor fallback to base model ({e})')
        processor = AutoProcessor.from_pretrained(BASE_MODEL, cache_dir=HF_CACHE)
    base = SmolVLMForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map='cuda:0', cache_dir=HF_CACHE
    )
    model = PeftModel.from_pretrained(base, checkpoint_path)
    model.eval()
    return model, processor

def generate(model, processor, keyframe_paths, prompt):
    images = [Image.open(p).convert('RGB') for p in keyframe_paths if os.path.exists(p)]
    if not images:
        return '(no valid keyframes)'
    messages = [{'role': 'user', 'content':
        [{'type': 'image', 'image': img} for img in images] +
        [{'type': 'text', 'text': prompt}]
    }]
    prompt_str = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=prompt_str, images=images, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, do_sample=False, repetition_penalty=1.3)
    decoded = processor.decode(out[0], skip_special_tokens=True)
    return decoded.split('Assistant:')[-1].strip()

SEP = '-' * 60

# ── sft_v2/best ──────────────────────────────────────────────
print(f'\n{"="*60}')
print('MODEL: sft_v2/best  |  PROMPT: BLV_PROMPT (short)')
print(f'{"="*60}')
model, processor = load_model(str(BASE_DIR / 'models/student/sft_v2/best/'))
for vid in TEST_VIDEOS:
    kfps = vid_to_kfps.get(vid, [])
    caption = generate(model, processor, kfps, BLV_PROMPT)
    print(f'\n[sft_v2] Video {vid}:')
    print(caption)
del model; torch.cuda.empty_cache()

# ── sft_v3/best ──────────────────────────────────────────────
print(f'\n{"="*60}')
print('MODEL: sft_v3/best  |  PROMPT: AD_SYSTEM_PROMPT (full)')
print(f'{"="*60}')
model, processor = load_model(
    str(BASE_DIR / 'models/student/sft_v3/best/'),
    processor_path=str(BASE_DIR / 'models/student/sft_v2/best/')
)
for vid in TEST_VIDEOS:
    kfps = vid_to_kfps.get(vid, [])
    caption = generate(model, processor, kfps, AD_SYSTEM_PROMPT)
    print(f'\n[sft_v3] Video {vid}:')
    print(caption)
del model; torch.cuda.empty_cache()

# ── Gemma targets ─────────────────────────────────────────────
print(f'\n{"="*60}')
print('GEMMA TARGETS (ground truth)')
print(f'{"="*60}')
for vid in TEST_VIDEOS:
    print(f'\n[GEMMA] Video {vid}:')
    print(gemma_map.get(vid, 'N/A'))
