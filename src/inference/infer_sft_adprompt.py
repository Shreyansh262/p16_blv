import torch, json, os
from PIL import Image
from transformers import AutoProcessor, SmolVLMForConditionalGeneration
from peft import PeftModel

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

CHECKPOINT  = 'models/student/sft_v2/best/'
BASE_MODEL  = 'HuggingFaceTB/SmolVLM2-500M-Video-Instruct'
TEST_VIDEOS = ['46GP8', 'N11GT', '0IH69']

print('Loading sft_v2 model...')
# Load processor from base model (checkpoint only has adapter weights)
try:
    processor = AutoProcessor.from_pretrained(CHECKPOINT)
except Exception:
    print('Processor not in checkpoint, loading from base model...')
    processor = AutoProcessor.from_pretrained(BASE_MODEL)

base = SmolVLMForConditionalGeneration.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map='cuda:0'
)
model = PeftModel.from_pretrained(base, CHECKPOINT)
model.eval()

with open('data/generated/dpo_pairs_v2.json') as f:
    data = json.load(f)
all_pairs = data['train'] + data['val']
vid_to_kfps = {p['video_id']: p['keyframe_paths'] for p in all_pairs}

with open('data/generated/sft_v2_captions.json') as f:
    sft_old = json.load(f)

with open('data/generated/all_captions_gemma.json') as f:
    gemma_data = json.load(f)
gemma_map = {e['video_id']: e['blv_description'] for e in gemma_data}

def generate(model, processor, keyframe_paths):
    images = [Image.open(p).convert('RGB') for p in keyframe_paths if os.path.exists(p)]
    if not images:
        return ''
    messages = [{'role': 'user', 'content':
        [{'type': 'image', 'image': img} for img in images] +
        [{'type': 'text', 'text': AD_SYSTEM_PROMPT}]
    }]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=prompt, images=images, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=150, do_sample=False, repetition_penalty=1.3)
    decoded = processor.decode(out[0], skip_special_tokens=True)
    return decoded.split('Assistant:')[-1].strip()

print()
for vid in TEST_VIDEOS:
    kfps = vid_to_kfps.get(vid, [])
    new_caption = generate(model, processor, kfps)
    print(f'=== Video: {vid} ===')
    print(f'GEMMA (target):        {gemma_map.get(vid, "N/A")}')
    print(f'SFT-v2 (old prompt):   {sft_old.get(vid, "N/A")}')
    print(f'SFT-v2 (AD prompt):    {new_caption}')
    print()
