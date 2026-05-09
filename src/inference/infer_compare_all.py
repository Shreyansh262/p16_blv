import torch, json, os
from PIL import Image
from transformers import AutoProcessor, SmolVLMForConditionalGeneration
from peft import PeftModel

BLV_PROMPT = (
    'You are an audio description system for blind and low vision users. '
    'Follow these rules strictly: '
    '1. First sentence: name the environment and say CAUTION if any hazard is within 2 meters. '
    '2. Every object and person must have a direction (left/right/center/ahead) and distance in meters. '
    '3. People: describe by clothing color, position, and movement direction only. '
    '4. Present tense, active voice. Maximum 4 sentences. Every word must serve navigation. '
    'Describe this video scene for a blind user.'
)

BASE_MODEL  = 'HuggingFaceTB/SmolVLM2-500M-Video-Instruct'
TEST_VIDEOS = ['46GP8', 'N11GT', '0IH69']

CHECKPOINTS = {
    'sft_v2':    'models/student/sft_v2/best/',
    'dpo_ck800': 'models/student/dpo_v2_sft_v2/checkpoint-800',
    'rlaif_dpo': 'models/student/rlaif_dpo/best',
}

with open('data/generated/dpo_pairs_v2.json') as f:
    data = json.load(f)
all_pairs = data['train'] + data['val']
vid_to_kfps = {p['video_id']: p['keyframe_paths'] for p in all_pairs}

with open('data/generated/all_captions_gemma.json') as f:
    gemma_data = json.load(f)
gemma_map = {e['video_id']: e['blv_description'] for e in gemma_data}

def load_model(checkpoint):
    # Some checkpoints only contain adapter weights — fall back to base model for processor
    try:
        processor = AutoProcessor.from_pretrained(checkpoint)
    except Exception:
        print(f'  (processor not in checkpoint, loading from base model)')
        processor = AutoProcessor.from_pretrained(BASE_MODEL)
    base = SmolVLMForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map='cuda:0'
    )
    model = PeftModel.from_pretrained(base, checkpoint)
    model.eval()
    return model, processor

def generate(model, processor, keyframe_paths):
    images = [Image.open(p).convert('RGB') for p in keyframe_paths if os.path.exists(p)]
    if not images:
        return ''
    messages = [{'role': 'user', 'content':
        [{'type': 'image', 'image': img} for img in images] +
        [{'type': 'text', 'text': BLV_PROMPT}]
    }]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=prompt, images=images, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=150, do_sample=False, repetition_penalty=1.3)
    decoded = processor.decode(out[0], skip_special_tokens=True)
    return decoded.split('Assistant:')[-1].strip()

for ckpt_name, ckpt_path in CHECKPOINTS.items():
    print(f'\n=== Loading {ckpt_name} ===')
    try:
        model, processor = load_model(ckpt_path)
        for vid in TEST_VIDEOS:
            kfps = vid_to_kfps.get(vid, [])
            caption = generate(model, processor, kfps)
            print(f'\n[{ckpt_name}] Video {vid}:')
            print(f'  {caption}')
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f'Failed to load {ckpt_name}: {e}')

print('\n=== GEMMA TARGETS ===')
for vid in TEST_VIDEOS:
    print(f'\n[GEMMA] Video {vid}:')
    print(f'  {gemma_map.get(vid, "N/A")}')
