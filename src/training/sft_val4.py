#!/usr/bin/env python3
"""
sft_val4 -- BLV SFT Training v4 -- P16 BLV Project
Changes from sft_val3:
  - num_train_epochs=2
  - save_steps=50, eval_steps=50
  - learning_rate=1e-5, warmup_steps=100
  - Eval set: last 500 of top 3000 (min 500 samples)
  - Train set: top 3000 by BLV score (uses blv_scores_all.json)
  - torch_dtype=torch.float16 on all model loading + explicit .half()
  - load_best_model_at_end=True, metric_for_best_model=eval_loss
  - EarlyStoppingCallback(early_stopping_patience=3)
  - model.generation_config.repetition_penalty = 1.3
  - Output dir: sft_val4_checkpoint
  - Data verification block
Run inside tmux:
  conda activate blv && cd ~/p16_blv
  python scripts/sft_val4.py --gpu 6
"""

import os, sys, json, random, logging, re as _re
from pathlib import Path

import yaml
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    SmolVLMForConditionalGeneration,
    TrainingArguments,
    Trainer,
    TrainerCallback, EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument('--gpu', type=int, default=6)
_ap.add_argument('--max_samples', type=int, default=3000)
_ap.add_argument('--epochs', type=int, default=2)
_args = _ap.parse_args()


# -- Logging ----------------------------------------------------------------
os.environ['CUDA_VISIBLE_DEVICES'] = str(_args.gpu)
PROJECT_ROOT = Path('/usershome/cs671_user2/p16_blv')
log_file     = PROJECT_ROOT / 'logs/training_logs/sft_val4.log'
log_file.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
)
log = logging.getLogger(__name__)

# -- Config -----------------------------------------------------------------
def load_config():
    p = PROJECT_ROOT / 'config' / 'paths_config.yaml'
    with open(p) as f:
        return yaml.safe_load(f)

cfg = load_config()

# -- GPU verify -------------------------------------------------------------
log.info(f'Using GPU: {torch.cuda.get_device_name(0)}')
free_bytes, _ = torch.cuda.mem_get_info(0)
free_gb = free_bytes / 1e9
log.info(f'VRAM free: {free_gb:.1f} GB')
if free_gb < 20.0:
    log.error(f'Only {free_gb:.1f} GB free -- need >=20 GB. Re-run get_free_gpu.sh.')
    sys.exit(1)

# -- Data loading -----------------------------------------------------------
captions_path = Path(cfg['data']['qwen_captions'])
scores_path   = PROJECT_ROOT / 'data/generated/blv_scores_all.json'
log.info(f'Loading captions from: {captions_path}')
log.info(f'Loading BLV scores from: {scores_path}')

with open(captions_path) as f:
    all_captions = json.load(f)
with open(scores_path) as f:
    raw_scores = json.load(f)

# Build score map: video_id -> score
def get_score(v):
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):
        return float(v.get('score', 0))
    return 0.0

if isinstance(raw_scores, list):
    score_map = {s['video_id']: get_score(s.get('score', 0)) for s in raw_scores if 'video_id' in s}
elif isinstance(raw_scores, dict):
    score_map = {k: get_score(v) for k, v in raw_scores.items()}
else:
    score_map = {}

# -- Data Verification Block ------------------------------------------------
log.info('=' * 60)
log.info('DATA VERIFICATION')
log.info(f'  Total captions loaded:  {len(all_captions)}')
log.info(f'  Total scores loaded:    {len(score_map)}')

all_score_vals = list(score_map.values())
if all_score_vals:
    log.info('  Score distribution:')
    for threshold in [90, 80, 70, 60, 50]:
        count = sum(1 for s in all_score_vals if s >= threshold)
        log.info(f'    >= {threshold}: {count}')
    log.info(f'  Min score: {min(all_score_vals):.0f}  Max score: {max(all_score_vals):.0f}')

# -- Sort by BLV score and select top 3000 ----------------------------------
scored_caps = []
for c in all_captions:
    if not isinstance(c, dict):
        continue
    if len(c.get('blv_description', '').strip()) <= 30:
        continue
    vid = c.get('video_id', c.get('id', ''))
    sc  = score_map.get(vid, 0.0)
    scored_caps.append((sc, c))

scored_caps.sort(key=lambda x: -x[0])
MAX_SAMPLES = _args.max_samples if _args.max_samples else 3000
top_samples = scored_caps[:MAX_SAMPLES]

log.info(f'  After quality filter + top {MAX_SAMPLES}: {len(top_samples)} samples')
if top_samples:
    log.info(f'  Top sample score: {top_samples[0][0]:.0f}')
    log.info(f'  Bottom of top sample score: {top_samples[-1][0]:.0f}')

EVAL_SIZE   = max(500, len(top_samples) // 6)
EVAL_SIZE   = min(EVAL_SIZE, len(top_samples) - 100)
train_data  = [c for _, c in top_samples[:-EVAL_SIZE]]
eval_data   = [c for _, c in top_samples[-EVAL_SIZE:]]

log.info(f'  Train split: {len(train_data)}')
log.info(f'  Eval split:  {len(eval_data)}')
log.info('=' * 60)

# -- Dataset ----------------------------------------------------------------
class BLVSFTDataset(Dataset):
    def __init__(self, captions_data, processor, max_samples=None):
        self.data      = captions_data[:max_samples]
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry       = self.data[idx]
        description = entry.get('blv_description', '').strip()
        kf_paths    = entry.get('keyframe_paths', [])

        MAX_SIDE = 364
        images = []
        for p in kf_paths:
            if Path(p).exists():
                try:
                    img = Image.open(p).convert('RGB')
                    if max(img.size) > MAX_SIDE:
                        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
                    images.append(img)
                except Exception:
                    pass

        if not images:
            return None

        messages = [
            {
                'role': 'user',
                'content': (
                    [{'type': 'image', 'image': img} for img in images]
                    + [{'type': 'text', 'text': 'Describe this video for a blind user.'}]
                ),
            },
            {'role': 'assistant', 'content': description},
        ]

        text   = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        inputs = self.processor(text=[text], images=images, return_tensors='pt')

        input_ids      = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        labels         = input_ids.clone()

        sep = self.processor.tokenizer.convert_tokens_to_ids('<|im_start|>')
        assistant_start = (input_ids == sep).nonzero()
        if len(assistant_start) >= 2:
            labels[: assistant_start[-1].item()] = -100

        return {
            'input_ids':      input_ids,
            'attention_mask': attention_mask,
            'labels':         labels,
            'pixel_values':   inputs.get('pixel_values', torch.zeros(1)).squeeze(0),
        }


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    max_len = max(b['input_ids'].shape[0] for b in batch)
    pad_id  = 0
    input_ids, attention_masks, labels = [], [], []
    for b in batch:
        pad_len = max_len - b['input_ids'].shape[0]
        input_ids.append(torch.cat([b['input_ids'],      torch.full((pad_len,), pad_id, dtype=torch.long)]))
        attention_masks.append(torch.cat([b['attention_mask'], torch.zeros(pad_len,               dtype=torch.long)]))
        labels.append(torch.cat([b['labels'],          torch.full((pad_len,), -100,  dtype=torch.long)]))
    pv_list    = [b['pixel_values'] for b in batch]
    max_frames = max(pv.shape[0] for pv in pv_list)
    padded_pv  = []
    for pv in pv_list:
        if pv.shape[0] < max_frames:
            pad = torch.zeros(max_frames - pv.shape[0], *pv.shape[1:], dtype=pv.dtype)
            pv  = torch.cat([pv, pad], dim=0)
        padded_pv.append(pv)

    return {
        'input_ids':      torch.stack(input_ids),
        'attention_mask': torch.stack(attention_masks),
        'labels':         torch.stack(labels),
        'pixel_values':   torch.stack(padded_pv),
    }


# -- Model: float16 ---------------------------------------------------------
model_id = cfg['models']['student_id']
hf_cache = cfg['cache']['hf_cache']
os.environ['HF_HOME']                  = hf_cache
os.environ['PYTORCH_CUDA_ALLOC_CONF']  = 'expandable_segments:True'

log.info(f'Loading model: {model_id} with torch_dtype=float16')
processor = AutoProcessor.from_pretrained(model_id, cache_dir=hf_cache, trust_remote_code=True)
model     = SmolVLMForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map={'': 0},
    cache_dir=hf_cache,
)
model = model.half()
log.info('Base model loaded and cast to float16')

lora_cfg = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj'],
    lora_dropout=0.1,
    bias='none',
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()
log.info('LoRA adapter applied and model cast to float16')

try:
    model.generation_config.repetition_penalty = 1.3
    log.info('Set generation_config.repetition_penalty = 1.3')
except Exception as e:
    log.warning(f'Could not set generation_config.repetition_penalty: {e}')

# -- Callbacks --------------------------------------------------------------
epoch_losses = {}

class EpochLossCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        for entry in reversed(state.log_history):
            if 'loss' in entry:
                epoch_num = round(state.epoch)
                epoch_losses[f'epoch{epoch_num}'] = entry['loss']
                log.info(f'>>> Epoch {epoch_num} loss: {entry["loss"]:.4f}')
                break

def _blv_score_fast(caption):
    sc = 0; t = caption.lower()
    rooms   = ['kitchen','bedroom','bathroom','living','hallway','corridor','office','dining','outdoor','street','cafeteria','gym']
    spatial = ['left','right','ahead','behind','center','near','far','front','back','side','corner','adjacent','towards']
    hazards = ['hazard','caution','careful','obstacle','trip','slip','wet','step','edge','slippery','cautious']
    heights = ['height','waist','knee','chest','floor','ceiling','wall','low','high','above','below','overhead']
    if any(r in t for r in rooms): sc += 20
    sc += min(20, sum(1 for s in spatial if s in t) * 4)
    if _re.search(r'[0-9]+\s*(meter|feet|foot|inch|cm|step)', t): sc += 20
    elif any(w in t for w in heights): sc += 10
    if any(h in t for h in hazards): sc += 20
    sents = t.count('.') + t.count('!') + t.count('?')
    words = len(t.split())
    if 2 <= sents <= 5 and words <= 100: sc += 20
    elif sents <= 8 and words <= 160:    sc += 10
    return sc

BLV_LOG_PATH = PROJECT_ROOT / 'logs/training_logs/sft_val4_blv_scores.jsonl'
GOLDEN_PATH  = Path('/tmp/golden_set_10.json')
_golden_set  = json.load(open(GOLDEN_PATH)) if GOLDEN_PATH.exists() else []
_blv_plateau = 0
_blv_best    = 0.0
log.info('BLV callback: {} golden samples loaded'.format(len(_golden_set)))

class BLVScoreCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        global _blv_plateau, _blv_best
        if not _golden_set: return
        model.eval()
        scores = []
        for g in _golden_set:
            images = []
            for p in g.get('keyframe_paths', []):
                if Path(p).exists():
                    try:
                        img = Image.open(p).convert('RGB')
                        if max(img.size) > 364:
                            img.thumbnail((364, 364), Image.LANCZOS)
                        images.append(img)
                    except Exception:
                        pass
            if not images:
                continue
            msgs = [{'role': 'user', 'content':
                     [{'type': 'image', 'image': im} for im in images] +
                     [{'type': 'text', 'text': 'Describe this video for a blind user.'}]}]
            prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp    = processor(text=[prompt], images=images, return_tensors='pt')
            inp    = {k: v.to('cuda') for k, v in inp.items()}
            if 'pixel_values' in inp:
                inp['pixel_values'] = inp['pixel_values'].to(torch.float16)
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=128, do_sample=True,
                                     temperature=0.7, repetition_penalty=1.3,
                                     no_repeat_ngram_size=3)
            text = processor.tokenizer.decode(
                out[0][inp['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            scores.append(_blv_score_fast(text))
        if not scores:
            return
        avg = sum(scores) / len(scores)
        entry = {'step': state.global_step, 'mean_blv_score': round(avg, 2), 'n': len(scores)}
        with open(BLV_LOG_PATH, 'a') as f:
            f.write(json.dumps(entry) + '\n')
        log.info('Step {} | BLV Score: {:.1f}/100'.format(state.global_step, avg))
        if avg <= _blv_best + 1.0:
            _blv_plateau += 1
            if _blv_plateau >= 3:
                log.warning('BLV score plateaued for 3 evals -- stopping training')
                control.should_training_stop = True
        else:
            _blv_best = avg
            _blv_plateau = 0

# -- Datasets ---------------------------------------------------------------
CKPT_DIR = PROJECT_ROOT / 'models/student/sft_val4_checkpoint'
CKPT_DIR.mkdir(parents=True, exist_ok=True)

train_ds = BLVSFTDataset(train_data, processor)
eval_ds  = BLVSFTDataset(eval_data,  processor)

log.info(f'Train dataset: {len(train_ds)} | Eval dataset: {len(eval_ds)}')

# -- TrainingArguments ------------------------------------------------------
training_args = TrainingArguments(
    output_dir=str(CKPT_DIR),
    num_train_epochs=_args.epochs,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    warmup_steps=100,
    lr_scheduler_type='cosine',
    fp16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant': False},
    logging_dir=str(PROJECT_ROOT / 'logs/training_logs'),
    logging_steps=10,
    eval_strategy='steps',
    eval_steps=50,
    save_steps=50,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    report_to='tensorboard',
    dataloader_num_workers=2,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=collate_fn,
    callbacks=[
        EpochLossCallback(),
        BLVScoreCallback(),
        EarlyStoppingCallback(early_stopping_patience=3),
    ],
)

log.info('=' * 60)
log.info('STARTING sft_val4 TRAINING')
log.info(f'Train: {len(train_ds)} | Eval: {len(eval_ds)} | Epochs: {_args.epochs} | LR: 1e-5 | Warmup: 100')
log.info('=' * 60)

trainer.train()
model.save_pretrained(str(CKPT_DIR))
processor.save_pretrained(str(CKPT_DIR))
log.info(f'Checkpoint saved: {CKPT_DIR}')

# -- Post-training inference ------------------------------------------------
log.info('\nRunning inference on held-out eval samples...')
model.eval()

inference_results = []
MAX_SIDE = 364
holdout_samples = eval_data[:5]

with torch.no_grad():
    for sample in holdout_samples:
        video_id  = sample.get('video_id', 'unknown')
        reference = sample.get('blv_description', 'N/A').strip()
        kf_paths  = sample.get('keyframe_paths', [])

        images = []
        for p in kf_paths:
            if Path(p).exists():
                try:
                    img = Image.open(p).convert('RGB')
                    if max(img.size) > MAX_SIDE:
                        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
                    images.append(img)
                except Exception:
                    pass

        if not images:
            inference_results.append({'video_id': video_id,
                                       'generated': '[no keyframes found]',
                                       'reference': reference})
            continue

        messages = [
            {
                'role': 'user',
                'content': (
                    [{'type': 'image', 'image': img} for img in images]
                    + [{'type': 'text', 'text': 'Describe this video for a blind user.'}]
                ),
            }
        ]
        prompt_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs     = processor(text=[prompt_text], images=images, return_tensors='pt')
        inputs     = {k: v.to('cuda') for k, v in inputs.items()}
        if 'pixel_values' in inputs:
            inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)
        output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=True,
                                     temperature=0.7, repetition_penalty=1.3,
                                     no_repeat_ngram_size=3)
        generated  = processor.tokenizer.decode(
            output_ids[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        inference_results.append({'video_id': video_id,
                                   'generated': generated,
                                   'reference': reference})

# -- Final Report -----------------------------------------------------------
e1   = epoch_losses.get('epoch1', float('nan'))
e2   = epoch_losses.get('epoch2', float('nan'))
drop = (e1 - e2) if (e1 == e1 and e2 == e2) else 0.0

print('\n' + '=' * 60)
print('sft_val4 FINAL REPORT')
print('=' * 60)

print('\n[1] LOSS CURVE')
print(f'  Epoch 1 loss: {e1:.4f}')
print(f'  Epoch 2 loss: {e2:.4f}')
if drop > 1.0:
    print(f'  Drop e1->e2:  {drop:.4f}  -> GOOD')
else:
    print(f'  Drop e1->e2:  {drop:.4f}  -> WEAK (check LR or data)')
if e2 < 0.5:
    print(f'  WARNING: epoch2_loss < 0.5 -> possible OVERFITTING')

print('\n[2] SAMPLE OUTPUTS (first 120 chars, up to 3 samples)')
for i, r in enumerate(inference_results[:3]):
    print(f'\n  Sample {i+1} | video_id: {r["video_id"]}')
    print(f'  GEN: {r["generated"][:120]}')
    print(f'  REF: {r["reference"][:120]}')

print('\n[3] GO / NO-GO VERDICT')
if 2.0 <= e2 <= 4.0:
    print(f'  epoch2_loss = {e2:.4f}  -> PROCEED to full training')
elif 4.0 < e2 <= 8.0:
    print(f'  epoch2_loss = {e2:.4f}  -> WEAK -- check learning rate')
elif e2 > 8.0:
    print(f'  epoch2_loss = {e2:.4f}  -> STOP -- debug data/tokenizer')
elif e2 < 1.0:
    print(f'  epoch2_loss = {e2:.4f}  -> WARNING -- check for overfitting')

print('=' * 60)
