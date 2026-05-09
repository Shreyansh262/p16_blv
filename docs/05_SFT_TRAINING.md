# 05 — SFT Training

> **Related:** [[00_AI_READ_THIS_FIRST]] | [[04_TEACHER_DATA_GEN]] | [[06_DPO_TRAINING]]
> **Owner:** Training Team | **Status:** See [[17_CURRENT_PROGRESS]]

---

## What Changed

Multi-turn REMOVED. Embedding distillation REMOVED. Single-turn only. Standard LoRA SFT confirmed.

---

## What SFT Does

Teaches SmolVLM2-500M to imitate Qwen AD-compliant descriptions.
After SFT: model knows BLV description format and vocabulary.
After DPO: model knows quality gradient within that format.

Single-turn only — one question, one complete answer. No follow-ups.
Multi-turn was rejected because it compounds latency on mobile with every turn.

---

## Training Data Stack

Stage A (optional): 10-20K LLaVA-558K filtered scene conversations — visual instruction following warm-up
Stage B (main): Qwen-generated AD descriptions, ~12k examples, single-turn

---

## LoRA Config (Confirmed by Mentor)

```python
LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
```

VRAM with SmolVLM2-500M + LoRA + batch 4 on A6000: ~14-16GB. Fits fine.

---

## Data Format

```json
{
  "messages": [
    {"role": "system", "content": "You are an assistive technology for BLV users..."},
    {"role": "user", "content": [
      {"type": "image", "image": "frame_1.jpg"},
      {"type": "image", "image": "frame_2.jpg"},
      {"type": "text", "text": "Describe this video for a blind user."}
    ]},
    {"role": "assistant", "content": "AD-compliant description here..."}
  ]
}
```

---

## Key Hyperparameters

```python
TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    bf16=True,
    eval_steps=100,
    save_steps=200,
)
```

Estimated training time: 6-10 hours on A6000.

---

## Running It

```bash
export CUDA_VISIBLE_DEVICES=6
cd ~/p16_blv

# Test first
python src/training/06_sft_train.py --max_samples 100 --epochs 1

# Full run
python src/training/06_sft_train.py --max_samples 10000 --epochs 3
```

Always inside tmux. Check GPU 6 is free first.

---

## Checklist

- [x] Format captions as single-turn conversation JSON
- [x] 90/10 train/val split
- [x] Test run: 100 samples, 1 epoch, no OOM
- [x] Full run: SFT Stage 1+2 complete (Mar 29–30)
- [x] SFT v2 single-stage complete: 4 epochs, train_loss 0.26, Apr 24
- [x] SFT v3 (Gemma-aligned) complete: Apr 30
- [x] Save SFT checkpoints — all conditions available for DPO/GRPO/ablation

*Next: [[06_DPO_TRAINING]]*

---

## SFT v2 -- Single-Stage Approach (Completed 2026-04-24)

### What Changed vs Stage 1+2

| Aspect | Original (Stage 1 + 2) | SFT v2 (Single-Stage) |
|--------|----------------------|----------------------|
| Stages | 2 (projector only, then full LoRA) | 1 unified run with phase switching |
| Phase 1 | Vision encoder + projector only, LoRA frozen | Same: steps 0->25% of total |
| Phase 2 | Full LoRA unfrozen | Same: steps 25%->end, vision LR halved to 5e-6 |
| LoRA target | LM layers only | Same (q/k/v/o/gate/up/down proj) |
| LoRA r / alpha | r=64, alpha=128 | r=64, alpha=128 (identical) |
| LR schedule | Cosine | LR per param group: vision 1e-5, connector 1e-5, LoRA 2e-5 |
| Epochs | Stage 1: 1, Stage 2: 3 | 4 epochs |
| Script | 06_sft_stage1_projector.py + 06_sft_train.py | 10_sft_v2.py |
| Output | models/student/sft_stage2_checkpoint/ | models/student/sft_v2/best/ |

### Key Design: TRANSITION_STEP

At 25% of total steps, `FreezeUnfreezeCallback` unfreezes LoRA and halves the vision LR.
This mimics the two-stage approach but in a single continuous training run with no checkpoint restart.

### Running SFT v2

```bash
conda activate blv && cd ~/p16_blv
python src/training/10_sft_v2.py --gpu 6
# Output: models/student/sft_v2/
# best/ = lowest eval_loss checkpoint
```

### Output

Checkpoints: models/student/sft_v2/checkpoint-epoch{1,2,3,4}/ and best/
Adapter only (LoRA): models/student/sft_v2/best/adapter_model.safetensors
Merged model: models/student/sft_v2_merged/ (created by 11_merge_lora.py for GGUF export)


---

## SFT v3 — Gemma-Aligned (Completed 2026-04-30)

SFT v3 re-trains from base SmolVLM2-500M using **Gemma-generated captions** instead of
Qwen captions as supervision. Gemma was prompted with the same AD system prompt used
during RLAIF-V DPO pair construction, so SFT v3 directly matches the RLAIF-V reference distribution.

Key difference from SFT v2: teacher is Gemma (AD_SYSTEM_PROMPT aligned), not Qwen.
- Source: data/generated/all_captions_gemma.json
- Output: models/student/sft_v3/best/
- Eval result (balanced, n=469): MCF=4.33, NAF=3.98 — second-best MCF after SFT v2 (4.38)

---

## SimPO Training (Completed 2026-05-05)

**SimPO (Simple Preference Optimisation)** is a DPO variant that removes the reference model,
using a length-normalised margin loss instead. Trained on the same RLAIF pairs as RLAIF-V DPO.

Why SimPO: no reference model forward pass means ~30% less VRAM per step, enabling larger batches.
The length normalisation prevents reward hacking on longer outputs.

Results:
- 450 steps, 1 epoch, val_loss=0.9727
- step 20: loss=0.9740, margin=-0.062 (chosen vs rejected)
- step 40: loss=0.9758, margin=0.000 (converging)
- Output: models/student/simpo/best/ (adapter ~77MB)

---

## KTO Training (Completed 2026-05-06)

**KTO (Kahneman-Tversky Optimisation)** trains on unpaired preference signals (individual
responses labelled good/bad rather than contrastive pairs). Better than DPO when pair
alignment is noisy.

Blocker resolved: TRL 0.17.0 KTOTrainer._process_tokens() is text-only — images were
silently ignored. Fixed with a custom collator that correctly pads image token sequences
alongside text tokens before the KTO forward pass.

Results:
- Output: models/student/kto_checkpoint/best/ (adapter ~38MB, smaller LoRA rank)
- Completed May 6 04:40

---

## SFT Results Summary (All Variants)

| Model | Train Loss | Eval Loss | Notes |
|-------|-----------|-----------|-------|
| SFT Stage 2 (original) | 0.2639 | 0.2357 | 3 epochs, two-stage |
| SFT v2 (single-stage) | — | best/ saved | 4 epochs, phase-switching |
| SFT v3 (Gemma-aligned) | — | best/ saved | Gemma teacher, Apr 30 |

Ablation (balanced eval, MCF 1-5 scale): SFT v2=4.38 > SFT v3=4.33 >> base=3.90.
SFT v2 remains best MCF model; SFT v3 best for NAF/objectivity alignment.
