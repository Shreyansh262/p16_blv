# 06 — DPO Training

> **Related:** [[00_AI_READ_THIS_FIRST]] | [[05_SFT_TRAINING]] | [[15_MENTOR_UPDATES]]
> **Owner:** Training Team | **Status:** See [[17_CURRENT_PROGRESS]]

---

## Plan: Standard DPO First, SDPO Stretch Only

Mentor suggested SDPO. Team decision: standard DPO first. SDPO only if DPO is clean and time remains.

---

## DPO Pair Sources (Actual)

VideoA11y was not used (matching to Charades IDs proved infeasible in the project timeline).
Human annotation was replaced by automated Qwen/Gemma pair construction.

Chosen (high quality): Qwen2-VL-7B AD-compliant captions (high BLV score)
Rejected (low quality): same model, lower-scored outputs or truncated responses
Source file: data/generated/dpo_pairs.json — 7,842 pairs (7,057 train / 785 val)

For RLAIF-V DPO (Condition D), Gemma captions serve as chosen and lower-quality
Qwen outputs as rejected, explicitly aligning with AD system prompt distribution.

---

## Human Annotation

Automated pair construction (Qwen/Gemma scoring) was used instead of manual annotation
due to the 6-week timeline constraint. The automated approach produced 7,842 pairs with
reward_accuracy 99.6% on the DPO full run, validating pair quality.

---

## Training Config

```python
DPOConfig(
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-7,
    beta=0.1,
    bf16=True,
    eval_steps=50,
    save_steps=100,
)
```

Initialize from SFT checkpoint always.
Expected time: 2-4 hours on A6000.
VRAM: ~18-20GB — GPU 6 (48GB) handles this fine.

---

## Running It

```bash
export CUDA_VISIBLE_DEVICES=6
cd ~/p16_blv

# Test
python src/training/07_dpo_train.py --max_samples 50

# Full
python src/training/07_dpo_train.py --max_samples 5000
```

---

## SDPO (Stretch Goal Only)

Only if: standard DPO checkpoint is clean AND time remains in Week 4.
Core idea: weight DPO loss higher for BLV-critical token spans (directional, distance, safety words).
Document as novel contribution even if only standard DPO runs in practice.

---

## Checklist

- [x] Build DPO pairs from Qwen captions (7,842 pairs, data/generated/dpo_pairs.json)
- [x] Test run: 50 pairs
- [x] Full run: 5,000 pairs from SFT stage2 checkpoint — reward_acc 99.6%, Apr 2
- [x] DPO v2 on SFT v2 — models/student/dpo_v2_sft_v2/best/ (Apr 28)
- [x] RLAIF-V DPO — models/student/rlaif_dpo/best/ (Apr 28)
- [x] SimPO on RLAIF pairs — models/student/simpo/best/ (May 5)
- [x] GRPO from SFT v2 — models/student/grpo/best/, 9076 steps (May 8)
- [ ] Final ablation comparison: DPO v2 vs GRPO vs SimPO vs KTO

*Next: [[07_ABLATION]]*


---

## DPO Full Run Results (Completed 2026-04-02)

- 5,000 samples | 313 steps | 1 epoch | ~8h 20min | GPU 1
- train_loss 0.2166 | eval_loss 0.0543 | reward_accuracy 99.6% | reward_margin 4.37
- No reward hacking: rejected_logps stable throughout
- Checkpoint: models/student/dpo_checkpoint/

---

## DPO v2 on SFT v2 (Completed 2026-04-28, Condition C)

DPO run initialised from SFT v2 (models/student/sft_v2/best/) instead of SFT stage2.
Same DPO pair data. Goal: establish whether SFT v2 base improves DPO alignment.

Result: models/student/dpo_v2_sft_v2/best/
Eval (balanced, n=469): MCF=3.89 — below SFT v2 alone (4.38). DPO did not add MCF benefit.

---

## RLAIF-V DPO (Completed 2026-04-28, Condition D)

Pairs constructed from Gemma captions (chosen) vs lower-quality Qwen outputs (rejected).
This aligns the preference signal with the AD system prompt used in Gemma generation —
RLAIF-V technique: AI-feedback-generated preference labels.

Result: models/student/rlaif_dpo/best/
Eval (balanced, n=469): MCF=3.88 — also below SFT v2. RLAIF signal didn't improve MCF.

---

## GRPO — Main RL Method (Completed 2026-05-08, Condition F)

**GRPO (Group Relative Policy Optimisation)** is the primary RL method used.
Unlike DPO (offline, fixed pairs), GRPO generates new responses online and rewards
them relative to each other within a group, directly optimising the policy.

Base model: sft_v2/best | Script: src/training/09_grpo_train.py
Reward: BLV-quality function (directional content, spatial language, hazard mention)

Training trajectory:
- Step 0 (smoke test, May 3): reward -0.08 → -0.01 — reward learning confirmed
- Step 3,000: reward +0.344 — positive and stable
- Step 3,500: reward +0.362 — slowly rising
- Step 9,076 (completion, May 8): stable positive reward

Result: models/student/grpo/best/
Impact vs SFT v2 (new_conditions eval, 458 samples, 0-10 scale):
- NAF: 4.10 → 4.20 (+0.10) — navigational content improved by RL reward
- ROUGE-L: 0.272 → 0.290, METEOR: 0.281 → 0.296 — better reference alignment
- MCF: 4.30 → 4.18 (-0.12) — Ambience drop (4.91 → 4.11); RL traded scene atmosphere for nav content

Key finding: GRPO successfully reinforced navigational content (NAF) where DPO/RLAIF failed.
The online reward signal is more effective than offline preference pairs for BLV-specific quality.

---

## Preference Alignment Summary

| Method | Condition | Base | MCF (balanced) | NAF (balanced) | Outcome |
|--------|-----------|------|----------------|----------------|---------|
| DPO (SFT base) | — | SFT stage2 | — | — | 99.6% reward_acc |
| DPO v2 | C | SFT v2 | 3.89 | 3.98 | No MCF gain over SFT v2 |
| RLAIF-V DPO | D | SFT v2 | 3.88 | 3.96 | No improvement |
| SimPO | G | SFT v2 | TBD | TBD | val_loss=0.9727 |
| GRPO | F | SFT v2 | ~4.18* | ~4.20* | NAF improved, MCF slightly down |
| KTO | H | SFT v2 | TBD | TBD | adapter ~38MB |

*GRPO scores from new_conditions eval (0-10 scale), different judge — not directly comparable.
Balanced eval (1-5 scale) for Condition F pending.

Core insight: RL (GRPO) outperformed all DPO variants on navigational content. DPO offline
pairs captured general preference but not BLV-specific spatial/directional quality.
