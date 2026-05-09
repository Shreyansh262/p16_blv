# 12 — Master Timeline

> **Related:** [[00_AI_READ_THIS_FIRST]] | [[14_PROGRESS_TRACKER]] | [[17_CURRENT_PROGRESS]]
> **Last updated: 2026-05-08**

---

## Overview

| Week | Theme | Milestone | Status |
|------|-------|-----------|--------|
| Week 1 | Setup + Data | Keyframes done, captions started | DONE |
| Week 2 | Teacher generation | Full SFT dataset ready | DONE |
| Week 3 | SFT Training | SFT model trained, DPO pairs ready | DONE |
| Week 4 | DPO + Eval setup | DPO model trained, eval script ready | DONE |
| Week 5 | Ablation + SFT v2 + GGUF | Eval A/B/C done, SFT v2 done, GGUF deployed | DONE |
| Week 6 | Android + DPO v2 + RLAIF | GGUF on Android, DPO v2, RLAIF-DPO, SFT v3 | DONE |
| Week 7 | GRPO + Balanced Eval + SimPO/KTO | GRPO done, all evals done, new models | DONE |
| Week 8 | Final Eval + Report | Final consolidated table, report submission | CURRENT |

---

## WEEK 1 — Setup + Data Infrastructure DONE

- [x] Server SSH, password changed, conda blv env created
- [x] All packages installed (torch, transformers, peft, trl, etc.)
- [x] Folder structure ~/p16_blv created
- [x] RAG database built (25 scenes, 60KB)
- [x] Charades zip downloaded (55GB) at data/raw/
- [x] Keyframe extraction — Charades: 7,985 videos, 0 errors
- [x] Keyframe extraction — AVCaps: 1,661 videos, 0 errors

---

## WEEK 2 — Teacher Caption Generation DONE

- [x] AD guidelines system prompt finalized
- [x] Teacher model: Qwen2-VL-7B 4-bit (fits in ~8GB VRAM)
- [x] Full Charades generation: 7,985 captions (~4s/video)
- [x] AVCaps generation: 1,661 captions
- [x] Merged all_captions.json: 9,646 entries, 11MB
- [x] Quality check: mean BLV score 69.5/100 (87% spatial, 78% hazard, 66% room ID)
- [x] DPO pairs built: 7,842 total (7,057 train / 785 val)

---

## WEEK 3 — SFT Training DONE

### What SFT Does
Teaches SmolVLM2-500M-Video-Instruct to imitate the teacher's BLV-compliant descriptions.
Single-turn only (one video + prompt -> one description). No multi-turn.

### Training Strategy: Two-Stage (approved by mentor Suryavanshi)
- Stage 1: Freeze LoRA, train only vision encoder + connector projector (1 epoch, 3h 19min)
  Purpose: teach the vision encoder to extract BLV-relevant features before the LM sees them.
- Stage 2: Unfreeze LoRA, train full model end-to-end (3 epochs, ~23h)
  Purpose: adapt the language model backbone to generate BLV descriptions.

### Critical Bugs Fixed (3 bugs, all in 06_sft_train.py)
1. apply_chat_template silently dropped assistant content -> description token never appeared in input
2. Label masking searched for wrong token ID -> loss computed over prompt, not just response
3. Pad token was id=0 (UNK) -> should be id=2 (actual pad token for SmolLM2 tokenizer)
All three found and fixed in val1-val5 validation runs before full training.

### Results
- Stage 1: train_loss 1.2621, eval_loss 1.5075, 3h 19min, GPU 6
- Stage 2: train_loss 0.2639, eval_loss 0.2357, ~23h, 3 epochs, GPU 6
- Checkpoint: models/student/sft_stage2_checkpoint/

---

## WEEK 4 — DPO Training + Eval Infrastructure DONE

### What DPO Does
After SFT, the model knows BLV format but does not distinguish good from bad descriptions.
DPO trains it on pairs (chosen=good BLV desc, rejected=bad desc) to prefer higher-quality outputs.
Uses TRL DPOTrainer with the SFT checkpoint as starting point.

### Critical Bug Fixed (07_dpo_train.py)
TRL DPOConfig defaults to max_length=1024. The concatenated_forward function truncates
sequences longer than 1024 from the front (keep_end). SmolVLM2 requires image token counts
divisible by 169 (169 tokens per image tile). Truncating mid-image breaks this invariant,
causing a ValueError in inputs_merger. Fix: max_length=None, max_prompt_length=None.

### DPO Full Run Results (completed 2026-04-02 02:11, GPU 1)
- 5,000 samples, 313 steps, 1 epoch, ~8h 20min
- train_loss: 0.2166 | eval_loss: 0.0543
- reward_accuracy: 99.6% | reward_margin: 4.37
- No reward hacking (rejected_logps stable throughout, no collapse)
- Checkpoint: models/student/dpo_checkpoint/

### Eval Infrastructure
- src/evaluation/blv_score.py: 4 metrics (Multi-Context BLV, Nav Assist, BLEU-4, ROUGE-L)
- Crash-safe resume (picks up from last completed video on restart)
- Ablation table auto-generated in logs/eval_results/scores_all.json

---

## WEEK 5 — Ablation Eval + SFT v2 + GGUF DONE

### Part A: Ablation Evaluation (Apr 3)

Three conditions evaluated on n=100 test videos (keyword-based):

| Metric | Condition A (Base) | Condition B (SFT) | Condition C (SFT+DPO) |
|--------|-------------------|------------------|----------------------|
| BLV Spatial | 0.38 | **1.00** | **1.00** |
| BLV Social | **0.94** | 0.52 | 0.28 |
| BLV Action | 0.76 | 0.00 | 0.00 |
| BLV Ambience | 0.89 | **1.00** | **1.00** |
| **BLV Mean** | **0.7425** | 0.63 | 0.57 |
| Nav Directions | 0.20 | 0.48 | **0.72** |
| **Nav Mean** | 0.308 | 0.296 | **0.344** |
| **BLEU-4** | 1.33 | 23.30 | **29.30** |
| **ROUGE-L** | 0.182 | 0.386 | **0.430** |

### Part B: LLM-Judge Evaluation
- Judge model: qwen2.5:32b via Ollama
- 700-video test set, 10 dimensions scored per video
- Scoring A-E complete: eval_outputs/condition_*_llm_avg.json (May 1-4)

### Part C: SFT v2 (completed Apr 24)
Single-stage SFT with phase-switching callback.
- Phase 1 (0-25%): vision encoder + connector trainable, LoRA frozen
- Phase 2 (25%-end): LoRA unfrozen, vision LR halved
- 4 epochs | LoRA r=64, alpha=128 | Checkpoint: models/student/sft_v2/best/

### Part D: GGUF Conversion for Mobile (completed Apr 24)
- models/gguf/sft_v2_f16.gguf: 783 MB (full precision)
- models/gguf/sft_v2_q4km.gguf: 290 MB (Q4_K_M quantized -- DEPLOY TARGET)

---

## WEEK 6 — Android Deployment + DPO v2 + RLAIF DONE

- [x] Android latency benchmark: 21 tok/s, 289ms TTFT (Apr 25)
- [x] DPO v2 on SFT v2 completed (Apr 28): models/student/dpo_v2_sft_v2/best/
- [x] RLAIF-V DPO completed (Apr 28): models/student/rlaif_dpo/best/
- [x] SFT v3 (Gemma-prompt aligned) completed (Apr 30): models/student/sft_v3/best/
- [x] GRPO smoke test passed (May 3): reward -0.08 -> -0.01
- [x] Gemma captions generated: data/generated/all_captions_gemma.json
- [x] Original inference + judge A-E complete: eval_outputs/ (May 1-4)
- [x] NLP metrics A-D: eval_outputs/nlp_metrics_ABCD.json (May 1)
- [x] Data augmentation: 105 outdoor synthetic captions (May 5)
- [x] Balanced splits created: train=1410, val=469, eval=469 (May 5)

---

## WEEK 7 — GRPO + Balanced Eval + SimPO/KTO DONE

- [x] GRPO full training completed (May 8): models/student/grpo/best/, checkpoint-9076
      Reward trajectory: -0.08 (smoke) -> +0.362 (step 3500) -> stable positive
- [x] Balanced eval inference A-E complete (May 5): results/balanced_eval/ (469 samples each)
- [x] Balanced LLM judge A-E complete (May 6): results/eval/judge_eval.log + balanced_scores/
      A: MCF=3.90, NAF=3.98 | B: MCF=4.38 (best), NAF=3.98 | C: MCF=3.89, NAF=3.98
      D: MCF=3.88, NAF=3.96 | E: MCF=4.33, NAF=3.98
- [x] SimPO training completed (May 5): models/student/simpo/best/, 450 steps, val_loss=0.9727
- [x] KTO training completed (May 6): models/student/kto_checkpoint/best/ — blocker resolved
- [x] New conditions eval (May 8): results/eval/new_conditions/summary_table.json
      GRPO improves NAF (+0.10) and text overlap over SFT v2
- [x] sft_patch_grpo and sft_patch_grpo_v2 models created (May 8)
- [x] sft_patch_v2 inference + judge (May 8): eval_outputs/sft_patch_v2_*

---

## WEEK 8 — Final Eval + Report CURRENT WEEK

### In Progress
- [ ] Final eval inference: scripts/eval_final.py in p16 tmux (~[95/458] at 17:19 May 8)
      Conditions: base, sft_v2, grpo, sft_patch_v2 on results/eval/final/

### Remaining Tasks
- [ ] GRPO condition F balanced inference (469 samples) + balanced LLM judge
- [ ] SimPO condition G and KTO condition H inference + judge on balanced set
- [ ] Compile final consolidated results table (all 8+ conditions)
- [ ] Mobile multimodal pairing (Pocketpal)
- [ ] Write and submit final report

---

## Critical Path (full pipeline)

    Keyframe extraction
        -> Teacher captions (Qwen2-VL-7B 4-bit)
            -> DPO pairs
                -> SFT Stage 1 (vision+connector)
                    -> SFT Stage 2 (full LoRA)
                        -> DPO full run (preference optimisation)
                            -> Eval A/B/C (keyword + LLM-judge)
    [parallel] SFT v2 (single-stage)
                -> GGUF merge -> convert -> Q4_K_M -> Android (21 tok/s)
                -> DPO v2, RLAIF-V DPO, SFT v3
                -> GRPO (9076 steps, +NAF, +text overlap)
                -> SimPO, KTO
                    -> Final consolidated eval + report

---

## Model Lineage Summary

| Condition | Base | Training | Checkpoint |
|-----------|------|----------|------------|
| A | SmolVLM2-500M | None (zero-shot) | (no adapter) |
| B | SmolVLM2-500M | SFT v2 single-stage | models/student/sft_v2/best/ |
| C | SmolVLM2-500M | SFT v2 + DPO v2 | models/student/dpo_v2_sft_v2/best/ |
| D | SmolVLM2-500M | RLAIF-V DPO | models/student/rlaif_dpo/best/ |
| E | SmolVLM2-500M | SFT v3 Gemma-aligned | models/student/sft_v3/best/ |
| F | SmolVLM2-500M | GRPO from SFT v2, 9076 steps | models/student/grpo/best/ |
| G | SmolVLM2-500M | SimPO on rlaif pairs | models/student/simpo/best/ |
| H | SmolVLM2-500M | KTO custom collator | models/student/kto_checkpoint/best/ |
| GGUF | Condition B | Q4_K_M quant | models/gguf/sft_v2_q4km.gguf (290MB) |
| sft_patch_v2 | Condition F | SFT patch post-GRPO | models/student/sft_patch_grpo_v2/ |
