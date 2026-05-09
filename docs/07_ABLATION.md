# 07 — Ablation Study

> **Related:** [[00_AI_READ_THIS_FIRST]] | [[05_SFT_TRAINING]] | [[06_DPO_TRAINING]] | [[08_EVALUATION]]
> **Owner:** Training Team + Eval Team (see [[11_TEAM_AND_ROLES]])
> **Status:** See [[14_PROGRESS_TRACKER]]
> **Last updated: 8 May 2026**

---

## What Is an Ablation Study (Plain English)

In medicine, "ablation" means removing something to see what happens when it's gone. In deep learning, an **ablation study** means running the same evaluation multiple times — each time with one component of your system removed — to prove that each component actually contributed to the final result.

Without this, your project makes claims like "DPO improved the model" without scientific evidence. With this, you have hard numbers proving it.

**The PS explicitly requires this:** *"Ablate RL vs. SFT"*

---

## Experimental Conditions (Extended to 8 + sft_patch)

| Cond | Model | Base | Training | Checkpoint |
|------|-------|------|----------|------------|
| **A** | Base SmolVLM2-500M | Base | None (zero-shot) | (no adapter) |
| **B** | SFT v2 | Base | Single-stage SFT, 4 epochs, LoRA r=64 | models/student/sft_v2/best/ |
| **C** | DPO v2 on SFT v2 | B | DPO on sft_v2/best | models/student/dpo_v2_sft_v2/best/ |
| **D** | RLAIF-V DPO | Base | Gemma teacher, RLAIF-V DPO | models/student/rlaif_dpo/best/ |
| **E** | SFT v3 | Base | AD_SYSTEM_PROMPT aligned with Gemma | models/student/sft_v3/best/ |
| **F** | GRPO | B | GRPO from sft_v2/best, 9076 steps | models/student/grpo/best/ |
| **G** | SimPO | Base | SimPO on rlaif pairs, 450 steps | models/student/simpo/best/ |
| **H** | KTO | Base | KTO with custom collator | models/student/kto_checkpoint/best/ |
| **sft_patch_v2** | SFT patch post-GRPO | F | SFT patching of GRPO model | models/student/sft_patch_grpo_v2/ |

---

## Results: Balanced Eval (n=469, qwen2.5:32b judge, 1-5 scale)
Source: results/eval/judge_eval.log and results/eval/balanced_scores/ | Completed: 6 May 2026

| Metric | A (Base) | B (SFT v2) | C (DPO v2) | D (RLAIF) | E (SFT v3) |
|--------|----------|------------|------------|-----------|------------|
| **MCF Score** | 3.90 | **4.38** | 3.89 | 3.88 | 4.33 |
| **NAF Score** | 3.98 | 3.98 | 3.98 | 3.96 | 3.98 |
| **Overall** | 3.92 | **3.97** | 3.92 | 3.91 | 3.96 |

Key findings:
- B (SFT v2) wins MCF by +0.48 over base. SFT training strongly improves mobility content focus.
- E (SFT v3, Gemma-aligned) has second-best MCF (4.33) and ties NAF.
- DPO variants (C, D) fall to or below base SFT on MCF — DPO reward signal did not reinforce
  mobile-centric features, instead optimised general preference.
- NAF scores are tightly clustered (3.96-3.98) across all conditions.

---

## Results: New Conditions Eval (n=458, 0-10 scale judge, results/eval/new_conditions/)
Source: results/eval/new_conditions/summary_table.json | Completed: 8 May 2026

| Metric | Base | SFT v2 | GRPO |
|--------|------|--------|------|
| **MCF Score** | **4.47** | 4.30 | 4.18 |
| **NAF Score** | 3.92 | 4.10 | **4.20** |
| Spatial Orient. | 3.214 | 3.526 | **3.786** |
| Social Interact. | 4.183 | 4.408 | **4.465** |
| Action Events | 4.197 | 4.360 | **4.365** |
| Ambience | **6.282** | 4.913 | 4.114 |
| Descriptiveness | 3.262 | 3.325 | **3.498** |
| Objectivity | 5.887 | **6.443** | 6.504 |
| Accuracy | 2.544 | 2.522 | **2.583** |
| Clarity | 4.007 | 4.120 | **4.203** |
| ROUGE-L | 0.2146 | 0.2722 | **0.2897** |
| METEOR | 0.2488 | 0.2810 | **0.2961** |

GRPO analysis:
- GRPO improves NAF (+0.10 over SFT v2), all individual NAF sub-dimensions improve.
- GRPO improves text overlap (ROUGE-L, METEOR) — descriptions better match reference structure.
- MCF slight decrease (4.30 -> 4.18) — Ambience drops sharply (4.91 -> 4.11), indicating GRPO
  de-prioritised scene atmosphere in favour of navigation-relevant content.
- This aligns with GRPO reward design: descriptions more focused on action/spatial/navigational features.

---

## Results: sft_patch_v2 (n=469, eval_outputs/sft_patch_v2_*)
Completed: 8 May 2026

| Metric | sft_patch_v2 |
|--------|-------------|
| BLEU-1 | 44.80 |
| BLEU-4 | 13.30 |
| ROUGE-L | 29.77 |
| METEOR | 35.14 |
| CIDEr | 0.011 |
| MCF avg (0-10 scale) | 4.71 |
| NAF avg (0-10 scale) | 6.24 |

Note: sft_patch_v2 NLP metrics are absolute values from sacrebleu/rouge, directly comparable
across conditions. MCF/NAF are from a different judge prompt and not directly comparable to
the 1-5 scale balanced eval above.

---

## Results: Original LLM Judge (n=200, qwen2.5:32b, 1-5 scale, eval_outputs/)
Source: eval_outputs/condition_*_llm_avg.json | Completed: 1-4 May 2026

| Metric | A (Base) | B (SFT v2) | C (DPO) | D (RLAIF) | E (SFT v3) |
|--------|----------|------------|---------|-----------|------------|
| MCF Score | 2.431 | **2.775** | 2.411 | 2.433 | 2.527 |
| NAF Score | 2.805 | 3.297 | 2.761 | 2.696 | **3.396** |
| Spatial Orient. | 1.880 | **2.305** | 1.795 | 1.840 | 2.170 |
| Objectivity | 4.580 | 5.510 | 4.575 | 4.415 | **5.835** |

B wins MCF. E wins NAF + objectivity. DPO/RLAIF at or below base SFT on all dimensions.

---

## NLP Metrics — Original Eval (A-D)
Source: eval_outputs/nlp_metrics_ABCD.json

| Metric | A | B | C | D |
|--------|---|---|---|---|
| BLEU-1 | 12.24 | **15.50** | 12.03 | 12.07 |
| METEOR | 13.15 | **17.17** | 13.19 | 13.02 |
| ROUGE-L | 8.27 | **12.88** | 8.21 | 8.24 |
| CIDEr | 0.027 | **0.068** | 0.028 | 0.026 |

B wins all NLP metrics by a wide margin. DPO/RLAIF do not improve over SFT v2.

---

## Eval Sets Summary

| Set | n | Scale | Source | Status |
|-----|---|-------|--------|--------|
| Balanced eval (A-E) | 469 | 1-5 | results/balanced_eval/ + results/eval/balanced_scores/ | DONE |
| Original eval (A-E) | ~200 | 1-5 | eval_outputs/ | DONE |
| New conditions (base/sft_v2/grpo) | 458 | 0-10 | results/eval/new_conditions/ | DONE |
| Final eval (base/sft_v2/grpo/sft_patch_v2) | 458 | - | results/eval/final/ | IN PROGRESS |
| Balanced eval (F, G, H) | 469 | 1-5 | TBD | PENDING |

---

## Interpretation Summary

1. **SFT v2 is the strongest single-stage model** on both MCF and NLP metrics across all eval sets.
2. **DPO and RLAIF-DPO do not improve over SFT v2** on MCF/NAF — the preference signal
   did not successfully align with BLV-specific quality (mobility, spatial content).
3. **SFT v3 (Gemma-aligned)** recovers NAF and objectivity but not MCF.
4. **GRPO improves NAF** (+0.10) and text overlap over SFT v2, at slight MCF cost.
   Most theoretically interesting finding: RL reward improves navigational content while
   SFT maximisation better captures mobile-centric focus.
5. **sft_patch_grpo_v2** attempts to combine GRPO NAF gains with SFT MCF strength (eval pending).

---

## Checklist

- [x] Finalize held-out test set (balanced_eval.json: 469 samples, 9 scenes, cap=250)
- [x] Run evaluation Condition A (Base) — balanced + original DONE
- [x] Run evaluation Condition B (SFT v2) — balanced + original DONE
- [x] Run evaluation Condition C (DPO v2) — balanced + original DONE
- [x] Run evaluation Condition D (RLAIF-DPO) — balanced + original DONE
- [x] Run evaluation Condition E (SFT v3) — balanced + original DONE
- [x] Run evaluation Condition F (GRPO) — new_conditions eval DONE
- [x] Fill in results tables — DONE (all tables above)
- [ ] Run GRPO balanced inference (469 samples) + balanced judge for Condition F
- [ ] Run Condition G (SimPO) and H (KTO) inference + judge on balanced set
- [ ] Final eval consolidation (results/eval/final/ — in progress)
- [ ] Human preference study (50 sample pairs) — stretch goal
- [ ] Write ablation section for final report

---

*Feeds into: [[08_EVALUATION]] | [[14_PROGRESS_TRACKER]]*
