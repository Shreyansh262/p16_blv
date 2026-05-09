# 08 — Evaluation

> **Related:** [[00_AI_READ_THIS_FIRST]] | [[07_ABLATION]] | [[09_DEPLOYMENT]]
> **Owner:** Eval + Deploy Team (see [[11_TEAM_AND_ROLES]])
> **Status:** See [[14_PROGRESS_TRACKER]]
> **Last updated: 8 May 2026**

---

## Overview

You evaluate your model using **four approaches**. Using your own published frameworks (Approaches 1 and 2) is the most important — this is how you show direct improvement over the paper baseline.

---

## Approach 1 — Multi-Context BLV Framework

**From your published paper.** This framework evaluates how well a description covers four context types that matter for BLV users:

| Context | What It Measures | Example |
|---------|-----------------|---------|
| **Spatial Orientation** | Does the description mention positions, directions, distances? | "to your left", "3 meters ahead" |
| **Social Interaction** | Are people's positions and movements described? | "a person walking toward you" |
| **Action Events** | Are actions described with enough detail? | "picks up mug from counter" not "takes something" |
| **Ambience** | Is the environment type and atmosphere described? | "indoor kitchen, moderate lighting" |

**Scoring:** Each context dimension gets a score. Average = Multi-Context BLV Score.

Since you built these frameworks in your paper, you should already have the scoring code. Reuse it exactly so results are directly comparable.

---

## Approach 2 — Navigational Assistance Framework

**From your published paper.** Specifically tests whether the description contains information a blind person needs for physical navigation:

- Are obstacles mentioned?
- Are step-ups or step-downs mentioned?
- Are directions given (left/right/straight)?
- Are moving hazards (people, vehicles) mentioned?
- Are distances/proximities mentioned?

**Why this matters:** A description could score well on general quality but fail to mention a step-down right in front of the user. This framework catches that.

---

## Approach 3 — CIDEr and BLEU

Standard automatic metrics for text generation. These compare your model's output to reference descriptions using mathematical overlap.

**BLEU-4 (sacrebleu):**
- Measures 4-gram word overlap with references
- Widely used benchmark metric
- Range: 0 to 100 (sacrebleu scale)

**ROUGE-L (rouge_score):**
- Measures longest common subsequence overlap
- Better than BLEU for longer texts
- Range: 0 to 1

Already installed in blv env (3 Apr 2026): pip install sacrebleu nltk rouge-score

Note: pycocoevalcap (CIDEr) was replaced with sacrebleu + rouge-score for simpler installation.
BLEU-4 and ROUGE-L serve the same comparison purpose.

---

## Approach 4 — Human Preference Study

For a random sample of 50 videos, show human evaluators:
- Description from Condition A (Baseline)
- Description from Condition B (SFT v2) — current best model

Ask: "Which description better serves a blind person navigating this scene?"

Report as: **Preference Rate** = % of cases where B was preferred over A.

This is the most convincing result for a non-technical audience.

---

## MVBench

MVBench is a standard video understanding benchmark. The specific capability it tests that matters for your project is **temporal reasoning** — understanding the order of events.

For blind users, understanding event sequences is critical for navigation. You use a subset of MVBench questions relevant to this.
Dataset: OpenGVLab/MVBench on HuggingFace.
Report **accuracy** on the temporal reasoning subset.

---

## Running the Evaluation

Scripts and locations for each eval type:

```bash
# Balanced eval inference (all conditions A-E) — DONE May 5
# Results: results/balanced_eval/condition_*_outputs.json

# Balanced LLM judge (run_balanced_eval.py) — DONE May 6
# Results: results/eval/judge_eval.log, results/eval/balanced_scores/

# New conditions eval (base/sft_v2/grpo) — DONE May 8
# Results: results/eval/new_conditions/summary_table.json

# Final eval (scripts/eval_final.py) — IN PROGRESS (p16 tmux, ~95/458 at 17:19 May 8)
# Results: results/eval/final/{base,sft_v2,grpo,sft_patch_v2}_outputs.json

# GRPO condition F balanced inference + judge:
python3 src/eval/run_llm_judge.py --conditions F --gpu 6
```

---

## Completed Evaluation Results

### Balanced Eval — LLM Judge (n=469, qwen2.5:32b, 1-5 scale)
Source: results/eval/judge_eval.log | Completed: 6 May 2026

| Condition | MCF | NAF | Overall |
|-----------|-----|-----|---------|
| A (Base) | 3.90 | 3.98 | 3.92 |
| B (SFT v2) | **4.38** | 3.98 | **3.97** |
| C (DPO v2) | 3.89 | 3.98 | 3.92 |
| D (RLAIF-DPO) | 3.88 | 3.96 | 3.91 |
| E (SFT v3) | 4.33 | 3.98 | 3.96 |

### Original LLM Judge (n~200, qwen2.5:32b, 1-5 scale)
Source: eval_outputs/condition_*_llm_avg.json | Completed: 1-4 May 2026

| Condition | MCF | NAF |
|-----------|-----|-----|
| A (Base) | 2.431 | 2.805 |
| B (SFT v2) | **2.775** | 3.297 |
| C (DPO v2) | 2.411 | 2.761 |
| D (RLAIF-DPO) | 2.433 | 2.696 |
| E (SFT v3) | 2.527 | **3.396** |

### Original NLP Metrics (A-D)
Source: eval_outputs/nlp_metrics_ABCD.json

| Condition | BLEU-1 | METEOR | ROUGE-L | CIDEr |
|-----------|--------|--------|---------|-------|
| A | 12.24 | 13.15 | 8.27 | 0.027 |
| B | **15.50** | **17.17** | **12.88** | **0.068** |
| C | 12.03 | 13.19 | 8.21 | 0.028 |
| D | 12.07 | 13.02 | 8.24 | 0.026 |

### New Conditions (n=458, 0-10 scale)
Source: results/eval/new_conditions/summary_table.json | Completed: 8 May 2026

| Model | MCF | NAF | ROUGE-L | METEOR |
|-------|-----|-----|---------|--------|
| Base | **4.47** | 3.92 | 0.2146 | 0.2488 |
| SFT v2 | 4.30 | 4.10 | 0.2722 | 0.2810 |
| GRPO | 4.18 | **4.20** | **0.2897** | **0.2961** |

### sft_patch_v2 NLP (n=469)
Source: eval_outputs/sft_patch_v2_nlp_metrics.json | Completed: 8 May 2026

| BLEU-1 | BLEU-4 | ROUGE-L | METEOR | CIDEr | MCF (0-10) | NAF (0-10) |
|--------|--------|---------|--------|-------|------------|------------|
| 44.80 | 13.30 | 29.77 | 35.14 | 0.011 | 4.71 | 6.24 |

---

## Evaluation Checklist

- [x] Implement Multi-Context BLV scoring (keyword-based, 4 dimensions)
- [x] Implement Navigational Assistance scoring (keyword-based, 5 dimensions)
- [x] Implement BLEU-4 evaluation (sacrebleu)
- [x] Implement ROUGE-L evaluation (rouge_score)
- [x] Build held-out test set construction (excludes DPO train/val IDs)
- [x] Write crash-safe resume logic (outputs saved per-video)
- [x] Write ablation table printer (A vs B vs C with delta)
- [x] Keyword-based eval Condition A — Apr 3: blv_mean 0.7425, BLEU4 1.33
- [x] Keyword-based eval Condition B — Apr 3: blv_mean 0.63, BLEU4 23.30
- [x] Keyword-based eval Condition C — Apr 3: blv_mean 0.57, BLEU4 29.30
- [x] LLM-judge eval script written (results/eval/run_eval.py, gpt-oss:20b judge)
- [x] LLM-judge inference outputs A-E generated (Apr 5 to May 4)
- [x] LLM-judge scoring A-E COMPLETE (eval_outputs/condition_*_llm_avg.json, May 1-4)
- [x] Balanced eval inference A-E COMPLETE (results/balanced_eval/, 469 samples, May 5)
- [x] Balanced LLM judge A-E COMPLETE (results/eval/judge_eval.log + balanced_scores/, May 6)
- [x] New conditions eval base/sft_v2/grpo COMPLETE (results/eval/new_conditions/, May 8)
- [x] sft_patch_v2 inference + judge COMPLETE (eval_outputs/sft_patch_v2_*, 469 samples, May 8)
- [ ] Final eval consolidation (results/eval/final/ — scripts/eval_final.py running, May 8)
- [ ] GRPO condition F balanced inference (469 samples) + balanced LLM judge
- [ ] SimPO condition G and KTO condition H inference + judge on balanced set
- [ ] Human preference study (50 sample pairs) — stretch goal
- [ ] Compile final consolidated table for [[07_ABLATION]]
- [ ] Write evaluation section for report

---

*Also see: [[07_ABLATION]] for full comparison table | [[09_DEPLOYMENT]] for latency metrics*

*Last updated: 2026-05-08*
