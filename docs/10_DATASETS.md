# 10 — Datasets

> **Related:** [[00_AI_READ_THIS_FIRST]] | [[04_TEACHER_DATA_GEN]] | [[06_DPO_TRAINING]] | [[08_EVALUATION]]

---

## Summary Table

| Dataset | Videos | Used For | Source |
|---------|--------|----------|--------|
| Charades | ~10,000 | SFT training input | HuggingFace: `HuggingFaceM4/charades` |
| AVCaps | ~2,000 | SFT training input | HuggingFace: `TUT-ARG/AVCaps` |
| VideoA11y | ~40,000 | DPO "Chosen" responses | https://people-robots.github.io/VideoA11y |
| Human annotations | ~300-500 pairs | DPO "Chosen" (gold) | Annotate yourselves (see [[06_DPO_TRAINING]]) |
| MVBench | Standard | Evaluation only | HuggingFace: `OpenGVLab/MVBench` |

**Rule:** MVBench is NEVER used during training. Only for evaluation.

---

## Charades

**What it is:** ~10,000 videos of daily indoor activities filmed by crowdsourced participants in their own homes. People drinking coffee, opening laptops, reading books, talking on phones, sitting down.

**Why you're using it:** Real-world daily life scenarios — exactly what a blind person navigating their home or workplace encounters. The indoor context is highly relevant.

**Problem with original captions:** Charades captions are action-code labels like "c092" or at best "drinking from a cup." Completely useless for your purpose. You generate new captions using Qwen (see [[04_TEACHER_DATA_GEN]]).

**How to load:**
```python
from datasets import load_dataset
ds = load_dataset("HuggingFaceM4/charades")
```

---

## AVCaps

**What it is:** ~2,000 videos with separate captions for Audio-only content, Visual-only content, and combined Audio-Visual content.

**Why you're using it:** BLV users don't just need visual descriptions — they need sound descriptions too. Footsteps approaching from behind, a car horn, a door buzzer — these are safety-relevant audio events. AVCaps teaches the model to describe *both* what it sees and what someone would hear.

**How to load:**
```python
from datasets import load_dataset
ds = load_dataset("TUT-ARG/AVCaps")
```

---

## VideoA11y

**What it is:** 40,000 videos with descriptions written by people following actual accessibility guidelines. These descriptions are:
- Objective (no interpretation)
- Descriptive (full spatial and color detail)
- Chronological (events in order)

**Why it's important:** This is your most credible source of "Chosen" responses for DPO. Human-written, guideline-following, already quality-controlled.

**How to access:**
- Website: https://people-robots.github.io/VideoA11y
- Check if HuggingFace mirror exists: search `VideoA11y` on HuggingFace
- If no direct download, contact authors — they are responsive (academic dataset)

**Note:** Match VideoA11y video IDs to your Charades/AVCaps videos where possible. For videos with no VideoA11y match, use Qwen-generated descriptions as Chosen.

---

## MVBench

**What it is:** A comprehensive video understanding benchmark with multiple-choice questions across many categories.

**What you're using from it:** Specifically the **temporal reasoning subset** — questions about the order of events in videos.

Example: "Does the person open the door before or after waving?"

**Why temporal reasoning matters:** For blind navigation, understanding *sequences* is critical. "There is a step-up" is far less useful than "after you pass the door on the left, there is a step-up approximately 1 meter ahead."

**How to load:**
```python
from datasets import load_dataset
ds = load_dataset("OpenGVLab/MVBench")
```

---

## What NOT to Use

**InternVid:** Your original strategy doc included this as a "pre-fine-tuning" dataset. Do NOT use it. It is terabytes of data, downloading alone would take days, and the PS does not require pre-training on large datasets. Remove this from your plan.

---

## Data Pipeline Overview

```
Charades (10k videos) ──→ LUV keyframes ──→ Qwen captions ──→ SFT dataset
AVCaps (2k videos)   ──/                                    /
                                                           /
VideoA11y (40k)  ─────────────────────────────────→ DPO "Chosen" pairs
Human annotations (500) ──────────────────────────/
Raw Charades labels ──────────────────────────────→ DPO "Rejected" pairs

MVBench ──────────────────────────────────────────→ Evaluation only
```

---

*See also: [[04_TEACHER_DATA_GEN]] for how Charades/AVCaps are processed | [[06_DPO_TRAINING]] for how VideoA11y is used*


---

## Dataset Usage Summary (Final)

### Charades Training
- Subset: 10,107 videos
- Captions: Qwen (SFT v2), Gemma (SFT v3)

### Balanced Evaluation (469 samples)
- Condition A (Base): MCF=3.90
- Condition B (SFT v2): MCF=4.38 (best)
- Condition C (DPO v2): MCF=3.89
- Condition D (RLAIF-V): MCF=3.88
- Condition E (SFT v3): MCF=4.33

### New Conditions Eval (458 samples)
- GRPO: NAF=4.20 on 0-10 scale

### Summary
- Balanced eval complete for A-E
- GRPO eval complete
- All checkpoints GGUF converted
- Android deployment validated