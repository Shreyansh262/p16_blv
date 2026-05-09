# 11 — Team and Roles

> **Related:** [[00_AI_READ_THIS_FIRST]] | [[12_MASTER_TIMELINE]]

---

## The 3 Workstreams (9 People, 3 per Team)

Keeping 9 people from stepping on each other requires clear ownership. Each workstream has complete responsibility for their components. Cross-workstream sync happens twice a week.

---

## Workstream 1: Data Team (3 people)

**Responsible for:** Everything that happens before training starts.

| Task | File |
|------|------|
| LUV keyframe extraction pipeline | [[02_LUV_KEYFRAME_EXTRACTION]] |
| RAG knowledge library + vector store | [[03_RAG_CONTEXT]] |
| Running Qwen at scale for caption generation | [[04_TEACHER_DATA_GEN]] |
| DPO pair curation (Chosen + Rejected) | [[06_DPO_TRAINING]] |
| Human annotation process coordination | [[06_DPO_TRAINING]] |
| Final dataset assembly and formatting | [[10_DATASETS]] |

**Deliverables to hand off to Training Team:**
- `keyframes/` folder (all videos keyframed)
- `scene_index.faiss` + `scene_metadata.json` (RAG)
- `qwen_generated_captions.json` (SFT data)
- `dpo_pairs.json` (DPO data)

---

## Workstream 2: Training Team (3 people)

**Responsible for:** All model training experiments.

| Task | File |
|------|------|
| SFT training setup (LoRA config, hyperparams) | [[05_SFT_TRAINING]] |
| DPO training setup (TRL, beta, pair loading) | [[06_DPO_TRAINING]] |
| Ablation experiments (3 conditions) | [[07_ABLATION]] |
| Saving and versioning checkpoints | [[05_SFT_TRAINING]] |

**Deliverables to hand off to Eval + Deploy Team:**
- `smolvlm_sft/checkpoint-final/` (SFT model)
- `smolvlm_dpo/checkpoint-final/` (DPO model)
- Training loss curves (figures for report)

---

## Workstream 3: Eval + Deploy Team (3 people)

**Responsible for:** Measuring everything and getting it on the phone.

| Task | File |
|------|------|
| Implement Multi-Context BLV scoring | [[08_EVALUATION]] |
| Implement Navigational Assistance scoring | [[08_EVALUATION]] |
| CIDEr/BLEU evaluation setup | [[08_EVALUATION]] |
| MVBench subset evaluation | [[08_EVALUATION]] |
| Human preference study (50 samples) | [[08_EVALUATION]] |
| GGUF conversion | [[09_DEPLOYMENT]] |
| Android deployment + latency benchmarking | [[09_DEPLOYMENT]] |

**Final deliverables:**
- Results tables (ablation + full eval)
- Latency/FPS numbers from phone
- Report sections

---

## Cross-Team Dependencies

```
Data Team finishes keyframes 
    → Training Team can start SFT

Data Team finishes DPO pairs
    → Training Team can start DPO (while SFT runs)

Training Team finishes SFT checkpoint
    → Eval Team can start evaluating Condition B

Training Team finishes DPO checkpoint
    → Eval Team can start evaluating Condition C
    → Eval Team can start GGUF conversion
```

---

## Sync Schedule

- **Daily standup:** 10 minutes — what did you do, what are you doing, any blockers?
- **Twice-weekly sync:** 30 minutes — cross-team dependency check
- **Weekly review:** 1 hour — progress against timeline ([[12_MASTER_TIMELINE]]), update [[14_PROGRESS_TRACKER]]

---

## Whole-Team Tasks (Everyone Participates)

- Human annotation (Week 3): Each team member annotates ~60 DPO pairs
- Report writing (Week 6): Each workstream writes their own section
- Final presentation: All 9 members

---

*See [[12_MASTER_TIMELINE]] for week-by-week schedule*
