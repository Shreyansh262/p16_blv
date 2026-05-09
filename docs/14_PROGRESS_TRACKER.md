# 14 — Progress Tracker

> Update this at every weekly review.
> **Related:** [[00_AI_READ_THIS_FIRST]] | [[17_CURRENT_PROGRESS]]
> **Last updated: 8 May 2026**

---

## Week 1 | 18 March 2026

### Infrastructure
| Component | Status |
|-----------|--------|
| Server access | DONE |
| conda blv env | DONE |
| All packages | DONE |
| Folder structure | DONE |
| paths_config.yaml | DONE |
| GPU 6 assigned | DONE |
| tmux configured | DONE |

### Data Pipeline
| Component | Status | Notes |
|-----------|--------|-------|
| Charades zip | DONE | 55GB |
| RAG database | DONE | 25 scenes, 60KB |
| Keyframe test (5 videos) | DONE | 0 errors |
| Keyframe extraction FULL — Charades | DONE | 7985 videos, 0 errors, manifest saved |
| Teacher captions test (10 videos) | DONE | 10/10 success, ~4s/video |
| AVCaps zip download | DONE | 8.5GB via wget, saved to data/raw/ |
| AVCaps keyframe extraction | DONE | 1661 videos, 0 errors |
| Full Charades teacher captions | DONE | 7985 entries, 9.0MB |
| AVCaps teacher captions | DONE | 1661 entries, 1.9MB |
| Merged all_captions.json | DONE | 9646 entries, 11MB |
| DPO pairs | DONE | 7842 pairs (7057 train / 785 val) |

### Training
| Component | Status | Notes |
|-----------|--------|-------|
| SFT script written | DONE | |
| DPO script written | DONE | |
| SFT test run | DONE | 100 samples, 22 Mar |
| SFT label masking bug fix | DONE | 3 bugs fixed, 29 Mar |
| SFT val5 validation run | DONE | 200 samples, loss 1.99->0.94, generation works |
| Stage 1 script (vision+projector) | DONE | 06_sft_stage1_projector.py, smoke tested |
| Stage 1 full training | DONE | train_loss 1.2621, eval_loss 1.5075, 3h 19min |
| Stage 2 SFT (LoRA on LM) | DONE | train_loss 0.2639, eval_loss 0.2357, ~23h, 3 epochs |
| DPO test run (50 samples) | DONE | train_loss 0.5444, ~11 min |
| DPO full run (5000 samples) | **DONE** | train_loss 0.2166, eval_loss 0.0543, reward_acc 99.6%, 8h 20min — completed 2026-04-02 02:11 |

### Evaluation
| Component | Status | Notes |
|-----------|--------|-------|
| src/evaluation/blv_score.py | **DONE** | All 4 approaches: Multi-Context BLV, Nav Assist, BLEU-4, ROUGE-L |
| sacrebleu + rouge-score packages | **DONE** | Installed 3 Apr 2026 |
| Condition A eval (base model) | **DONE** | Apr 3 14:27 — blv_mean 0.7425, nav_mean 0.308, BLEU4 1.33, ROUGE-L 0.182 |
| LLM-judge script (run_eval.py) | **DONE** | qwen2.5:32b judge, 700-video test set, crash-safe resume |
| LLM-judge inference outputs A-E | **DONE** | Apr 5-May 4 — eval_outputs/condition_{A-E}_outputs.json |
| LLM-judge scoring A-E | **DONE** | eval_outputs/condition_*_llm_avg.json, May 1-4 |
| NLP metrics A-D | **DONE** | eval_outputs/nlp_metrics_ABCD.json (May 1) |
| GGUF export | **DONE** | sft_v2_q4km.gguf 290MB, Android 21 tok/s 289ms TTFT |

---

## Week 2 | 22 March 2026

### Completed
- [x] AVCaps keyframe extraction complete
- [x] Full Charades teacher captions complete
- [x] Full AVCaps teacher captions complete
- [x] all_captions.json merged (9646 entries)
- [x] DPO pair curation complete (7842 pairs)
- [x] SFT test run (100 samples, 1 epoch) — training loop validated

---

## Week 3 | 29 March 2026

### Completed
- [x] SFT label masking bug diagnosed and fixed (3 critical bugs)
- [x] SFT val5 validation run — confirmed fix works (5/5 non-empty outputs)
- [x] Stage 1 training script created (vision encoder + projector)
- [x] Stage 1 smoke test passed (50 samples, loss=2.42, no OOM)
- [x] 2-stage training strategy adopted (mentor Suryavanshi approved)
- [x] Stage 1 full training complete (train_loss 1.2621, eval_loss 1.5075, 3h 19min)
- [x] Stage 2 SFT full training complete (train_loss 0.2639, eval_loss 0.2357, ~23h)

---

## Week 4 | 31 March – 3 April 2026

### Completed
- [x] Stage 2 SFT confirmed complete — train_loss 0.2639, eval_loss 0.2357, ~23h, checkpoint saved
- [x] DPO script (07_dpo_train.py) written and debugged
- [x] DPO image token divisibility bug diagnosed and fixed (TRL max_length=1024 default)
- [x] DPO test run (50 samples, 4 steps, train_loss 0.5444) — PASSED
- [x] DPO efficiency improvements — gradient_checkpointing, MAX_SIDE=364, dataloader_num_workers=2
- [x] **DPO full run COMPLETE** — train_loss 0.2166, eval_loss 0.0543, reward_accuracy 99.6%, reward_margin 4.37, 8h 20min, 313 steps
- [x] **src/evaluation/blv_score.py written** — all 4 evaluation approaches, crash-safe resume, ablation table output
- [x] sacrebleu, nltk, rouge-score installed in blv env

### Next (Week 5)
- [x] Run Condition A eval (baseline) — Apr 3
- [x] Run Condition B eval (SFT) — Apr 3
- [x] Run Condition C eval (SFT+DPO) — Apr 3
- [ ] Human preference study (50 pairs)
- [x] GGUF export + Android benchmark

---

## Week 5 | 3-6 April 2026

### Completed
- [x] Evaluation script (blv_score.py) — packages installed 3 Apr
- [x] Evaluation Condition A (base SmolVLM2-500M) — Apr 3 14:27
- [x] Evaluation Condition B (SFT Stage 2) — Apr 3 14:32
- [x] Evaluation Condition C (SFT+DPO) — Apr 3 14:37
- [x] Ollama binary + CUDA libs set up — Apr 4 05:17
- [x] LLM-judge eval script (run_eval.py) written — Apr 5
- [x] LLM-judge inference: all 700 outputs generated for Conditions A, B, C — Apr 5

---

## Code Fixes Log

| Date | Fix | File | Detail |
|------|-----|------|--------|
| 19 Mar | device_map={: 0} | 03_generate_teacher_captions.py | Was auto, now explicit GPU 0 |
| 19 Mar | Removed --multiturn | 03_generate_teacher_captions.py | Single-turn only per CLAUDE.md |
| 19 Mar | transformers 4.46.3 | conda blv env | Needed for Qwen2VL arch support |
| 19 Mar | MAX_SIDE=640 resize | 03_generate_teacher_captions.py | Fixes CUDA OOM on large keyframes |
| 19 Mar | Rewrote process_avcaps | 01_extract_keyframes.py | HF Hub zip+JSON download |
| 22 Mar | AVCaps merged | 04_build_dpo_pairs.py | Merged all_captions.json |
| 22 Mar | transformers 4.53.0 | conda blv env | SmolVLM2 support |
| 22 Mar | SmolVLMForConditionalGeneration | 06_sft_train.py | AutoModelForVision2Seq wrong class |
| 22 Mar | prepare_model_for_kbit_training | 06_sft_train.py | dtype Half/Float mismatch |
| 29 Mar | **Description dropped fix** | 06_sft_train.py | apply_chat_template dropped assistant content |
| 29 Mar | **Label masking fix** | 06_sft_train.py | Used prompt_len split instead of token search |
| 29 Mar | **Pad token fix** | 06_sft_train.py | id=0 (UNK) -> id=2 (actual pad) |
| 31 Mar | max_length=None | 07_dpo_train.py | TRL DPOConfig default=1024 truncated image tokens mid-image -> SmolVLM inputs_merger ValueError |
| 31 Mar | max_prompt_length=None | 07_dpo_train.py | Same root cause, prevent prompt truncation |
| 1 Apr | gradient_checkpointing=True | 07_dpo_train.py | Reduces peak VRAM ~30-40%; trades compute for memory |
| 1 Apr | MAX_SIDE=364 + thumbnail resize | 07_dpo_train.py | Caps image resolution before tokenisation; keeps tile count manageable |
| 1 Apr | dataloader_num_workers=2 | 07_dpo_train.py | Parallel CPU prefetch; eliminates GPU idle between steps |
| 3 Apr | pip install sacrebleu nltk rouge-score | conda blv env | Required for blv_score.py eval metrics |
| 6 Apr | timeout=120 -> timeout=600 | results/eval/run_eval.py | qwen2.5:32b thinking model; reasoning chain takes >120s on complex prompts |
| May 5 | ModuleSpec torchvision mock | balanced_inference.py | transformers 5.5.1 requires __spec__ set via ModuleSpec; simple sys.modules mock fails |
| May 6 | KTO custom collator | kto_train.py | TRL KTOTrainer text-only; custom collator handles image tokens |

*Last updated: 2026-05-08*

---

## Week 6 | 21-27 April 2026

### Completed
- [x] sft_single_stage.py -- single-stage SFT experiment, Apr 21
- [x] dpo_checkpoint_v2 -- DPO on top of sft_single_stage, Apr 22
- [x] 10_sft_v2.py (single-stage SFT v2) -- 4 epochs, phase-1 vision+connector then phase-2 full LoRA (r=64, alpha=128), completed Apr 24
- [x] GGUF pipeline: 11_merge_lora.py + 12_gguf_pipeline.sh, sft_v2_q4km.gguf (290MB), Apr 24
- [x] Android latency benchmark: 21 tok/s, 289ms TTFT, Apr 25
- [x] DPO v2 on SFT v2: models/student/dpo_v2_sft_v2/best/, Apr 28
- [x] RLAIF-V DPO: models/student/rlaif_dpo/best/, Apr 28
- [x] SFT v3 (Gemma-prompt aligned): models/student/sft_v3/best/, Apr 30
- [x] Gemma captions for RLAIF: data/generated/all_captions_gemma.json

---

## Week 7 | 1-8 May 2026

### Completed
- [x] LLM-judge scoring A-E on original eval (eval_outputs/, qwen2.5:32b, n~200, May 1-4)
- [x] NLP metrics A-D: nlp_metrics_ABCD.json (May 1)
- [x] GRPO smoke test passed: reward -0.08 -> -0.01 (May 3)
- [x] Data augmentation: 105 outdoor synthetic captions added (May 5)
- [x] Balanced splits: train=1410, val=469, eval=469 created (May 5)
- [x] Balanced eval inference A-E: ALL 469 samples DONE (May 5, results/balanced_eval/)
- [x] SimPO training: models/student/simpo/best/ (May 5, 450 steps, val_loss=0.9727)
- [x] KTO training: models/student/kto_checkpoint/best/ (May 6, blocker resolved)
- [x] Balanced LLM judge A-E COMPLETE (May 6, results/eval/balanced_scores/)
      A: MCF=3.90, NAF=3.98 | B: MCF=4.38* | C: MCF=3.89 | D: MCF=3.88 | E: MCF=4.33
- [x] GRPO full training COMPLETE: models/student/grpo/best/, 9076 steps (May 8)
      Reward: +0.36 stable | GRPO improves NAF and text overlap over SFT v2
- [x] New conditions eval (base/sft_v2/grpo): results/eval/new_conditions/ (May 8)
- [x] sft_patch_grpo + sft_patch_grpo_v2 models created (May 8)
- [x] sft_patch_v2 inference + judge: eval_outputs/sft_patch_v2_* (May 8)

### In Progress (as of 8 May 2026)
- [ ] Final eval inference: scripts/eval_final.py in p16 tmux (~[95/458] at 17:19)
      results/eval/final/{base,sft_v2,grpo,sft_patch_v2}_outputs.json

---

## Week 8 | 8 May 2026 — Report Week CURRENT

### Remaining Tasks
- [ ] Finish final eval inference (results/eval/final/)
- [ ] GRPO condition F balanced inference (469 samples) + balanced judge
- [ ] SimPO condition G and KTO condition H inference + judge on balanced set
- [ ] Compile final consolidated results table (all conditions, both eval sets)
- [ ] Mobile multimodal pairing (Pocketpal)
- [ ] Write and submit final report
