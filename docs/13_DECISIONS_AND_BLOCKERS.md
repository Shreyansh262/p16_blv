# 13 — Decisions and Blockers Log

> **Related:** [[00_AI_READ_THIS_FIRST]] | [[15_MENTOR_UPDATES]] | [[14_PROGRESS_TRACKER]]

---

## Pre-Project Decisions

### Drop InternVid
Terabytes, not required by PS. Charades + AVCaps sufficient.

### Use GGUF not TFLite
Native VLM support, simpler conversion, pre-built Android bindings.

---

## Server Setup Decisions (17-18 March 2026)

### Server is University Lab, Not Cloud
IP 10.8.1.106 is a private network address. Must be on university
network or VPN to SSH in.

### Username Clarification
cs671_user2 is our exclusive account. cs671_user1 is Group 2.
Different passwords, different home directories. No conflict.

### Password Changed
Default password changed. Team knows new password.

### Storage Layout
- Home (~): 15TB available on /usershome — use for everything
- /scratch: no write permission for our user — do not use
- All project files in ~/p16_blv/
- HF cache in ~/p16_blv/models/.hf_cache/

### GPU Assignment
8x RTX A6000 (48GB each) on this server.
GPU 6 most free based on nvidia-smi check.
Set: export CUDA_VISIBLE_DEVICES=6

### tmux Rule
NEVER type exit inside tmux — kills session permanently.
Always use Ctrl+B then D to detach safely.

### Conda Environment
Installed miniconda3 in ~/miniconda3/
Environment name: blv (Python 3.10.20)
Activate every session: conda activate blv

---

## Mentor Meeting 1 — 8 March 2026

### AD Guidelines Prompt — IMPLEMENTED
Replace custom prompt with ITC/Netflix Audio Description standards.
Status: In 04_TEACHER_DATA_GEN.md

### Multi-Turn — KEEP (Mentor Insisted)
Despite latency concerns, mentor specifically required multi-turn.
Solution: visual feature caching — encode once, cache, reuse.
Train on 3-turn conversations max to match inference context length.
Status: In 05_SFT_TRAINING.md

### Better Student Model — DECIDED: Stay at 500M
Researched alternatives. Going above 1B params directly hurts
mobile latency. SmolVLM2-500M is correct choice:
- Already published baseline in our paper
- Fits on mobile with acceptable latency
- Direct comparison story with paper results
Status: Decided. Do not reopen.

### 20B Teacher Model — DECIDED: Stay at 7B
InternVL2-20B tested — generation 3x slower than 7B.
With 6-week timeline, the generation phase would take too long.
Quality gain marginal when student is 500M.
Status: Using Qwen2-VL-7B-Instruct. Do not reopen.

### Embedding Distillation — REJECTED
Adds complexity, same deployed model size, zero latency benefit.
Standard text SFT with LoRA achieves same result simpler.
Status: Removed from plan entirely.

### Mobile RAG — IMPLEMENTED
SQLite + precomputed embeddings + MobileNetV3 scene classifier.
RAG fires once per scene, cached across all multi-turn turns.
Status: In 03_RAG_CONTEXT.md

---

## Mentor Meeting 2 — 11 March 2026

### Selective Decoding — HIGH PRIORITY
Constrain decoding to BLV-relevant tokens, terminate early.
Directly reduces inference latency on mobile.
Status: In 09_DEPLOYMENT.md — implement after training

### SDPO — STRETCH GOAL ONLY
Good direction but too risky to block main training on.
Run standard DPO first. Implement SDPO if time permits.
Status: Stretch goal in 06_DPO_TRAINING.md

### Reasoning Traces — REJECTED
More tokens generated = worse latency. Contradicts #1 priority.
Status: Removed from plan entirely.

### LLaVA-558K — KEEP FILTERED SUBSET
Use 10-20k filtered examples for Stage A SFT warm-up only.
Do not train on full 558K — irrelevant content wastes time.
Status: Stage A in 05_SFT_TRAINING.md

### OpenRouter — IMPLEMENT
Free API, parallel generation, zero GPU cost.
Status: In 04_TEACHER_DATA_GEN.md

### Self-Distillation — REJECTED
Multiple training runs, wrong priority, time sink.
Status: Removed from plan. Mention as future work in report.

### Diffusion Model — REJECTED
Separate research project, out of scope for 6 weeks.
Status: Removed from plan. Mention as future work in report.

### DINO — IMPLEMENT AS COMPARISON EXPERIMENT
Use DINOv2 for keyframe selection during training data generation.
Compare LUV vs DINO keyframe quality on 50 videos.
Use LUV at mobile inference (zero overhead).
Status: In 02_LUV_KEYFRAME_EXTRACTION.md

---

## Active Blockers

| Blocker | Impact | Status |
|---------|--------|--------|
| Charades metadata streaming slow | Minor — resolved, use local zip | Resolved |
| trust_remote_code deprecated | Fixed with datasets==2.21.0 | Resolved |
| typing_extensions missing | Fixed with pip install | Resolved |
| filelock + rich missing | Fixed with pip install | Resolved |

---

## Add New Entries Below
## 19 March 2026 — Output File Collision Between teacher_charades and teacher_avcaps
Type: Blocker (resolved)
What: Both 03_generate_teacher_captions.py jobs were writing to the same file
  data/generated/qwen_captions.json. teacher_avcaps (1661 videos, ~2.5h) would
  finish before teacher_charades (7985 videos, ~7h), then teacher_charades final
  save would overwrite and destroy all AVCaps captions.
Why: Script hardcoded out_path from config["data"]["qwen_captions"] with no
  --output_file override. No collision detection.
Fix: Added --output_file argument to 03_generate_teacher_captions.py (line 185).
  Killed and restarted teacher_avcaps with:
    --output_file data/generated/avcaps_captions.json --gpu 3
  teacher_charades left untouched, continues writing to qwen_captions.json.
  Final intended layout:
    data/generated/qwen_captions.json    <- Charades captions (7985 videos)
    data/generated/avcaps_captions.json  <- AVCaps captions (1661 videos)
Impact: ~50 AVCaps captions lost on restart (small). Both jobs now safe.
Status: RESOLVED — teacher_avcaps restarted, writing to separate file.

## 19 March 2026 — CUDA_VISIBLE_DEVICES Ignored by BitsAndBytes (device_map integer)
Type: Discovery / Fix
What: Setting CUDA_VISIBLE_DEVICES externally before running
  03_generate_teacher_captions.py had no effect. Model always loaded on physical
  GPU 0, causing OOM (GPU 0 was in use by other teams).
Why: The script sets os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) at
  runtime, overriding any external env var. Additionally, device_map={"": 0}
  with an integer 0 in BitsAndBytes maps to the raw physical device index,
  not the CUDA-visible index. These combined meant CUDA_VISIBLE_DEVICES=N was
  silently ignored.
Fix: Always use --gpu N argument to the script. Do NOT set CUDA_VISIBLE_DEVICES
  externally. The script handles it correctly via --gpu.
Impact: teacher_avcaps successfully running on GPU 3 via --gpu 3.
Status: RESOLVED — documented in CLAUDE.md and 17_CURRENT_PROGRESS.md.



---

*Template:*
*### [Date] — [Title]*
*Type: Decision / Blocker / Discovery*
*What:*
*Why:*
*Impact:*
*Status:*

---

## 22 March 2026 — AVCaps Data Inclusion in DPO Pipeline (Bug Fix)

Type: Bug Fix
What: `04_build_dpo_pairs.py` was hardcoded to read only `qwen_captions.json` (Charades).
  `avcaps_captions.json` (1,661 videos) was silently ignored — 17% of training data lost.
Why: Script written 16 March before AVCaps data existed (finished 19 March).
  Script was never updated after AVCaps captions became available.
  No intentional design decision — confirmed by file timestamps and absence of any docs
  mentioning AVCaps exclusion.
Fix:
  1. Merged both caption files into all_captions.json (9,646 entries total)
  2. Updated config/paths_config.yaml: qwen_captions path now points to all_captions.json
  3. Updated 04_build_dpo_pairs.py docstring to reflect merged dataset
Impact:
  - DPO pairs before fix (Charades only): ~7,057 train pairs
  - DPO pairs after fix (Charades + AVCaps): 7,842 total (7,057 train + 785 val)
  - 1,421 AVCaps pairs added — different domain adds diversity (everyday social scenes
    vs Charades household actions), improves student model robustness
Status: RESOLVED — caught before SFT training, no downstream impact


---

## 22 March 2026 — SFT Script Bugs Fixed (06_sft_train.py)

Type: Bug Fix (6 issues, all resolved same session)
Status: RESOLVED — test run passed, 100 samples/1 epoch/7 steps in 7 min

### Bug 1: transformers version too old for SmolVLM2
What: transformers==4.46.3 does not have smolvlm in CONFIG_MAPPING. Also trl==0.12.2 capped transformers<4.47.
Fix: pip install transformers==4.53.0 trl==0.17.0 num2words
Note: Qwen2-VL teacher generation is complete — upgrading transformers no longer risks breaking it.

### Bug 2: AutoModelForVision2Seq does not map to SmolVLM
What: SmolVLMForConditionalGeneration is not registered under AutoModelForVision2Seq in 4.53.0.
Fix: Import and use SmolVLMForConditionalGeneration directly.

### Bug 3: CUDA invalid device ordinal
What: Script sets CUDA_VISIBLE_DEVICES=6 then uses device_map={"": 6}.
  After CUDA_VISIBLE_DEVICES=6, GPU 6 is remapped to index 0 — device 6 does not exist.
  Same class of bug as the teacher script (documented 19 March).
Fix: device_map={"": 0} — always 0 after CUDA_VISIBLE_DEVICES is set.

### Bug 4: num2words missing
What: SmolVLMProcessor requires num2words package not in the blv conda env.
Fix: pip install num2words

### Bug 5: Image token truncation mismatch
What: processor called with max_length=512 and truncation=True.
  SmolVLM2 expands 4 keyframes into ~3328 image tokens — far exceeds 512.
  Truncation cuts input_ids to 479 tokens but text still references 3328 image tokens → ValueError.
Fix: Removed padding="max_length", truncation=True, max_length=512 from processor call.
  collate_fn rewritten to pad dynamically to longest sequence in each batch.

### Bug 6: dtype mismatch Half/Float in forward pass
What: 4-bit quantized model + LoRA causes dtype conflict in image embeddings during forward.
  Image hidden states are float32 but model expects float16.
Fix: Added prepare_model_for_kbit_training(model) before get_peft_model(model, lora_cfg).
  This is the standard PEFT pattern for 4-bit/8-bit training.

### Additional optimisations applied
- MAX_SIDE=364 image resize in BLVSFTDataset.__getitem__ (reduces token count)
- gradient_checkpointing=True in TrainingArguments (reduces VRAM during backprop)
- PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (reduces fragmentation OOM)

### Test result
- 100 samples, 1 epoch, batch_size=1: 7 steps, 7 minutes, loss=12.58
- LoRA trainable: 38,273,024 / 545,755,328 params (7.01%)
- Checkpoint saved to models/student/sft_checkpoint/

---

## 30 March 2026 — SFT Stage 2 Complete

Type: Milestone
What: Stage 2 SFT (LoRA on full SmolVLM2-500M LM, from Stage 1 checkpoint) completed.
Results:
  - train_loss: 0.2639 | eval_loss: 0.2357
  - Runtime: ~23h (82,774s) | 3 epochs | 1,629 steps
  - 38.3M trainable / 545M total params (7.0%)
  - Checkpoint: models/student/sft_stage2_checkpoint/
Impact: DPO training can now proceed from this checkpoint.
Status: COMPLETE

---

## 31 March 2026 — DPO Image Token Divisibility Error (07_dpo_train.py)

Type: Bug Fix
What: DPO training crashed on first forward pass with:
  ValueError: At least one sample has <image> tokens not divisible by patch_size.
Root cause:
  TRL's DPOConfig defaults to max_length=1024 and truncation_mode='keep_end'.
  In concatenated_forward, after flush_left, sequences exceeding 1024 tokens are
  truncated by cutting from the front (removing the earliest prompt tokens).
  SmolVLM2 uses 169 tokens per image (364×364 → 26×26 patches / scale_factor² of 4).
  Cutting the beginning of the prompt truncates image token blocks mid-image, leaving
  a count not divisible by 169. SmolVLM's inputs_merger raises the error.
Fix:
  Added max_length=None and max_prompt_length=None to DPOConfig in 07_dpo_train.py.
  This disables TRL's internal truncation entirely.
Validation:
  DPO test run with --max_samples 50 completed cleanly:
  - 4 steps, train_loss 0.5444, ~11 min
  - Model saved to models/student/dpo_checkpoint/
Impact: DPO full run (5000 samples) can now proceed.
Status: RESOLVED

---

## 1 April 2026 — DPO Efficiency Improvements (07_dpo_train.py)

Type: Optimisation
What: Three efficiency changes applied to 07_dpo_train.py before launching the full DPO run.

1. **gradient_checkpointing=True** added to DPOConfig
   - Effect: Trades compute for VRAM — recomputes activations during backward instead of storing them.
   - Reduces peak VRAM by ~30-40%, critical on a shared server where GPU memory fluctuates.

2. **MAX_SIDE=364 image resize** in load_images() pipeline
   - Added: img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS) before appending to images list.
   - Effect: Caps resolution before tokenisation. SmolVLM2 uses 169 tokens per image tile;
     large images produce many tiles. Capping at 364px keeps token count predictable and manageable.

3. **dataloader_num_workers=2** in DPOConfig
   - Effect: Prefetches batches on 2 CPU workers in parallel with GPU training.
   - Eliminates the CPU-bound stall between steps; GPU stays busy.

Impact: These changes reduced per-step time and stabilised memory use, enabling the full 5000-sample
DPO run to start cleanly on GPU 1 after earlier attempts on GPU 6 and GPU 7 were terminated (OOM/contention).
Status: ACTIVE — all 3 in current 07_dpo_train.py


---

## 5 May 2026 — SimPO Training Complete

Type: Milestone
What: SimPO (Simple Preference Optimisation) training on RLAIF pairs completed.
Results:
  - 450 steps, 1 epoch, val_loss=0.9727
  - step=20: loss=0.9740, chosen_logp=-8.062, rejected_logp=-8.000, margin=-0.062
  - step=40: loss=0.9758, chosen_logp=-7.469, rejected_logp=-7.469, margin=0.000
  - Checkpoint: models/student/simpo/best/ (saved May 5 21:30)
  - Adapter size: ~77MB
Status: COMPLETE

---

## 6 May 2026 — KTO Blocker Resolved

Type: Blocker -> Resolved
Previous status: TRL 0.17.0 KTOTrainer._process_tokens() was text-only; images silently ignored.
Resolution: Custom collator approach succeeded. KTO training completed.
Results:
  - Checkpoint: models/student/kto_checkpoint/best/ (saved May 6 04:40)
  - Adapter size: ~38MB (smaller LoRA rank than full conditions)
Status: RESOLVED — KTO now Condition H in ablation

---

## 8 May 2026 — GRPO Full Training Complete

Type: Milestone
What: GRPO (Group Relative Policy Optimisation) training on sft_v2/best completed.
Results:
  - Total steps: 9076 (estimated ~4540 from epoch-1 projection, actual ran to completion)
  - Final checkpoint: models/student/grpo/checkpoint-9076 -> models/student/grpo/best/
  - Reward trajectory: smoke test -0.08 -> step 3500 +0.362 -> stable positive at completion
  - Loss oscillation normal throughout (GRPO expected behaviour)
Impact (from new_conditions eval vs sft_v2, 458 samples):
  - NAF improved: sft_v2=4.10 -> grpo=4.20 (+0.10)
  - ROUGE-L improved: 0.2722 -> 0.2897 (+0.0175)
  - METEOR improved: 0.2810 -> 0.2961 (+0.0151)
  - MCF slightly down: 4.30 -> 4.18 (-0.12) — trades mobile-centric specificity for general fluency
Status: COMPLETE — Condition F in ablation

---

## 8 May 2026 — sft_patch_grpo Models Created

Type: Experiment
What: Post-GRPO SFT patching experiments — two variants created.
  - sft_patch_grpo (May 8 10:41): models/student/sft_patch_grpo/
  - sft_patch_grpo_v2 (May 8 11:58): models/student/sft_patch_grpo_v2/
Why: SFT patch attempts to recover MCF (mobile-centric content) lost during GRPO reward optimisation,
  while preserving GRPO gains in NAF and text overlap.
Status: Eval in progress (results/eval/final/ inference running via scripts/eval_final.py)

---

## 8 May 2026 — Final Eval Set Created (results/eval/final/)

Type: Milestone
What: Final evaluation set with 4 conditions being evaluated via scripts/eval_final.py (p16 tmux).
  - base_outputs.json (312KB, May 8 16:42)
  - sft_v2_outputs.json (256KB, May 8 16:58)
  - grpo_outputs.json (277KB, May 8 17:15)
  - sft_patch_v2_outputs.json (69KB, May 8 17:19 — still generating ~[95/458])
Status: IN PROGRESS — sft_patch_v2 outputs still being generated
