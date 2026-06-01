# P16 BLV Project — CS671 Deep Learning

## CRITICAL RULES — READ BEFORE DOING ANYTHING
- ALWAYS activate conda first: `conda activate blv`
- ALWAYS cd to project root: `cd ~/p16_blv`
- NEVER kill any tmux session
- NEVER use /scratch — no write permission
- NEVER use sudo
- NEVER go above 1B params for student model
- NEVER set CUDA_VISIBLE_DEVICES externally — pass --gpu N to scripts (scripts set it internally)
- Run ALL long jobs inside tmux
- Check GPU before any training run: `nvidia-smi --query-gpu=index,memory.free --format=csv`

## Server
- Host: 10.8.1.106 | User: cs671_user2
- SSH: `ssh p16server` — NEVER use IP directly
- Home: /usershome/cs671_user2/
- Project root: ~/p16_blv/
- GPU: 8x RTX A6000 (48GB each) — USE ONLY GPU 6 (default)
- Other teams on this server — do not touch other GPUs
- Get freest GPU: `bash ~/p16_blv/scripts/get_free_gpu.sh`

## Environment
- Conda env: blv (Python 3.10.20)
- Activate every session: `conda activate blv`
- Key packages: torch, transformers==4.53.0, peft==0.18.1, trl==0.17.0, tqdm, num2words
- HF cache: ~/p16_blv/models/.hf_cache/

## Standard Setup Before Any Script
```bash
conda activate blv
cd ~/p16_blv
nvidia-smi --query-gpu=index,memory.free --format=csv
```

## Project Summary
Fine-tuning SmolVLM2-500M for blind/low-vision (BLV) video descriptions.
Teacher-Student pipeline + DPO. Phone deployment. Latency is #1 priority.
Single-turn only — no multi-turn. No reasoning traces. No embedding distillation.

## Models
- Student: SmolVLM2-500M (HuggingFaceTB/SmolVLM2-500M-Video-Instruct)
- Teacher: Qwen2-VL-7B-Instruct (DONE — do not rerun)
- Hard limit: DO NOT go above 1B params for student

## Pipeline Status (as of 1 April 2026)

| Script | Status | Output |
|--------|--------|--------|
| 01_extract_keyframes.py | ✅ DONE | charades_luv_manifest.json (7985), avcaps_luv_manifest.json (1661) |
| 02_build_rag_db.py | ✅ DONE | scene_rag.db (25 scenes, 60KB) |
| 03_generate_teacher_captions.py | ✅ DONE | qwen_captions.json (7985), avcaps_captions.json (1661) |
| merge → all_captions.json | ✅ DONE | all_captions.json (9646 entries, 11MB) |
| 04_build_dpo_pairs.py | ✅ DONE | dpo_pairs.json (7842 pairs: 7057 train / 785 val) |
| 06_sft_stage1_projector.py | ✅ DONE | models/student/sft_stage1_checkpoint/ (train_loss 1.2621) |
| 06_sft_train.py (Stage 2) | ✅ DONE | models/student/sft_stage2_checkpoint/ (train_loss 0.2639, eval_loss 0.2357) |
| 07_dpo_train.py (test) | ✅ DONE | models/student/dpo_checkpoint/ (train_loss 0.5444, 50 samples) |
| 07_dpo_train.py (full) | 🔄 RUNNING | PID 1149841, tmux dpo_full, 5000 samples, ETA ~18-19h |

## NEXT ACTION: Wait for DPO Full Run, Then Evaluate
```bash
# DPO full run is active — monitor with:
tail -f ~/p16_blv/logs/dpo_full.log

# After DPO completes (~Apr 2 07:00), run evaluation:
python src/eval/blv_score.py --condition A  # base model
python src/eval/blv_score.py --condition B  # SFT only
python src/eval/blv_score.py --condition C  # SFT+DPO
```

## Directory Structure
~/p16_blv/
├── CLAUDE.md                    ← this file
├── config/
│   └── paths_config.yaml        ← qwen_captions key → all_captions.json
├── data/
│   ├── raw/                     ← Charades_v1.zip (55GB), avcaps_train_videos.zip (8.5GB)
│   ├── keyframes/
│   │   ├── charades/            ← 7985 videos × 4 keyframes
│   │   └── avcaps/              ← 1661 videos × 4 keyframes
│   ├── generated/
│   │   ├── qwen_captions.json   ← Charades teacher captions (7985)
│   │   ├── avcaps_captions.json ← AVCaps teacher captions (1661)
│   │   ├── all_captions.json    ← MERGED — use this for training (9646)
│   │   └── dpo_pairs.json       ← DPO train/val pairs (7842 total)
│   └── rag/
│       ├── scene_library.json   ← 25 scene descriptions
│       └── scene_rag.db         ← SQLite RAG database (60KB)
├── models/
│   ├── .hf_cache/               ← HuggingFace model cache
│   └── student/
│       └── sft_checkpoint/      ← SFT output (test run exists; full run will overwrite)
├── logs/
│   ├── training_logs/           ← sft_training.log written here
│   └── eval_results/
├── scripts/
│   └── get_free_gpu.sh
├── src/
│   ├── data_pipeline/
│   │   ├── 01_extract_keyframes.py          ← DONE
│   │   ├── 02_build_rag_db.py               ← DONE
│   │   ├── 03_generate_teacher_captions.py  ← DONE
│   │   └── 04_build_dpo_pairs.py            ← DONE (reads all_captions.json)
│   └── training/
│       ├── 06_sft_train.py                  ← DONE (Stage 2 SFT)
│       └── 07_dpo_train.py                  ← RUNNING (full, 5000 samples)
└── docs/
    ├── 00_AI_READ_THIS_FIRST.md
    ├── 05_SFT_TRAINING.md
    ├── 06_DPO_TRAINING.md
    ├── 13_DECISIONS_AND_BLOCKERS.md  ← all bug fixes logged here
    └── 17_CURRENT_PROGRESS.md       ← authoritative status — read each session

## Key Fixes in 06_sft_train.py (do not revert)
- transformers==4.53.0 + trl==0.17.0 (SmolVLM2 requires ≥4.52)
- SmolVLMForConditionalGeneration used directly (not AutoModelForVision2Seq)
- device_map={"": 0} — always 0 after CUDA_VISIBLE_DEVICES remaps GPU
- prepare_model_for_kbit_training() before apply_lora() — fixes dtype mismatch
- No truncation in processor — dynamic padding in collate_fn instead
- MAX_SIDE=364 image resize, gradient_checkpointing=True, PYTORCH_CUDA_ALLOC_CONF set

## Key Decisions (do not reopen)
- Teacher model: Qwen2-VL-7B (InternVL2-20B tested, 3x slower, rejected)
- Student model: SmolVLM2-500M — stay ≤1B params, phone latency is #1
- Multi-turn: DROPPED — compounds latency on mobile
- Embedding distillation: DROPPED — complex, zero latency benefit
- Reasoning traces: DROPPED — more tokens = worse latency
- SDPO: stretch goal only, after standard DPO works

## GPU Rule for Scripts
- ALWAYS pass --gpu N to the script — never set CUDA_VISIBLE_DEVICES externally
- Scripts set CUDA_VISIBLE_DEVICES internally, then use device_map={"": 0}

## How to Resume Each Session
1. ssh p16server
2. conda activate blv && cd ~/p16_blv
3. cat docs/17_CURRENT_PROGRESS.md
4. nvidia-smi --query-gpu=index,memory.free --format=csv
