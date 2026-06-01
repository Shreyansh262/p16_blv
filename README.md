# P16 BLV -- Blind-and-Low-Vision Video Description

Course: CS671 Deep Learning  |  Team: P16  |  Model: SmolVLM2-500M

Fine-tuning a lightweight vision-language model to generate accessibility-focused
video descriptions for blind and low-vision users. Pipeline covers teacher-student
distillation , SFT, GRPO, and SFT-patch, with a final GGUF export for deployment.

---

## Final Evaluation Results

See results/final_eval.png for the full table.

4-condition ablation on 458 balanced videos:

    Condition   Model               Best metrics
    A (base)    SmolVLM2-500M       Baseline -- no fine-tuning
    B           SFT_v2              BLEU-1: 18.08, blv_mean: 86%
    C           SFT_v2 + GRPO       blv_mean: 93%, action best (LLM judge)
    D (winner)  sft_patch_v2        BLEU-1: 45.57, nav_mean: 73%, objectivity: 5.38

Key improvements of D over baseline (D - A):

    BLEU-1        +26.55   (19.02 -> 45.57)
    METEOR        +22.23   (13.76 -> 35.99)
    nav_mean      +39%     (34%   -> 73%)
    nav_directions +71%    (29%   -> 100%)
    objectivity   +2.67    (2.71  -> 5.38)  [LLM judge, 1-10 scale]

Full results: results/final_eval.png
Raw data    : results/analysis/final/

---

## Directory Structure

    p16_blv/
    |
    +-- src/                 All source code (see src/README.md for full file list)
    |   +-- data_pipeline/   Steps 01-04: keyframes -> captions -> DPO pairs
    |   +-- training/        SFT, GRPO, SFT-patch training scripts
    |   +-- eval/            Inference runner, LLM judge, NLP metrics
    |   +-- inference/       Comparison and ad-hoc inference scripts
    |   +-- evaluation/      BLV scoring utilities
    |   +-- deployment/      GGUF export and Ollama benchmarking
    |   +-- debug/           Sanity check and format debug scripts
    |
    +-- data/                All datasets (see data/README.md)
    |   +-- keyframes/       Extracted video frames (AVCaps + Charades)
    |   +-- generated/       AI-generated captions, DPO pairs, training splits
    |   +-- augmented/       Balanced train/val/eval splits, KTO labels
    |
    +-- models/              All model weights (see models/README.md)
    |   +-- student/         Fine-tuned checkpoints (SFT, GRPO, sft_patch, etc.)
    |   +-- teacher/         Qwen2-VL teacher model
    |   +-- gguf/            Quantized GGUF models for deployment
    |
    +-- results/             All evaluation outputs (see results/README.md)
    |   +-- final_eval.png   FINAL EVALUATION TABLE (presentation-ready)
    |   +-- inference/       Raw model outputs for all 4 conditions
    |   +-- scores/          LLM judge scores and NLP metrics
    |   +-- analysis/        Final result tables and reference data
    |   +-- archive/         Old superseded run outputs
    |
    +-- logs/                All run logs grouped by category
    |   +-- training/        SFT, GRPO, sft_patch training logs
    |   +-- eval/            Evaluation and LLM judge logs
    |   +-- data/            Data generation logs
    |   +-- deployment/      GGUF export and Ollama benchmark logs
    |   +-- inference/       Inference comparison logs
    |
    +-- config/              YAML configs for training runs
    +-- docs/                Detailed documentation (17 docs covering full pipeline)
    +-- notebooks/           Exploratory notebooks
    +-- tools/               llama.cpp venv for GGUF conversion
    +-- bin/                 Ollama binary
    +-- ollama_models/       Ollama model files
    +-- lib/                 Shared library code
    +-- misc/                Miscellaneous files not in active pipeline

---

## Key Checkpoints

    Checkpoint            Path                                    Role
    Base (no FT)          HuggingFace SmolVLM2-500M              Condition A
    SFT v2                models/student/sft_v2/best/             Condition B
    SFT v2 + GRPO         models/student/grpo/best/               Condition C
    sft_patch_v2 (BEST)   models/student/sft_patch_grpo_v2/       Condition D -- winner
    Quantized GGUF        models/gguf/sft_v2_q4km.gguf            Deployment

---

## Quick Start

    ssh -i ~/.ssh/p16_key cs671_user2@10.8.1.106
    conda activate blv
    cd ~/p16_blv
    nvidia-smi --query-gpu=index,memory.free --format=csv   # check GPU availability

Run the main evaluation pipeline:

    python src/eval/run_inference_all_conditions.py --gpu 6
    python src/eval/run_llm_judge.py
    python src/eval/run_nlp_metrics_ABCD.py

Reproduce final table (conditions A-D):

    python src/eval/eval_final.py

---

## Documentation Index

    docs/00_AI_READ_THIS_FIRST.md        Project rules and server constraints
    docs/01_PROBLEM_EXPLAINED_SIMPLY.md  Problem statement
    docs/05_SFT_TRAINING.md              SFT training details
    docs/06_DPO_TRAINING.md              DPO training details
    docs/08_EVALUATION.md                Evaluation methodology
    docs/09_DEPLOYMENT.md                GGUF export and Ollama serving
    docs/12_MASTER_TIMELINE.md           Full project timeline
    docs/17_CURRENT_PROGRESS.md          Latest status
    src/README.md                        Every source file explained

---

## Server Rules

- Always activate conda first : conda activate blv
- Always cd to project root   : cd ~/p16_blv
- Never kill tmux session     : p16
- Use GPU 6 by default (check nvidia-smi first -- shared server)
- Run all long jobs inside tmux
- Never use sudo or write outside ~/p16_blv 
