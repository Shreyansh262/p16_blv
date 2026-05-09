# 17 — Current Server and Code Status
> Last updated: 8 May 2026

## Server Access
    SSH: ssh -i ~/.ssh/p16_key cs671_user2@10.8.1.106
    GPUs: 8x RTX A6000 48GB. Our GPU: 6 by default.
    Daily: source ~/miniconda3/etc/profile.d/conda.sh && conda activate blv && cd ~/p16_blv

## Critical Environment Notes
- torch: 2.11.0+cu128 | transformers: 5.5.1 (upgraded from 4.53.0 — all scripts working)
- torchvision mock: use EXACT block from src/eval/run_inference_all_conditions.py
  (simple sys.modules mock fails in transformers 5.5.1 — __spec__ must be set via ModuleSpec)
- If torch CUDA disappears: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

## Pipeline Status (8 May 2026)
DONE:
  - All data pipeline (keyframes, captions, DPO pairs)
  - SFT v2/best, SFT v3/best, DPO v2/best, RLAIF-DPO/best
  - GGUF Q4_K_M (290MB), Android latency (21 tok/s, 289ms TTFT)
  - Original inference + judge A-E (eval_outputs/, n=200-1929)
  - NLP metrics A-D (nlp_metrics_ABCD.json)
  - Data augmentation: 105 synthetic outdoor captions (May 5)
  - Balanced splits: train=1410, val=469, eval=469 (data/augmented/balanced_splits/)
  - Balanced eval inference A-E: ALL 469 samples DONE (May 5, results/balanced_eval/)
  - LLM judge balanced eval A-E: ALL DONE (May 6, results/eval/judge_eval.log)
  - SimPO training: models/student/simpo/best/ (May 5 21:30, 450 steps, val_loss=0.9727)
  - KTO training: models/student/kto_checkpoint/best/ (May 6 04:40) — BLOCKER RESOLVED
  - GRPO full run: models/student/grpo/best/ (May 8 08:50, checkpoint-9076, 9076 total steps)
  - New conditions eval (base/sft_v2/grpo) on 458-sample set (May 8, results/eval/new_conditions/)
  - sft_patch_grpo model (May 8 10:41) and sft_patch_grpo_v2 model (May 8 11:58)
  - sft_patch_v2 inference + judge (May 8 16:01, n=469, eval_outputs/sft_patch_v2_*)

IN PROGRESS:
  - Final eval inference (scripts/eval_final.py in p16 tmux, ~[95/458], started 16:22 May 8)
    Covers: base, sft_v2, grpo, sft_patch_v2 on results/eval/final/ dataset
  - simpo_v2 tmux: SimPO v2 complete, smoke test running

PENDING:
  - GRPO condition F judge scoring on balanced set
  - SimPO / KTO inference + judge
  - Final consolidated results table (all conditions, both eval sets)
  - Mobile multimodal pairing (Pocketpal)

## LLM Judge Results — Balanced Eval (balanced_eval.json, 469 samples, qwen2.5:32b, 1-5 scale)
  MCF:  A=3.90 | B=4.38* | C=3.89 | D=3.88 | E=4.33
  NAF:  A=3.98 | B=3.98  | C=3.98 | D=3.96 | E=3.98
  Overall: A=3.92 | B=3.97* | C=3.92 | D=3.91 | E=3.96
  (* = best in dimension. B wins MCF. DPO/RLAIF at or below base SFT.)

## LLM Judge Results — New Conditions (458 samples, 0-10 scale, results/eval/new_conditions/)
  MCF:  base=4.47 | sft_v2=4.30 | grpo=4.18
  NAF:  base=3.92 | sft_v2=4.10 | grpo=4.20*
  ROUGE-L: base=0.2146 | sft_v2=0.2722 | grpo=0.2897*
  METEOR:  base=0.2488 | sft_v2=0.2810 | grpo=0.2961*
  Key: GRPO improves NAF and text overlap metrics. SFT v2 maintains higher MCF.

## LLM Judge Results — Original Eval (qwen2.5:32b, n=200, 1-5 scale, eval_outputs/)
  MCF:  A=2.431 | B=2.775* | C=2.411 | D=2.433 | E=2.527
  NAF:  A=2.805 | B=3.297  | C=2.761 | D=2.696 | E=3.396*
  (* = best in dimension)

## NLP Metrics (original eval, A-D)
  BLEU-1:  A=12.24 | B=15.50* | C=12.03 | D=12.07
  METEOR:  A=13.15 | B=17.17* | C=13.19 | D=13.02
  ROUGE-L: A=8.27  | B=12.88* | C=8.21  | D=8.24

## sft_patch_v2 Eval (n=469, eval_outputs/sft_patch_v2_*)
  BLEU-1=44.80 | BLEU-4=13.30 | ROUGE-L=29.77 | METEOR=35.14 | CIDEr=0.011
  MCF_avg=4.71 | NAF_avg=6.24 (0-10 scale judge)

## GRPO Summary
  Base: sft_v2/best | Total steps: 9076 | Final checkpoint: grpo/checkpoint-9076
  Reward trajectory: -0.08 (smoke) -> +0.36 (step 3500) -> stable positive
  Improvement (new_conditions eval vs sft_v2): NAF +0.10, ROUGE-L +0.0175, METEOR +0.0151
  MCF slightly down (-0.12) — trades mobile-centric specificity for general quality

## SimPO Summary
  Model: models/student/simpo/best/ (saved May 5 21:30)
  Training: 450 steps, 1 epoch, val_loss=0.9727
  Trained on: rlaif pairs (same as RLAIF-DPO but with SimPO loss)

## KTO Summary
  Model: models/student/kto_checkpoint/best/ (saved May 6 04:40)
  Status: Blocker RESOLVED — custom collator approach succeeded
  Adapter size: ~38MB (smaller than full LoRA)

## Data Augmentation (all in data/augmented/ — originals untouched)
  synthetic_captions/outdoor_synthetic.json  105 entries (7 sub-types x 15)
  balanced_splits/balanced_train.json        1,410 entries
  balanced_splits/balanced_val.json            469 entries
  balanced_splits/balanced_eval.json           469 entries (9 scenes, cap=250)

## Decisions Made (do not reopen)
  1. Balanced eval set: balanced_eval.json 469 entries, cap=250/scene — removes kitchen bias
  2. Outdoor augmentation: 7 RAG sub-types x 15 synthetics via qwen2.5:32b
  3. GRPO base confirmed: sft_v2/best (already corrected in 09_grpo_train.py)
  4. Condition labels: A_base, B_sft_v2, C_dpo, D_rlaif, E_sft_v3, F_grpo, G_simpo, H_kto

## Remaining Blockers
  1. transformers processor_kwargs warnings: harmless noise, all scripts working.
  2. eval_final.py in p16 tmux still running (458 samples, ~[95/458] at 16:22 May 8)

## Next Actions (priority order)
  1. Wait for eval_final.py to complete (results/eval/final/)
  2. Run LLM judge on GRPO balanced inference (condition F)
  3. Run SimPO/KTO inference on balanced_eval set (conditions G, H)
  4. Run LLM judge on G/H balanced inference
  5. Compile final results table: all 8 conditions, both eval sets
  6. Write final report (ablation section, deployment section)
  7. Mobile multimodal pairing (Pocketpal)

## Active Tmux Sessions (8 May 2026)
  p16            — main (eval_final.py running, ~[95/458])
  balanced_eval  — done (inference A-E all complete)
  judge_eval     — done (balanced judge A-E complete)
  deploy_build   — idle
  kto_train      — done (kto training complete)
  rlaif_simpo    — done (SimPO complete)
  simpo_v2       — SimPO v2 + smoke test running

## Key Files
  results/eval/final/           — final eval outputs (base, sft_v2, grpo, sft_patch_v2)
  results/eval/new_conditions/  — new conditions summary_table.json (base/sft_v2/grpo)
  results/balanced_eval/        — balanced inference outputs A-E (469 samples each)
  results/eval/judge_eval.log   — balanced LLM judge log (A-E final scores)
  models/student/grpo/best/     — GRPO trained model (9076 steps, May 8)
  models/student/simpo/best/    — SimPO trained model (May 5)
  models/student/kto_checkpoint/best/ — KTO trained model (May 6)
  models/student/sft_patch_grpo/      — SFT patch post-GRPO (May 8)
  models/student/sft_patch_grpo_v2/   — SFT patch v2 post-GRPO (May 8)
  src/eval/run_inference_all_conditions.py  — reference for torchvision mock pattern
  src/eval/run_llm_judge.py      — LLM judge script
  src/training/09_grpo_train.py  — GRPO (sft_v2/best base confirmed)
