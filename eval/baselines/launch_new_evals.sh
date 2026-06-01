#!/bin/bash
# Launch new baseline evals
# GPU 1: moondream2, qwen2_vl_2b, paligemma2_3b_instruct  (sequential)
# GPU 4: llava15_7b
# GPU 6: florence2_large, phi35_vision  (sequential)

PYTHON=/usershome/cs671_user2/miniconda3/envs/blv/bin/python3
MODELS=/usershome/cs671_user2/p16_blv/models/baselines
SCRIPTS=/usershome/cs671_user2/p16_blv/eval/baselines
OUT=/usershome/cs671_user2/p16_blv/eval/baselines

tmux new-session -d -s new_baseline_eval -n "gpu6_florence"
tmux new-window -t new_baseline_eval -n "gpu1_moondream"
tmux new-window -t new_baseline_eval -n "gpu4_llava"
tmux new-window -t new_baseline_eval -n "gpu4_phi"
tmux new-window -t new_baseline_eval -n "gpu1_qwen2"
tmux new-window -t new_baseline_eval -n "gpu1_paligemma_inst"

# GPU 6: florence2 OCR -> VQA -> phi35 OCR -> VQA
tmux send-keys -t new_baseline_eval:gpu6_florence "
$PYTHON $SCRIPTS/run_florence.py --model $MODELS/florence2_large --task ocr --gpu 6 --out $OUT/florence2_ocr_preds.json 2>&1 | tee $OUT/florence2_ocr.log &&
$PYTHON $SCRIPTS/run_florence.py --model $MODELS/florence2_large --task vqa --gpu 6 --out $OUT/florence2_vqa_preds.json 2>&1 | tee $OUT/florence2_vqa.log &&
$PYTHON $SCRIPTS/run_phi.py --model $MODELS/phi35_vision --task ocr --gpu 6 --out $OUT/phi35_ocr_preds.json 2>&1 | tee $OUT/phi35_ocr.log &&
$PYTHON $SCRIPTS/run_phi.py --model $MODELS/phi35_vision --task vqa --gpu 6 --out $OUT/phi35_vqa_preds.json 2>&1 | tee $OUT/phi35_vqa.log &&
echo DONE_GPU6" Enter

# GPU 1: moondream2 OCR -> VQA -> qwen2_vl_2b OCR -> VQA -> paligemma inst
tmux send-keys -t new_baseline_eval:gpu1_moondream "
$PYTHON $SCRIPTS/run_moondream.py --model $MODELS/moondream2 --task ocr --gpu 1 --out $OUT/moondream2_ocr_preds.json 2>&1 | tee $OUT/moondream2_ocr.log &&
$PYTHON $SCRIPTS/run_moondream.py --model $MODELS/moondream2 --task vqa --gpu 1 --out $OUT/moondream2_vqa_preds.json 2>&1 | tee $OUT/moondream2_vqa.log &&
$PYTHON $SCRIPTS/run_qwen.py --model $MODELS/qwen2_vl_2b --task ocr --gpu 1 --out $OUT/qwen2_vl_2b_ocr_preds.json 2>&1 | tee $OUT/qwen2_vl_2b_ocr.log &&
$PYTHON $SCRIPTS/run_qwen.py --model $MODELS/qwen2_vl_2b --task vqa --gpu 1 --out $OUT/qwen2_vl_2b_vqa_preds.json 2>&1 | tee $OUT/qwen2_vl_2b_vqa.log &&
$PYTHON $SCRIPTS/run_paligemma.py --model $MODELS/paligemma2_3b_instruct --task ocr --gpu 1 --out $OUT/paligemma_mix448_ocr_preds.json 2>&1 | tee $OUT/paligemma_mix448_ocr.log &&
$PYTHON $SCRIPTS/run_paligemma.py --model $MODELS/paligemma2_3b_instruct --task vqa --gpu 1 --out $OUT/paligemma_mix448_vqa_preds.json 2>&1 | tee $OUT/paligemma_mix448_vqa.log &&
echo DONE_GPU1" Enter

# GPU 4: llava15_7b OCR -> VQA
tmux send-keys -t new_baseline_eval:gpu4_llava "
$PYTHON $SCRIPTS/run_llava.py --model $MODELS/llava15_7b --task ocr --gpu 4 --out $OUT/llava15_7b_ocr_preds.json 2>&1 | tee $OUT/llava15_7b_ocr.log &&
$PYTHON $SCRIPTS/run_llava.py --model $MODELS/llava15_7b --task vqa --gpu 4 --out $OUT/llava15_7b_vqa_preds.json 2>&1 | tee $OUT/llava15_7b_vqa.log &&
echo DONE_GPU4" Enter

echo "All eval windows launched in new_baseline_eval session"
