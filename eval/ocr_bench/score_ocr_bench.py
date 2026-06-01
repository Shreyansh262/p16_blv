#!/usr/bin/env python3
import json
from pathlib import Path
from nltk.metrics.distance import edit_distance
from jiwer import wer as compute_wer

def anls(pred, gt, threshold=0.5):
    pred, gt = pred.lower().strip(), gt.lower().strip()
    if not gt: return 1.0 if not pred else 0.0
    nl = edit_distance(pred, gt) / max(len(pred), len(gt))
    return 0.0 if nl > threshold else 1.0 - nl

preds = json.loads(Path("ocr_bench_preds.json").read_text())
anls_scores, wer_scores = [], []
for r in preds:
    anls_scores.append(anls(r["pred"], r["gt"]))
    try: wer_scores.append(compute_wer(r["gt"], r["pred"]))
    except: wer_scores.append(1.0)
mean_anls = sum(anls_scores) / len(anls_scores)
mean_wer  = sum(wer_scores)  / len(wer_scores)
print("=== OCR-Bench Results ===")
print(f"ANLS : {mean_anls:.4f}  ({mean_anls*100:.2f}%)")
print(f"WER  : {mean_wer:.4f}  ({mean_wer*100:.2f}%)")
print(f"N    : {len(preds)}")
