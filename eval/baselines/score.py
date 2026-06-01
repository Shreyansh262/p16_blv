#!/usr/bin/env python3
"""
Universal scoring script for OCR-Bench and TextVQA.
Usage: python score.py --task <ocr|vqa> --preds <path_to_json> --model <name>
"""
import json, re, argparse, numpy as np
from pathlib import Path
from nltk.metrics.distance import edit_distance
from jiwer import wer as compute_wer

parser = argparse.ArgumentParser()
parser.add_argument("--task",  required=True, choices=["ocr", "vqa"])
parser.add_argument("--preds", required=True)
parser.add_argument("--model", required=True)
args = parser.parse_args()

def anls(pred, gt, thr=0.5):
    pred, gt = pred.lower().strip(), gt.lower().strip()
    if not gt: return 1.0 if not pred else 0.0
    nl = edit_distance(pred, gt) / max(len(pred), len(gt))
    return 0.0 if nl > thr else 1.0 - nl

def normalize(s):
    return re.sub(r"[^\w\s]", "", s.lower().strip()).strip()

def vqa_acc(pred, gts):
    return min(1.0, sum(normalize(g)==normalize(pred) for g in gts)/3.0)

def best_anls(pred, gts):
    return max(anls(pred, g) for g in gts)

data = json.loads(Path(args.preds).read_text())

print(f"\n{'='*55}")
print(f"  Model : {args.model}")
print(f"  Task  : {args.task.upper()}")
print(f"  N     : {len(data)}")
print(f"{'='*55}")

if args.task == "ocr":
    a_scores, w_scores = [], []
    for r in data:
        a_scores.append(anls(r["pred"], r["gt"]))
        try: w_scores.append(compute_wer(r["gt"], r["pred"]))
        except: w_scores.append(1.0)
    print(f"  ANLS : {np.mean(a_scores)*100:.2f}%")
    print(f"  WER  : {np.mean(w_scores)*100:.2f}%")

elif args.task == "vqa":
    acc_scores  = [vqa_acc(r["pred"], r["gts"])  for r in data]
    anls_scores = [best_anls(r["pred"], r["gts"]) for r in data]
    print(f"  VQA Accuracy : {np.mean(acc_scores)*100:.2f}%")
    print(f"  ANLS         : {np.mean(anls_scores)*100:.2f}%")

print(f"{'='*55}\n")
