#!/usr/bin/env python3
import json, re
from pathlib import Path
from nltk.metrics.distance import edit_distance

def normalize(s):
    return re.sub(r"[^\w\s]", "", s.lower().strip()).strip()

def vqa_acc(pred, gts):
    return min(1.0, sum(normalize(g)==normalize(pred) for g in gts)/3.0)

def anls_score(pred, gt, threshold=0.5):
    pred, gt = pred.lower().strip(), gt.lower().strip()
    if not gt: return 1.0 if not pred else 0.0
    nl = edit_distance(pred, gt) / max(len(pred), len(gt))
    return 0.0 if nl > threshold else 1.0 - nl

def best_anls(pred, gts):
    return max(anls_score(pred, g) for g in gts)

preds = json.loads(Path("textvqa_preds_v2.json").read_text())
acc_scores, anls_scores = [], []
for r in preds:
    acc_scores.append(vqa_acc(r["pred"], r["gts"]))
    anls_scores.append(best_anls(r["pred"], r["gts"]))
print("=== TextVQA val Results ===")
print(f"VQA Accuracy : {sum(acc_scores)/len(acc_scores)*100:.2f}%")
print(f"ANLS         : {sum(anls_scores)/len(anls_scores):.4f}  ({sum(anls_scores)/len(anls_scores)*100:.2f}%)")
print(f"N            : {len(preds)}")
