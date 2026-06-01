#!/usr/bin/env python3
"""Aggregate all three benchmark results into one summary table."""
import json, re
from pathlib import Path
from nltk.metrics.distance import edit_distance
from jiwer import wer as compute_wer

def load(p):
    p = Path(p); return json.loads(p.read_text()) if p.exists() else None

def anls(pred, gt, thr=0.5):
    pred, gt = pred.lower().strip(), gt.lower().strip()
    if not gt: return 1.0 if not pred else 0.0
    nl = edit_distance(pred, gt) / max(len(pred), len(gt))
    return 0.0 if nl > thr else 1.0 - nl

def normalize(s):
    return re.sub(r"[^\w\s]", "", s.lower().strip()).strip()

print("\n" + "="*58)
print("  P16 BLV -- Extended Benchmark Results")
print("  Checkpoint: sft_patch_grpo_v2")
print("="*58)

ocr = load("ocr_bench/ocr_bench_preds.json")
if ocr:
    a_scores, w_scores = [], []
    for r in ocr:
        a_scores.append(anls(r["pred"], r["gt"]))
        try: w_scores.append(compute_wer(r["gt"], r["pred"]))
        except: w_scores.append(1.0)
    print(f"\n[1] OCR-Bench  (n={len(ocr)})")
    print(f"    ANLS : {sum(a_scores)/len(a_scores)*100:.2f}%")
    print(f"    WER  : {sum(w_scores)/len(w_scores)*100:.2f}%")
else:
    print("\n[1] OCR-Bench  -- results not found")

tvqa = load("textvqa/textvqa_preds.json")
if tvqa:
    def vqa_acc(pred, gts): return min(1.0, sum(normalize(g)==normalize(pred) for g in gts)/3.0)
    def best_anls(pred, gts): return max(anls(pred, g) for g in gts)
    accs  = [vqa_acc(r["pred"], r["gts"])   for r in tvqa]
    anlss = [best_anls(r["pred"], r["gts"]) for r in tvqa]
    print(f"\n[2] TextVQA val  (n={len(tvqa)})")
    print(f"    VQA Accuracy : {sum(accs)/len(accs)*100:.2f}%")
    print(f"    ANLS         : {sum(anlss)/len(anlss)*100:.2f}%")
else:
    print("\n[2] TextVQA val  -- results not found")

rtc = load("rtc_latency_results.json")
if rtc:
    print(f"\n[3] RTC Latency  (n={rtc['n_frames']} frames)")
    print(f"    Mean latency : {rtc['mean_ms']} ms")
    print(f"    P95  latency : {rtc['p95_ms']} ms")
    print(f"    FPS          : {rtc['fps']}")
else:
    print("\n[3] RTC Latency  -- results not found")

print("\n" + "="*58)
