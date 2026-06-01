import json, os
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

REF_JSON = "data/augmented/balanced_splits/balanced_eval.json"
OUT_DIR  = "results/benchmark_graph"

ref_map = {s["video_id"]: s.get("blv_description", s.get("original_caption",""))
           for s in json.load(open(REF_JSON))}

MODEL_FILES = {
    "SmolVLM2-500M-base":     "SmolVLM2-500M-base_outputs.json",
    "sft_patch_grpo_v2-ours": "sft_patch_grpo_v2-ours_outputs.json",
    "SmolVLM2-2.2B":          "SmolVLM2-2.2B_outputs.json",
    "Qwen2.5-VL-3B":          "Qwen2.5-VL-3B_outputs.json",
    "PaliGemma2-3B":          "PaliGemma2-3B_outputs.json",
    "moondream2":             "moondream2_outputs.json",
}

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
sf = SmoothingFunction().method3
nlp_results = {}

for model_id, fname in MODEL_FILES.items():
    fpath = os.path.join(OUT_DIR, fname)
    if not os.path.exists(fpath): continue
    outputs = json.load(open(fpath))

    refs_bleu, hyps_bleu, rouge_scores = [], [], []

    for item in outputs:
        ref = ref_map.get(item["video_id"], "")
        hyp = item.get("output", "")
        if not ref or not hyp: continue
        ref_tok = ref.lower().split()
        hyp_tok = hyp.lower().split()
        refs_bleu.append([ref_tok])
        hyps_bleu.append(hyp_tok)
        rouge_scores.append(scorer.score(ref, hyp)["rougeL"].fmeasure)

    bleu1 = corpus_bleu(refs_bleu, hyps_bleu, weights=(1,0,0,0), smoothing_function=sf)
    bleu4 = corpus_bleu(refs_bleu, hyps_bleu, weights=(.25,.25,.25,.25), smoothing_function=sf)

    try:
        from nltk.translate.meteor_score import meteor_score
        meteor_scores = [meteor_score([r[0]], h) for r,h in zip(refs_bleu, hyps_bleu)]
        met = np.mean(meteor_scores)
    except:
        met = 0.0

    nlp_results[model_id] = {
        "BLEU1":   round(bleu1*100, 2),
        "BLEU4":   round(bleu4*100, 2),
        "ROUGE_L": round(np.mean(rouge_scores)*100, 2),
        "METEOR":  round(met*100, 2),
        "n":       len(hyps_bleu),
    }
    r = nlp_results[model_id]
    print(f"{model_id}: BLEU1={r['BLEU1']} BLEU4={r['BLEU4']} ROUGE-L={r['ROUGE_L']} METEOR={r['METEOR']}")

json.dump(nlp_results, open(f"{OUT_DIR}/nlp_metrics.json","w"), indent=2)
print("\nSaved -> results/benchmark_graph/nlp_metrics.json")
