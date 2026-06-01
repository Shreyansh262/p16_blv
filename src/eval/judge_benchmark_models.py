"""
BLV judge scoring for all 6 benchmark models.
Scores 8 dims (1-10) via qwen2.5:32b on ollama port 11435.
Input: results/benchmark_graph/*_outputs.json
Output: results/benchmark_graph/judge_scores.json
"""

import sys, types, importlib.util
spec = importlib.util.spec_from_loader("torchvision", loader=None)
tv_mod = importlib.util.module_from_spec(spec); tv_mod.__spec__ = spec
sys.modules["torchvision"] = tv_mod

import json, os, re, time, requests
import numpy as np

OLLAMA_URL  = "http://localhost:11435/api/generate"
JUDGE_MODEL = "qwen2.5:32b"
OUT_DIR     = "results/benchmark_graph"
DIMS = ["spatial_orientation","social_interaction","action_events",
        "ambience","descriptiveness","objectivity","accuracy","clarity"]

JUDGE_PROMPT = """You are evaluating an AI-generated scene description for blind and low-vision (BLV) users.

Reference description (ground truth):
{reference}

Model output:
{output}

Score the model output on these 8 dimensions (1-10 each):
- spatial_orientation: mentions directions, positions, distances
- social_interaction: describes people's positions and movements
- action_events: describes actions with sufficient detail
- ambience: describes environment type and atmosphere
- descriptiveness: level of visual detail provided
- objectivity: factual, no interpretation or assumptions
- accuracy: consistent with reference scene content
- clarity: clear, well-structured language

Respond ONLY with valid JSON, no other text:
{{"spatial_orientation": X, "social_interaction": X, "action_events": X, "ambience": X, "descriptiveness": X, "objectivity": X, "accuracy": X, "clarity": X}}"""

MODEL_FILES = {
    "SmolVLM2-500M-base":     "SmolVLM2-500M-base_outputs.json",
    "sft_patch_grpo_v2-ours": "sft_patch_grpo_v2-ours_outputs.json",
    "SmolVLM2-2.2B":          "SmolVLM2-2.2B_outputs.json",
    "Qwen2.5-VL-3B":          "Qwen2.5-VL-3B_outputs.json",
    "PaliGemma2-3B":          "PaliGemma2-3B_outputs.json",
    "moondream2":             "moondream2_outputs.json",
}

# Load reference descriptions from balanced_eval
ref_map = {}
for s in json.load(open("data/augmented/balanced_splits/balanced_eval.json")):
    ref_map[s["video_id"]] = s.get("blv_description", s.get("original_caption", ""))

def call_judge(reference, output, retries=3):
    prompt = JUDGE_PROMPT.format(reference=reference[:500], output=output[:500])
    for attempt in range(retries):
        try:
            r = requests.post(OLLAMA_URL, json={
                "model": JUDGE_MODEL, "prompt": prompt,
                "stream": False, "options": {"temperature": 0, "num_predict": 200}
            }, timeout=120)
            text = r.json()["response"].strip()
            m = re.search(r'\{[^}]+\}', text, re.DOTALL)
            if m:
                scores = json.loads(m.group())
                if all(d in scores for d in DIMS):
                    return {d: float(scores[d]) for d in DIMS}
        except Exception as e:
            print(f"    judge error (attempt {attempt+1}): {e}")
            time.sleep(2)
    return None

def score_model(model_id, fname):
    fpath = os.path.join(OUT_DIR, fname)
    if not os.path.exists(fpath):
        print(f"  MISSING: {fpath}")
        return None
    outputs = json.load(open(fpath))
    print(f"\n[{model_id}] scoring {len(outputs)} samples...", flush=True)

    all_scores = []
    for i, item in enumerate(outputs):
        vid = item["video_id"]
        ref = ref_map.get(vid, "")
        out = item.get("output", "")
        if not ref or not out:
            continue
        scores = call_judge(ref, out)
        if scores:
            all_scores.append(scores)
        if (i+1) % 10 == 0:
            if all_scores:
                mcf = np.mean([np.mean([s["spatial_orientation"],s["social_interaction"],
                               s["action_events"],s["ambience"]]) for s in all_scores])
                print(f"  [{i+1}/{len(outputs)}] MCF so far: {mcf:.2f}", flush=True)

    if not all_scores:
        return None

    avg = {d: round(np.mean([s[d] for s in all_scores]), 3) for d in DIMS}
    avg["MCF"] = round(np.mean([avg["spatial_orientation"], avg["social_interaction"],
                                avg["action_events"],       avg["ambience"]]), 3)
    avg["NAF"] = round(np.mean([avg["spatial_orientation"], avg["descriptiveness"],
                                avg["accuracy"],            avg["clarity"]]), 3)
    avg["n"]   = len(all_scores)
    print(f"  DONE -- MCF={avg['MCF']} NAF={avg['NAF']} (n={avg['n']})", flush=True)
    return avg

def main():
    results = {}
    for model_id, fname in MODEL_FILES.items():
        r = score_model(model_id, fname)
        if r:
            results[model_id] = r
        json.dump(results, open(f"{OUT_DIR}/judge_scores.json","w"), indent=2)
        print(f"  saved -> {OUT_DIR}/judge_scores.json", flush=True)

    print("\n\n=== BLV JUDGE RESULTS ===")
    print(f"{'Model':<30} {'MCF':>6} {'NAF':>6} {'Spatial':>8} {'Ambience':>9} {'n':>5}")
    print("-"*65)
    for mid, s in results.items():
        print(f"{mid:<30} {s['MCF']:>6.2f} {s['NAF']:>6.2f} {s['spatial_orientation']:>8.2f} {s['ambience']:>9.2f} {s['n']:>5}")

if __name__ == "__main__":
    main()
