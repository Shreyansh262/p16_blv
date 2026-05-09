import os, json, time, requests, re
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

BALANCED_DIR = Path("results/inference/balanced")
OUT_DIR      = Path("results/analysis/balanced_scores")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OLLAMA_URL   = "http://localhost:11434/api/generate"
JUDGE_MODEL  = "qwen2.5:32b"
TIMEOUT      = 300

CONDITIONS = [
    ("A", "condition_A_base_outputs.json"),
    ("B", "condition_B_sft_v2_outputs.json"),
    ("C", "condition_C_dpo_outputs.json"),
    ("D", "condition_D_rlaif_outputs.json"),
    ("E", "condition_E_sft_v3_outputs.json"),
]

JUDGE_PROMPT = """You are an expert evaluator for BLV (Blind and Low Vision) assistive AI systems.

Given a reference description and a model-generated description of a video scene, score the model output on these dimensions (1-10 each):

1. spatial_orientation: Does it mention positions, directions, distances? (left/right/ahead/3m away)
2. social_interaction: Are people's positions and movements described?
3. action_events: Are actions described with enough detail?
4. ambience: Is the environment type and atmosphere described?
5. hazard_id: Are obstacles, steps, or hazards mentioned?
6. accuracy: Does the description match what the reference describes?
7. clarity: Is the description clear and easy to understand?
8. overall: Overall quality for a blind person navigating this scene.

Reference: {reference}

Model output: {prediction}

Respond ONLY with a JSON object like:
{{"spatial_orientation": 3, "social_interaction": 5, "action_events": 4, "ambience": 6, "hazard_id": 2, "accuracy": 4, "clarity": 7, "overall": 4}}"""

def score_one(reference, prediction):
    prompt = JUDGE_PROMPT.format(reference=reference, prediction=prediction)
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": JUDGE_MODEL, "prompt": prompt,
            "stream": False, "options": {"temperature": 0, "num_predict": 200}
        }, timeout=TIMEOUT)
        text = resp.json().get("response", "")
        match = re.search(r'\{.*?\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        print(f"  Judge error: {e}")
    return None

for cond_name, fname in CONDITIONS:
    input_file = BALANCED_DIR / fname
    out_file   = OUT_DIR / f"scores_{cond_name}.json"

    if not input_file.exists():
        print(f"Skipping {cond_name} -- file not found")
        continue

    with open(input_file) as f:
        items = json.load(f)

    # Resume support
    existing = {}
    if out_file.exists():
        with open(out_file) as f:
            existing = {x["video_id"]: x for x in json.load(f)}

    results = list(existing.values())
    print(f"\n=== Condition {cond_name} -- {len(items)} videos, {len(results)} already scored ===")

    for i, item in enumerate(items):
        vid = item["video_id"]
        if vid in existing:
            continue

        scores = score_one(item["reference"], item["prediction"])
        if scores is None:
            print(f"  [{i+1}/{len(items)}] {vid} -- judge failed, skipping")
            continue

        scores["video_id"] = vid
        results.append(scores)

        if (i+1) % 10 == 0:
            with open(out_file, "w") as f:
                json.dump(results, f, indent=2)
            dims = ["spatial_orientation","social_interaction","action_events",
                    "ambience","hazard_id","accuracy","clarity","overall"]
            avgs = {d: sum(r.get(d,0) for r in results)/len(results) for d in dims}
            mcf  = sum(avgs[d] for d in ["spatial_orientation","social_interaction","action_events","ambience"])/4
            naf  = sum(avgs[d] for d in ["spatial_orientation","hazard_id","accuracy","clarity"])/4
            print(f"  [{i+1}/{len(items)}] {vid} | MCF={mcf:.2f} NAF={naf:.2f} overall={avgs['overall']:.2f}")

    # Final save
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    dims = ["spatial_orientation","social_interaction","action_events",
            "ambience","hazard_id","accuracy","clarity","overall"]
    avgs = {d: sum(r.get(d,0) for r in results)/len(results) for d in dims if results}
    mcf  = sum(avgs[d] for d in ["spatial_orientation","social_interaction","action_events","ambience"])/4
    naf  = sum(avgs[d] for d in ["spatial_orientation","hazard_id","accuracy","clarity"])/4
    print(f"\nCondition {cond_name} FINAL: MCF={mcf:.2f} NAF={naf:.2f} overall={avgs.get('overall',0):.2f}")

print("\n=== All conditions scored ===")
