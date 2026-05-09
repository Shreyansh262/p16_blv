import json, requests, time, argparse, os, random

OLLAMA_URL  = "http://localhost:11435/api/chat"
JUDGE_MODEL = "qwen2.5:32b"
LOG_PATH    = "logs/eval_judge_32b.log"
N_SAMPLES   = 200
SEED        = 42
TIMEOUT_S   = 60    # 32b on dedicated GPU 6: ~2-3s/sample warm

JUDGE_PROMPT = (
    "Reference description (by an expert audio describer):\n{reference}\n\n"
    "Generated description (model output to evaluate):\n{generated}\n\n"
    "Score the generated description on 8 dimensions from 1-10.\n"
    "Return ONLY a valid JSON object - no explanation, no markdown, no extra text:\n\n"
    '{{\n'
    '  "spatial_orientation": 0,\n'
    '  "social_interaction": 0,\n'
    '  "action_events": 0,\n'
    '  "ambience": 0,\n'
    '  "descriptiveness": 0,\n'
    '  "objectivity": 0,\n'
    '  "accuracy": 0,\n'
    '  "clarity": 0\n'
    '}}\n\n'
    "Rubric:\n"
    "- spatial_orientation: mentions directions (left/right/ahead/behind), distances in meters, spatial layout\n"
    "- social_interaction: describes people positions, movements, interpersonal dynamics\n"
    "- action_events: actions described with correct temporal order and completeness\n"
    "- ambience: environment type, lighting, atmosphere captured\n"
    "- descriptiveness: spatial detail, hazard identification, obstacles, pathways present\n"
    "- objectivity: factual only, no assumptions or subjective interpretation\n"
    "- accuracy: precise spatial relationships, object positions, distance estimates\n"
    "- clarity: logically sequenced for navigation decision-making"
)

def judge_sample(reference, generated, retries=3):
    for attempt in range(retries):
        try:
            resp = requests.post(OLLAMA_URL, json={
                "model": JUDGE_MODEL,
                "messages": [
                    {"role": "system", "content": "You are an expert accessibility evaluator for blind and low-vision users."},
                    {"role": "user", "content": JUDGE_PROMPT.format(reference=reference, generated=generated)}
                ],
                "options": {"temperature": 0},
                "stream": False
            }, timeout=TIMEOUT_S)
            text = resp.json()["message"]["content"].strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text.strip())
        except requests.exceptions.Timeout:
            log(f"    Timeout on attempt {attempt+1}/{retries} (>{TIMEOUT_S}s)")
        except requests.exceptions.ConnectionError as e:
            log(f"    Connection error attempt {attempt+1}: {e}")
        except json.JSONDecodeError as e:
            log(f"    JSON parse error attempt {attempt+1}: {e} | text={text[:80]!r}")
        except Exception as e:
            log(f"    Unexpected error attempt {attempt+1}: {type(e).__name__}: {e}")
        if attempt < retries - 1:
            time.sleep(5)
    return None

def log(msg):
    print(msg, flush=True)
    os.makedirs("logs", exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(msg + "\n")

parser = argparse.ArgumentParser()
parser.add_argument("--conditions", nargs="+", default=["A","B","C","D"])
args = parser.parse_args()

os.makedirs("results/scores", exist_ok=True)
os.chdir("/usershome/cs671_user2/p16_blv")

# Verify ollama is up
try:
    r = requests.get("http://localhost:11434/api/tags", timeout=10)
    models = [m["name"] for m in r.json()["models"]]
    log(f"Ollama OK. Models: {models}")
    if JUDGE_MODEL not in models:
        log(f"ERROR: {JUDGE_MODEL} not available!")
        exit(1)
except Exception as e:
    log(f"ERROR: Cannot reach ollama: {e}")
    exit(1)

all_results = {}
random.seed(SEED)

for cond in args.conditions:
    path = f"results/inference/conditions/condition_{cond}_outputs.json"
    if not os.path.exists(path):
        log(f"Condition {cond}: output file not found, skipping.")
        continue

    data = json.load(open(path))
    subset = random.sample(data, min(N_SAMPLES, len(data)))
    log(f"\n=== Condition {cond}: scoring {len(subset)}/{len(data)} samples (seed={SEED}) ===")

    raw_scores = []
    failed = 0
    t_start = time.time()

    for i, sample in enumerate(subset):
        t0 = time.time()
        scores = judge_sample(sample["reference"], sample["generated"])
        elapsed = time.time() - t0

        if scores is None:
            failed += 1
            log(f"  [{cond}] Sample {i} FAILED ({elapsed:.1f}s)")
            continue

        scores["video_id"] = sample["video_id"]
        raw_scores.append(scores)

        if (i + 1) % 10 == 0 or (i + 1) == len(subset):
            done = i + 1
            rate = (time.time() - t_start) / done
            eta  = rate * (len(subset) - done)
            log(f"  [{cond}] {done}/{len(subset)} done | {elapsed:.1f}s/sample | ETA {eta/60:.1f}min | failed={failed}")

    json.dump(raw_scores, open(f"results/scores/condition_{cond}_llm_scores.json", "w"), indent=2)

    dims = ["spatial_orientation","social_interaction","action_events","ambience",
            "descriptiveness","objectivity","accuracy","clarity"]
    avgs = {}
    for d in dims:
        vals = [s[d] for s in raw_scores if d in s and isinstance(s[d], (int, float))]
        avgs[d] = round(sum(vals)/len(vals), 3) if vals else 0

    avgs["MCF_score"] = round((avgs["spatial_orientation"]+avgs["social_interaction"]+avgs["action_events"]+avgs["ambience"])/4, 3)
    avgs["NAF_score"] = round((avgs["descriptiveness"]+avgs["objectivity"]+avgs["accuracy"]+avgs["clarity"])/4, 3)
    avgs["total_samples"] = len(raw_scores)
    avgs["failed"] = failed

    all_results[cond] = avgs
    json.dump(avgs, open(f"results/scores/condition_{cond}_llm_avg.json", "w"), indent=2)
    log(f"  [{cond}] DONE: MCF={avgs['MCF_score']} NAF={avgs['NAF_score']} (scored={len(raw_scores)}, failed={failed})")

json.dump(all_results, open("results/scores/llm_judge_all.json", "w"), indent=2)

print("\n" + "="*80)
print(f"FINAL LLM JUDGE RESULTS ({JUDGE_MODEL}, n={N_SAMPLES}/condition)")
print("="*80)
dims_display = [
    ("spatial_orientation", "Spatial Orientation"),
    ("social_interaction",  "Social Interaction"),
    ("action_events",       "Action & Events"),
    ("ambience",            "Ambience"),
    ("MCF_score",           "--- MCF Score ---"),
    ("descriptiveness",     "Descriptiveness"),
    ("objectivity",         "Objectivity"),
    ("accuracy",            "Accuracy"),
    ("clarity",             "Clarity"),
    ("NAF_score",           "--- NAF Score ---"),
]
conds = [c for c in ["A","B","C","D","E"] if c in all_results]
col_labels = {"A":"A(Base)","B":"B(SFTv2)","C":"C(DPO)","D":"D(RLAIF)","E":"E(SFTv3)"}
header = f"{'Metric':<25}" + "".join(f"{col_labels[c]:>12}" for c in conds)
print(header)
print("-"*73)
for key, label in dims_display:
    vals = [all_results[c].get(key, 0) for c in conds]
    best = max(vals)
    row = f"{label:<25}"
    for c, v in zip(conds, vals):
        marker = "*" if v == best and not label.startswith("---") else " "
        row += f"{str(v)+marker:>12}"
    print(row)

print(f"\nSaved: results/scores/llm_judge_all.json")
