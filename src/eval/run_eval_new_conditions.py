import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ["HF_HOME"] = "/usershome/cs671_user2/p16_blv/models/.hf_cache"
os.chdir("/usershome/cs671_user2/p16_blv")

import gc, json, re, time, glob, requests
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_MODEL = (
    "models/.hf_cache/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct"
    "/snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467"
)
CONDITIONS = {
    "base":   None,
    "sft_v2": "models/student/sft_v2/best",
    "grpo":   "models/student/grpo/best",
}
EVAL_DATA      = "data/augmented/balanced_splits/balanced_eval.json"
OUT_DIR        = Path("results/analysis/new_conditions")
JUDGE_URL      = "http://localhost:11434/api/generate"
JUDGE_MODEL    = "qwen2.5:32b"
KEYFRAME_ROOT  = "data/keyframes"
MAX_NEW_TOKENS = 300
CKPT_EVERY     = 25

INFER_PROMPT = (
    "Describe this scene for a blind or low-vision person. "
    "Focus on spatial layout, objects, people, hazards, and navigation cues. "
    "Be specific and concise."
)

JUDGE_TMPL = """\
You are an expert evaluator for audio descriptions designed to assist blind and low-vision (BLV) users.

Evaluate the following generated description against the reference description on 8 dimensions.
Score each dimension from 1 to 10.

Reference description:
{reference}

Generated description:
{generated}

Score these 8 dimensions (1-10):
1. spatial_orientation: How well does it convey spatial layout, directions, distances?
2. social_interaction: How well does it describe people, their positions, actions, relationships?
3. action_events: How accurately and completely are actions and events described?
4. ambience: How well is the environment, lighting, and atmosphere captured?
5. descriptiveness: How detailed and specific is the spatial and hazard information?
6. objectivity: How factual and free from subjective interpretation is the description?
7. accuracy: How precise are spatial relationships, object positions, distances?
8. clarity: How logically organized and navigation-friendly is the description?

Respond ONLY with a JSON object with keys: spatial_orientation, social_interaction,
action_events, ambience, descriptiveness, objectivity, accuracy, clarity.
All values must be integers between 1 and 10."""

DIMS = ["spatial_orientation", "social_interaction", "action_events", "ambience",
        "descriptiveness", "objectivity", "accuracy", "clarity"]

# ── HELPERS ───────────────────────────────────────────────────────────────────
def p(msg=""):
    print(msg, flush=True)

SEP  = "=" * 70
THIN = "-" * 50

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
with open(EVAL_DATA) as f:
    eval_data = json.load(f)
ref_map = {e["video_id"]: e["blv_description"] for e in eval_data}
p(SEP)
p(f"Eval data: {len(eval_data)} entries from {EVAL_DATA}")
p(f"Conditions: {list(CONDITIONS.keys())}")
p(f"Output dir: {OUT_DIR}")
p(SEP)

# ── KEYFRAME LOADER ────────────────────────────────────────────────────────────
def load_frames(entry):
    paths = [p_ for p_ in entry.get("keyframe_paths", [])[:4] if os.path.exists(p_)]
    if not paths:
        vid = entry["video_id"]
        found = sorted(glob.glob(f"{KEYFRAME_ROOT}/**/{vid}*.jpg", recursive=True))[:4]
        paths = found
    frames = []
    for p_ in paths:
        try:
            frames.append(Image.open(p_).convert("RGB"))
        except Exception as e:
            p(f"    [WARN] cannot load {p_}: {e}")
    return frames

# ── MODEL LOADER ───────────────────────────────────────────────────────────────
def load_model(condition_path):
    p(f"  Loading base: {BASE_MODEL}")
    proc = AutoProcessor.from_pretrained(BASE_MODEL)
    if hasattr(proc, "image_processor") and hasattr(proc.image_processor, "max_image_tiles"):
        proc.image_processor.max_image_tiles = 1
    base = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"}
    )
    if condition_path is None:
        p("  No adapter — running base model standalone.")
        model = base
    else:
        p(f"  Loading adapter: {condition_path}")
        model = PeftModel.from_pretrained(base, condition_path).merge_and_unload()
    model.eval()
    return model, proc

def unload_model(model, base=None):
    del model
    if base is not None:
        del base
    torch.cuda.empty_cache()
    gc.collect()

# ── INFERENCE ─────────────────────────────────────────────────────────────────
def run_inference(cond_name, condition_path):
    out_file  = OUT_DIR / f"{cond_name}_outputs.json"
    ckpt_file = OUT_DIR / f"{cond_name}_outputs_ckpt.json"

    # Resume support
    existing = {}
    if out_file.exists():
        with open(out_file) as f:
            existing = json.load(f)
        if len(existing) > 10:
            p(f"  [{cond_name}] {len(existing)} outputs already exist — skipping inference.")
            return existing

    p(f"\n{SEP}")
    p(f"INFERENCE — {cond_name}")
    p(SEP)

    model, proc = load_model(condition_path)
    outputs = dict(existing)

    for i, entry in enumerate(eval_data, 1):
        vid = entry["video_id"]
        if vid in outputs:
            continue

        frames = load_frames(entry)
        if not frames:
            p(f"  [{cond_name}] [{i}/{len(eval_data)}] {vid} — no frames, skipping")
            outputs[vid] = "[ERROR: no frames]"
            continue

        n = len(frames)
        messages = [{"role": "user", "content":
                     [{"type": "image"}] * n + [{"type": "text", "text": INFER_PROMPT}]}]
        text_in    = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs     = proc(text=text_in, images=frames, return_tensors="pt")
        inputs     = {k: v.to("cuda:0") for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        try:
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            caption = proc.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
        except Exception as e:
            p(f"  [{cond_name}] [{i}/{len(eval_data)}] {vid} — generate error: {e}")
            caption = f"[ERROR: {e}]"

        outputs[vid] = caption

        if i % CKPT_EVERY == 0 or i == len(eval_data):
            with open(ckpt_file, "w") as f:
                json.dump(outputs, f, indent=2)
            p(f"  [{cond_name}] {i}/{len(eval_data)} done — checkpoint saved")

    with open(out_file, "w") as f:
        json.dump(outputs, f, indent=2)
    p(f"  [{cond_name}] Inference complete — {len(outputs)} outputs saved to {out_file}")

    unload_model(model)
    return outputs

# ── JUDGE ─────────────────────────────────────────────────────────────────────
def call_judge(reference, generated, retries=3):
    prompt = JUDGE_TMPL.format(reference=reference, generated=generated)
    for attempt in range(retries):
        try:
            resp = requests.post(JUDGE_URL, json={
                "model": JUDGE_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0}
            }, timeout=120)
            text = resp.json().get("response", "").strip()
            # strip markdown fences
            if "```" in text:
                m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
                text = m.group(1) if m else re.sub(r"```[a-z]*", "", text).strip()
            # extract first JSON object
            m = re.search(r"\{.*?\}", text, re.DOTALL)
            if not m:
                raise ValueError(f"No JSON found in: {text[:120]!r}")
            scores = json.loads(m.group())
            for dim in DIMS:
                if dim not in scores:
                    raise ValueError(f"Missing dim: {dim}")
                scores[dim] = max(1, min(10, int(scores[dim])))
            return scores
        except Exception as e:
            p(f"    [JUDGE] attempt {attempt+1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(5)
    return None

def run_judge(cond_name, outputs):
    scores_file = OUT_DIR / f"{cond_name}_judge_scores.json"

    existing = {}
    if scores_file.exists():
        with open(scores_file) as f:
            existing = json.load(f)
        p(f"  [{cond_name}] Resuming judge: {len(existing)} already scored.")

    all_scores = dict(existing)
    todo = [
        (vid, ref_map[vid], outputs[vid])
        for vid in outputs
        if vid not in all_scores
        and vid in ref_map
        and outputs[vid]
        and not outputs[vid].startswith("[ERROR")
    ]

    p(f"\n{SEP}")
    p(f"JUDGE — {cond_name}  ({len(todo)} to score, {len(existing)} already done)")
    p(SEP)

    t0 = time.time()
    for i, (vid, ref, gen) in enumerate(todo, 1):
        scores = call_judge(ref, gen)
        if scores is None:
            p(f"  [{cond_name}] [{i}/{len(todo)}] {vid} — judge failed, skipping")
            continue

        scores["MCF_Score"] = round(
            (scores["spatial_orientation"] + scores["social_interaction"] +
             scores["action_events"] + scores["ambience"]) / 4, 4)
        scores["NAF_Score"] = round(
            (scores["descriptiveness"] + scores["objectivity"] +
             scores["accuracy"] + scores["clarity"]) / 4, 4)
        all_scores[vid] = scores

        if i % CKPT_EVERY == 0 or i == len(todo):
            with open(scores_file, "w") as f:
                json.dump(all_scores, f, indent=2)
            elapsed = time.time() - t0
            rate    = elapsed / i
            eta_min = rate * (len(todo) - i) / 60
            p(f"  [{cond_name}] {i}/{len(todo)} judged | {rate:.1f}s/sample | ETA {eta_min:.1f}min")

    with open(scores_file, "w") as f:
        json.dump(all_scores, f, indent=2)
    p(f"  [{cond_name}] Judge complete — {len(all_scores)} scored")
    return all_scores

# ── NLP METRICS ───────────────────────────────────────────────────────────────
def compute_nlp(cond_name, outputs):
    pairs = [
        (outputs[e["video_id"]], e["blv_description"])
        for e in eval_data
        if e["video_id"] in outputs
        and outputs[e["video_id"]]
        and not outputs[e["video_id"]].startswith("[ERROR")
    ]
    hyps = [h for h, _ in pairs]
    refs = [r for _, r in pairs]

    metrics = {"n": len(pairs)}

    try:
        import sacrebleu as sb
        metrics["BLEU1"] = round(sb.corpus_bleu(hyps, [refs], max_ngram_order=1).score, 4)
        metrics["BLEU4"] = round(sb.corpus_bleu(hyps, [refs]).score, 4)
    except Exception as e:
        p(f"  [{cond_name}] BLEU failed: {e}")
        metrics["BLEU1"] = metrics["BLEU4"] = None

    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        rl = [scorer.score(r, h)["rougeL"].fmeasure for h, r in pairs]
        metrics["ROUGE_L"] = round(sum(rl) / len(rl), 4)
    except Exception as e:
        p(f"  [{cond_name}] ROUGE-L failed: {e}")
        metrics["ROUGE_L"] = None

    try:
        import nltk
        for pkg in ["punkt", "wordnet", "punkt_tab"]:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass
        from nltk.translate.meteor_score import meteor_score
        from nltk.tokenize import word_tokenize
        scores = [
            meteor_score([word_tokenize(r)], word_tokenize(h))
            for h, r in pairs
        ]
        metrics["METEOR"] = round(sum(scores) / len(scores), 4)
    except Exception as e:
        p(f"  [{cond_name}] METEOR failed: {e}")
        metrics["METEOR"] = None

    return metrics

# ── AVERAGE JUDGE SCORES ──────────────────────────────────────────────────────
def avg_scores(judge_scores):
    avgs = {}
    for dim in DIMS + ["MCF_Score", "NAF_Score"]:
        vals = [v[dim] for v in judge_scores.values() if dim in v]
        avgs[dim] = round(sum(vals) / len(vals), 4) if vals else None
    avgs["n"] = len(judge_scores)
    return avgs

# ── PRINT TABLE ───────────────────────────────────────────────────────────────
def print_table(summary):
    conds = list(summary.keys())
    rows = [
        ("MCF_Score",           "MCF_Score"),
        ("NAF_Score",           "NAF_Score"),
        ("spatial_orientation", "spatial_orientation"),
        ("social_interaction",  "social_interaction"),
        ("action_events",       "action_events"),
        ("ambience",            "ambience"),
        ("descriptiveness",     "descriptiveness"),
        ("objectivity",         "objectivity"),
        ("accuracy",            "accuracy"),
        ("clarity",             "clarity"),
        ("BLEU1",               "BLEU-1"),
        ("BLEU4",               "BLEU-4"),
        ("ROUGE_L",             "ROUGE-L"),
        ("METEOR",              "METEOR"),
        ("n_videos",            "n_videos"),
    ]
    col_w = 10
    p(f"\n{'Metric':<25}" + "".join(f"{c:>{col_w}}" for c in conds))
    p("-" * (25 + col_w * len(conds)))
    for key, label in rows:
        if key == "n_videos":
            p("-" * (25 + col_w * len(conds)))
        row = f"{label:<25}"
        for c in conds:
            v = summary[c].get(key)
            if v is None:
                row += f"{'N/A':>{col_w}}"
            elif isinstance(v, int):
                row += f"{v:>{col_w}}"
            else:
                row += f"{v:>{col_w}.4f}"
        p(row)

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    p(SEP)
    p("BLV EVAL — base / sft_v2 / grpo — balanced_eval.json (469 videos)")
    p(f"Judge: {JUDGE_MODEL} @ {JUDGE_URL}")
    p(SEP)

    # quick judge ping
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=10)
        models = [m["name"] for m in r.json()["models"]]
        p(f"Ollama OK. Models: {models}")
        if JUDGE_MODEL not in models:
            raise RuntimeError(f"{JUDGE_MODEL} not in ollama model list!")
    except Exception as e:
        p(f"[FATAL] Cannot reach ollama: {e}")
        raise

    summary = {}
    all_outputs   = {}
    all_judge     = {}

    # ── Phase 1: inference for all conditions ─────────────────────────────────
    for cond_name, cond_path in CONDITIONS.items():
        all_outputs[cond_name] = run_inference(cond_name, cond_path)

    # ── Phase 2: judge scoring for all conditions ─────────────────────────────
    for cond_name in CONDITIONS:
        all_judge[cond_name] = run_judge(cond_name, all_outputs[cond_name])

    # ── Phase 3: NLP metrics + assemble summary ───────────────────────────────
    p(f"\n{SEP}\nNLP METRICS\n{SEP}")
    for cond_name in CONDITIONS:
        p(f"\n  [{cond_name}] Computing NLP metrics ...")
        avgs = avg_scores(all_judge[cond_name])
        nlp  = compute_nlp(cond_name, all_outputs[cond_name])
        summary[cond_name] = {
            "condition":          cond_name,
            "n_videos":           avgs["n"],
            "MCF_Score":          avgs["MCF_Score"],
            "NAF_Score":          avgs["NAF_Score"],
            "spatial_orientation":avgs["spatial_orientation"],
            "social_interaction": avgs["social_interaction"],
            "action_events":      avgs["action_events"],
            "ambience":           avgs["ambience"],
            "descriptiveness":    avgs["descriptiveness"],
            "objectivity":        avgs["objectivity"],
            "accuracy":           avgs["accuracy"],
            "clarity":            avgs["clarity"],
            "BLEU1":              nlp.get("BLEU1"),
            "BLEU4":              nlp.get("BLEU4"),
            "ROUGE_L":            nlp.get("ROUGE_L"),
            "METEOR":             nlp.get("METEOR"),
        }
        p(f"  [{cond_name}] NLP done: BLEU4={nlp.get('BLEU4')} ROUGE-L={nlp.get('ROUGE_L')} METEOR={nlp.get('METEOR')}")

    # ── Save summary ──────────────────────────────────────────────────────────
    summary_file = OUT_DIR / "summary_table.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    p(f"\nSummary saved → {summary_file}")

    # ── Print table ───────────────────────────────────────────────────────────
    p(f"\n{SEP}")
    p("FINAL RESULTS TABLE")
    p(SEP)
    print_table(summary)
    p(f"\n{SEP}")
    p("=== EVAL COMPLETE ===")


if __name__ == "__main__":
    main()
