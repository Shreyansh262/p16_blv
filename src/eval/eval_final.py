import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["HF_HOME"] = "/usershome/cs671_user2/p16_blv/models/.hf_cache"
os.chdir("/usershome/cs671_user2/p16_blv")

import gc, json, re, time, requests
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL = (
    "models/.hf_cache/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct"
    "/snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467"
)
CONDITIONS = {
    "base":           None,
    "sft_v2":         ("single", "models/student/sft_v2/best"),
    "grpo":           ("single", "models/student/grpo/best"),
    "sft_patch_v2":   ("chain",  "models/student/grpo/best",
                                 "models/student/sft_patch_grpo_v2"),
}
EVAL_DATA      = "data/augmented/balanced_splits/balanced_eval.json"
OUT_DIR        = Path("results/analysis/final")
JUDGE_URL      = "http://localhost:11434/api/generate"
JUDGE_MODEL    = "qwen2.5:32b"
MAX_NEW_TOKENS = 150
CKPT_EVERY     = 25

PROMPT = (
    "You are a BLV navigation assistant. Describe the scene for a blind user. "
    "Rules: environment type first, CAUTION only if real hazard present, "
    "meter distances for all objects and people, directional words, "
    "4 sentences max, present tense."
)

JUDGE_DIMS = [
    "spatial_orientation", "social_interaction", "action_events", "ambience",
    "descriptiveness", "objectivity", "accuracy", "clarity",
]

SEP  = "=" * 70
THIN = "-" * 50

def p(msg=""): print(msg, flush=True)

OUT_DIR.mkdir(parents=True, exist_ok=True)

p(SEP)
p("FINAL EVAL — 4 conditions on balanced_eval.json")
p(f"GPU: 1 (CUDA_VISIBLE_DEVICES=1 → cuda:0)")
p(f"Prompt: {PROMPT[:80]}...")
p(f"gen: max_new_tokens={MAX_NEW_TOKENS}, repetition_penalty=1.3, do_sample=False")
p(SEP)

# ── BLV keyword banks (from src/evaluation/blv_score.py) ─────────────────────
_SPATIAL = [
    "left","right","ahead","behind","above","below","near","far",
    "meters","feet","distance","position","next to","beside",
    "in front","to your","on your","at approximately","straight",
    "forward","close","nearby","across","steps away","directly",
    "on the left","on the right",
]
_SOCIAL = [
    "person","people","someone","man","woman","child","individual",
    "walking toward","approaching","facing","seated","standing near",
    "figure","human","group","crowd","pedestrian","user","you",
]
_ACTION = [
    "picks up","walking","sitting","standing","opens","closes",
    "carries","reaches","bends","turns","moves","steps","places",
    "puts","takes","holds","sets","lifts","pushes","pulls",
    "reaches for","grabs","uses","cooking","eating","running",
]
_AMBIENCE = [
    "indoor","outdoor","kitchen","bedroom","living room","street",
    "office","hallway","bathroom","dining","bright","dim","dark",
    "light","morning","evening","room","space","area","environment",
    "floor","ceiling","wall","natural light","artificial","carpeted",
    "tiled","wooden","outside","inside",
]
_OBSTACLE = [
    "obstacle","furniture","table","chair","door","wall","step",
    "curb","barrier","counter","desk","sofa","couch","bed",
    "cabinet","shelf","stair","column","pillar","railing","fence",
    "appliance","box","bin",
]
_STEP_KW = [
    "step up","step down","stairs","steps","curb","ramp","uneven",
    "elevation","threshold","ledge","staircase","raised","lowered",
    "slope","gradient",
]
_DIRECTION = [
    "left","right","straight","forward","behind","turn","head toward",
    "proceed","continue","go to","move toward","face","clockwise",
    "counterclockwise","bear left","bear right",
]
_MOVING_HAZARD = [
    "moving","approaching","vehicle","car","bicycle","bike",
    "coming toward","walking toward","running toward","rushing",
    "swinging","opening door","closing door","falling","rolling",
]
_DISTANCE = [
    "meters","feet","close","nearby","far","approximately","about",
    "distance","roughly","within","less than","more than",
    "steps away","meters away","a few",
]

def _has(text, kws):
    t = text.lower()
    return 1.0 if any(kw in t for kw in kws) else 0.0

def score_blv(text):
    return {
        "blv_spatial":  _has(text, _SPATIAL),
        "blv_social":   _has(text, _SOCIAL),
        "blv_action":   _has(text, _ACTION),
        "blv_ambience": _has(text, _AMBIENCE),
    }

def score_nav(text):
    return {
        "nav_obstacles":      _has(text, _OBSTACLE),
        "nav_step_changes":   _has(text, _STEP_KW),
        "nav_directions":     _has(text, _DIRECTION),
        "nav_moving_hazards": _has(text, _MOVING_HAZARD),
        "nav_distances":      _has(text, _DISTANCE),
    }

# ── Load eval data ────────────────────────────────────────────────────────────
with open(EVAL_DATA) as f:
    raw = json.load(f)

valid_entries = [e for e in raw if "keyframe_paths" in e]
valid_ids     = [e["video_id"] for e in valid_entries]
ref_map       = {e["video_id"]: e.get("blv_description", "") for e in valid_entries}

p(f"Total entries: {len(raw)} | With keyframes: {len(valid_entries)} | Synthetic skipped: {len(raw)-len(valid_entries)}")
p(f"Valid sample IDs locked — all 4 conditions will run on these exact {len(valid_entries)} samples.")
p()

# ── Model loader / unloader ───────────────────────────────────────────────────
def load_model(cond_key):
    spec = CONDITIONS[cond_key]
    proc = AutoProcessor.from_pretrained(BASE_MODEL)
    if hasattr(proc, "image_processor") and hasattr(proc.image_processor, "max_image_tiles"):
        proc.image_processor.max_image_tiles = 1

    base = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device_map={"": "cuda:0"})

    if spec is None:
        p(f"  Loaded base (no LoRA)")
        model = base
    elif spec[0] == "single":
        lora_path = spec[1]
        p(f"  Merging LoRA: {lora_path}")
        model = PeftModel.from_pretrained(base, lora_path).merge_and_unload()
    elif spec[0] == "chain":
        lora1, lora2 = spec[1], spec[2]
        p(f"  Merging LoRA 1: {lora1}")
        merged = PeftModel.from_pretrained(base, lora1).merge_and_unload()
        p(f"  Applying LoRA 2: {lora2}")
        model = PeftModel.from_pretrained(merged, lora2).merge_and_unload()

    model.eval()
    return model, proc

def unload_model(model, proc):
    del model, proc
    gc.collect()
    torch.cuda.empty_cache()

# ── Inference for one condition ───────────────────────────────────────────────
def run_inference(cond_key, out_file):
    p(f"\n{THIN}")
    p(f"INFERENCE: {cond_key}")
    p(THIN)

    # Resume
    if out_file.exists():
        with open(out_file) as f:
            results = json.load(f)
        done_ids = {r["video_id"] for r in results}
    else:
        results, done_ids = [], set()

    todo = [e for e in valid_entries if e["video_id"] not in done_ids]
    p(f"  {len(done_ids)} already done, {len(todo)} remaining")

    if not todo:
        p(f"  All done — skipping inference.")
        return results

    model, proc = load_model(cond_key)
    t0 = time.time()

    for i, entry in enumerate(todo):
        vid      = entry["video_id"]
        kf_paths = entry["keyframe_paths"][:4]
        ref      = ref_map[vid]

        try:
            frames = [Image.open(p_).convert("RGB") for p_ in kf_paths]
        except Exception as e:
            p(f"  [{len(done_ids)+i+1}/{len(valid_entries)}] {vid}: SKIP frame err: {e}")
            results.append({"video_id": vid, "reference": ref, "generated": ""})
            done_ids.add(vid)
            continue

        n       = len(frames)
        msgs    = [{"role": "user", "content":
                    [{"type": "image"}] * n + [{"type": "text", "text": PROMPT}]}]
        text_in = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs  = proc(text=text_in, images=frames, return_tensors="pt")
        inputs  = {k: v.to("cuda:0") for k, v in inputs.items()}
        plen    = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False, repetition_penalty=1.3)
        generated = proc.decode(out[0][plen:], skip_special_tokens=True).strip()

        results.append({"video_id": vid, "reference": ref, "generated": generated})
        done_ids.add(vid)

        elapsed = time.time() - t0
        done_so_far = i + 1
        rate    = done_so_far / elapsed
        eta     = (len(todo) - done_so_far) / max(rate, 1e-9)
        idx     = len(done_ids)
        p(f"  [{idx}/{len(valid_entries)}] {vid} | {len(generated.split())}w | ETA {eta/60:.1f}m")

        if done_so_far == 1:
            p(f"  ** First sample done in {elapsed:.1f}s → est. total {len(todo)*elapsed/60:.1f}m **")

        if done_so_far % CKPT_EVERY == 0:
            with open(out_file, "w") as f:
                json.dump(results, f, indent=2)
            p(f"  [ckpt] {len(results)} saved")

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    p(f"  Done. {len(results)} saved to {out_file}")

    unload_model(model, proc)
    return results

# ── NLP metrics ───────────────────────────────────────────────────────────────
def compute_nlp(results):
    import sacrebleu as sb
    from rouge_score import rouge_scorer
    import nltk
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    from nltk.translate.meteor_score import meteor_score as _meteor
    from nltk.tokenize import word_tokenize
    from collections import Counter
    import math

    valid = [(r["generated"], r["reference"])
             for r in results if r.get("generated","").strip()]
    hyps = [h for h, _ in valid]
    refs = [r for _, r in valid]

    bleu1 = sb.BLEU(max_ngram_order=1).corpus_score(hyps, [refs]).score
    bleu4 = sb.BLEU(max_ngram_order=4).corpus_score(hyps, [refs]).score

    rscorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = sum(rscorer.score(r, h)["rougeL"].fmeasure
                  for h, r in valid) / len(valid) * 100

    meteor_vals = []
    for h, r in valid:
        ht = word_tokenize(h.lower())
        rt = word_tokenize(r.lower())
        meteor_vals.append(_meteor([rt], ht) if ht else 0.0)
    meteor = sum(meteor_vals) / len(meteor_vals) * 100

    # CIDEr (TF-IDF cosine, 4-gram)
    def tok(s): return s.lower().split()
    def ngrams(toks, n): return [tuple(toks[i:i+n]) for i in range(len(toks)-n+1)]
    N = len(valid)
    doc_freq = Counter()
    for h, r in valid:
        for ng in set(ngrams(tok(r), 4)):
            doc_freq[ng] += 1
    scores = []
    for h, r in valid:
        hn = Counter(ngrams(tok(h), 4))
        rn = Counter(ngrams(tok(r), 4))
        if not hn or not rn:
            scores.append(0.0); continue
        num = den_h = den_r = 0.0
        for ng in set(hn) | set(rn):
            idf = math.log((N + 1) / (doc_freq.get(ng, 0) + 1))
            hw = hn.get(ng, 0) * idf
            rw = rn.get(ng, 0) * idf
            num += hw * rw; den_h += hw**2; den_r += rw**2
        scores.append(num / (math.sqrt(den_h) * math.sqrt(den_r))
                      if den_h > 0 and den_r > 0 else 0.0)
    cider = sum(scores) / len(scores)

    return {
        "n": len(valid),
        "BLEU-1": round(bleu1, 4), "BLEU-4": round(bleu4, 4),
        "ROUGE-L": round(rouge_l, 4), "METEOR": round(meteor, 4),
        "CIDEr": round(cider, 6),
    }

# ── BLV/NAV keyword scores ────────────────────────────────────────────────────
def compute_blv(results):
    valid = [r for r in results if r.get("generated","").strip()]
    def avg(key, fn):
        return round(sum(fn(r["generated"])[key] for r in valid) / len(valid), 4)

    blv_dims = ["blv_spatial","blv_social","blv_action","blv_ambience"]
    nav_dims  = ["nav_obstacles","nav_step_changes","nav_directions",
                 "nav_moving_hazards","nav_distances"]
    out = {"n": len(valid)}
    for k in blv_dims: out[k] = avg(k, score_blv)
    out["blv_mean"] = round(sum(out[k] for k in blv_dims) / 4, 4)
    for k in nav_dims: out[k] = avg(k, score_nav)
    out["nav_mean"] = round(sum(out[k] for k in nav_dims) / 5, 4)
    return out

# ── LLM judge ────────────────────────────────────────────────────────────────
JUDGE_TMPL = (
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
    "- spatial_orientation: directions (left/right/ahead/behind), distances in meters, spatial layout\n"
    "- social_interaction: people positions, movements, interpersonal dynamics\n"
    "- action_events: actions with correct temporal order and completeness\n"
    "- ambience: environment type, lighting, atmosphere\n"
    "- descriptiveness: spatial detail, hazard identification, obstacles, pathways\n"
    "- objectivity: factual only, no assumptions or subjective interpretation\n"
    "- accuracy: precise spatial relationships, object positions, distance estimates\n"
    "- clarity: logically sequenced for navigation decision-making"
)

def judge_one(ref, gen, retries=3):
    prompt = JUDGE_TMPL.format(reference=ref, generated=gen)
    for attempt in range(retries):
        try:
            resp = requests.post(
                JUDGE_URL,
                json={"model": JUDGE_MODEL, "prompt": prompt,
                      "stream": False, "options": {"temperature": 0}},
                timeout=120,
            )
            resp.raise_for_status()
            text = resp.json().get("response", "")
            # strip markdown fences if present
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"): text = text[4:]
            m = re.search(r'\{[^}]+\}', text, re.DOTALL)
            if m:
                scores = json.loads(m.group())
                if all(d in scores for d in JUDGE_DIMS):
                    return {d: int(scores[d]) for d in JUDGE_DIMS}
        except Exception as e:
            p(f"    judge attempt {attempt+1} err: {e}")
        time.sleep(3)
    return {d: 0 for d in JUDGE_DIMS}

def run_judge(cond_key, results, judge_file):
    p(f"\n{THIN}")
    p(f"JUDGE: {cond_key}")
    p(THIN)

    if judge_file.exists():
        with open(judge_file) as f:
            judged = json.load(f)
        done_ids = {r["video_id"] for r in judged}
    else:
        judged, done_ids = [], set()

    todo = [r for r in results
            if r["video_id"] not in done_ids and r.get("generated","").strip()]
    p(f"  {len(done_ids)} already scored, {len(todo)} remaining")

    t0 = time.time()
    for i, r in enumerate(todo):
        vid = r["video_id"]
        scores = judge_one(r["reference"], r["generated"])
        mcf = sum(scores[d] for d in JUDGE_DIMS[:4]) / 4
        naf = sum(scores[d] for d in JUDGE_DIMS[4:]) / 4
        judged.append({"video_id": vid, "scores": scores,
                       "MCF": round(mcf, 4), "NAF": round(naf, 4)})
        done_ids.add(vid)

        elapsed = time.time() - t0
        rate = (i + 1) / max(elapsed, 1e-9)
        eta  = (len(todo) - i - 1) / max(rate, 1e-9)
        p(f"  [{i+1}/{len(todo)}] {vid} MCF={mcf:.2f} NAF={naf:.2f} | ETA {eta/60:.1f}m")

        if (i + 1) % CKPT_EVERY == 0:
            with open(judge_file, "w") as f:
                json.dump(judged, f, indent=2)
            p(f"  [ckpt] {len(judged)} saved")

    with open(judge_file, "w") as f:
        json.dump(judged, f, indent=2)
    p(f"  Judge done. {len(judged)} scored → {judge_file}")
    return judged

def avg_judge(judged):
    valid = [r for r in judged if any(r["scores"][d] > 0 for d in JUDGE_DIMS)]
    if not valid:
        return {"n": 0}
    out = {"n": len(valid)}
    for d in JUDGE_DIMS:
        out[d] = round(sum(r["scores"][d] for r in valid) / len(valid), 4)
    out["MCF"] = round(sum(out[d] for d in JUDGE_DIMS[:4]) / 4, 4)
    out["NAF"] = round(sum(out[d] for d in JUDGE_DIMS[4:]) / 4, 4)
    return out

# ── MAIN ─────────────────────────────────────────────────────────────────────
COND_LABELS = {
    "base":         "A (base)",
    "sft_v2":       "B (SFT_v2)",
    "grpo":         "C (SFT_v2+GRPO)",
    "sft_patch_v2": "D (sft_patch_v2)",
}
COND_ORDER = ["base", "sft_v2", "grpo", "sft_patch_v2"]
OUT_FILES = {c: OUT_DIR / f"{c}_outputs.json"     for c in COND_ORDER}
JUDGE_FILES = {c: OUT_DIR / f"{c}_judge.json"     for c in COND_ORDER}

# ── PHASE 1: Inference ────────────────────────────────────────────────────────
p(SEP); p("PHASE 1 — INFERENCE (all 4 conditions)"); p(SEP)
all_results = {}
for cond in COND_ORDER:
    all_results[cond] = run_inference(cond, OUT_FILES[cond])

# ── PHASE 2: NLP + BLV metrics ───────────────────────────────────────────────
p(f"\n{SEP}"); p("PHASE 2 — NLP + BLV METRICS"); p(SEP)
nlp_all = {}
blv_all = {}
for cond in COND_ORDER:
    p(f"\nComputing NLP + BLV for {cond} ...")
    nlp_all[cond] = compute_nlp(all_results[cond])
    blv_all[cond] = compute_blv(all_results[cond])
    p(f"  NLP: {nlp_all[cond]}")
    p(f"  BLV: {blv_all[cond]}")

with open(OUT_DIR / "nlp_metrics_final.json", "w") as f:
    json.dump(nlp_all, f, indent=2)
with open(OUT_DIR / "blv_scores_final.json", "w") as f:
    json.dump(blv_all, f, indent=2)
p(f"\nNLP + BLV metrics saved.")

# ── PHASE 3: LLM judge ────────────────────────────────────────────────────────
p(f"\n{SEP}"); p("PHASE 3 — LLM JUDGE (qwen2.5:32b)"); p(SEP)
judge_avgs = {}
for cond in COND_ORDER:
    judged = run_judge(cond, all_results[cond], JUDGE_FILES[cond])
    judge_avgs[cond] = avg_judge(judged)

with open(OUT_DIR / "judge_scores_final.json", "w") as f:
    json.dump(judge_avgs, f, indent=2)

# ── PHASE 4: Final table ──────────────────────────────────────────────────────
p(f"\n{SEP}"); p("FINAL CONSOLIDATED TABLE"); p(SEP)

def fmt(v, pct=False):
    if v is None or v == "": return "  —   "
    if pct:   return f"{v*100:6.2f}%"
    return f"{v:7.4f}"

def fmtf(v):
    if v is None: return "  —   "
    return f"{v:7.4f}"

cols = COND_ORDER
W = 16
hdr = f"{'Metric':<28}" + "".join(f"{COND_LABELS[c]:>{W}}" for c in cols) + f"{'Δ D-A':>10}"
p(hdr)
p("-" * (28 + W * 4 + 10))

def row(label, getter, pct=False, fmt4=False):
    vals = []
    for c in cols:
        try:   v = getter(c)
        except: v = None
        vals.append(v)
    delta = None
    if vals[0] is not None and vals[-1] is not None:
        delta = vals[-1] - vals[0]
    cells = []
    for v in vals:
        if v is None: cells.append(f"{'—':>{W}}")
        elif pct:     cells.append(f"{v*100:>{W}.2f}")
        elif fmt4:    cells.append(f"{v:>{W}.4f}")
        else:         cells.append(f"{v:>{W}.4f}")
    dcell = ""
    if delta is not None:
        if pct:   dcell = f"{delta*100:>+10.2f}"
        else:     dcell = f"{delta:>+10.4f}"
    p(f"{label:<28}" + "".join(cells) + dcell)

p(f"\n{'— NLP METRICS —'}")
row("BLEU-1",  lambda c: nlp_all[c]["BLEU-1"])
row("BLEU-4",  lambda c: nlp_all[c]["BLEU-4"])
row("ROUGE-L", lambda c: nlp_all[c]["ROUGE-L"])
row("METEOR",  lambda c: nlp_all[c]["METEOR"])
row("CIDEr",   lambda c: nlp_all[c]["CIDEr"])

p(f"\n{'— BLV KEYWORD SCORES —'}")
row("blv_spatial",      lambda c: blv_all[c]["blv_spatial"], pct=True)
row("blv_social",       lambda c: blv_all[c]["blv_social"],  pct=True)
row("blv_action",       lambda c: blv_all[c]["blv_action"],  pct=True)
row("blv_ambience",     lambda c: blv_all[c]["blv_ambience"],pct=True)
row("blv_mean",         lambda c: blv_all[c]["blv_mean"],    pct=True)
row("nav_obstacles",    lambda c: blv_all[c]["nav_obstacles"],    pct=True)
row("nav_step_changes", lambda c: blv_all[c]["nav_step_changes"], pct=True)
row("nav_directions",   lambda c: blv_all[c]["nav_directions"],   pct=True)
row("nav_moving_hazards",lambda c:blv_all[c]["nav_moving_hazards"],pct=True)
row("nav_distances",    lambda c: blv_all[c]["nav_distances"],    pct=True)
row("nav_mean",         lambda c: blv_all[c]["nav_mean"],         pct=True)

p(f"\n{'— LLM JUDGE SCORES (1-10) —'}")
row("MCF",                  lambda c: judge_avgs[c]["MCF"])
row("  spatial_orientation",lambda c: judge_avgs[c]["spatial_orientation"])
row("  social_interaction", lambda c: judge_avgs[c]["social_interaction"])
row("  action_events",      lambda c: judge_avgs[c]["action_events"])
row("  ambience",           lambda c: judge_avgs[c]["ambience"])
row("NAF",                  lambda c: judge_avgs[c]["NAF"])
row("  descriptiveness",    lambda c: judge_avgs[c]["descriptiveness"])
row("  objectivity",        lambda c: judge_avgs[c]["objectivity"])
row("  accuracy",           lambda c: judge_avgs[c]["accuracy"])
row("  clarity",            lambda c: judge_avgs[c]["clarity"])

p(f"\n{SEP}")
p("=== ALL DONE ===")
