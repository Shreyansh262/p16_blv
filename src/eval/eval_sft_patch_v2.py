import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["HF_HOME"] = "/usershome/cs671_user2/p16_blv/models/.hf_cache"
os.chdir("/usershome/cs671_user2/p16_blv")

import gc, json, time, requests, re
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

# ── Config ──────────────────────────────────────────────────────────────────
BASE_MODEL = (
    "models/.hf_cache/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct"
    "/snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467"
)
GRPO_PATH    = "models/student/grpo/best"
SFT_V2_PATH  = "models/student/sft_patch_grpo_v2"
EVAL_DATA    = "data/augmented/balanced_splits/balanced_eval.json"
OUT_DIR      = Path("results/scores")
OUT_INFER    = OUT_DIR / "sft_patch_v2_outputs.json"
OUT_JUDGE    = OUT_DIR / "sft_patch_v2_judge_scores.json"
OUT_NLP      = OUT_DIR / "sft_patch_v2_nlp_metrics.json"
JUDGE_URL    = "http://localhost:11434/api/generate"
JUDGE_MODEL  = "qwen2.5:32b"
MAX_NEW_TOKENS = 150
CKPT_EVERY   = 25

INSTRUCTION = (
    "You are a BLV navigation assistant. Describe the scene for a blind user. "
    "Rules: environment type first, CAUTION only if real hazard present, "
    "meter distances for all objects and people, directional words, "
    "4 sentences max, present tense."
)

JUDGE_DIMS = [
    "spatial_orientation", "social_interaction", "action_events", "ambience",
    "descriptiveness", "objectivity", "accuracy", "clarity"
]

SEP  = "=" * 70
THIN = "-" * 50

def p(msg=""):
    print(msg, flush=True)

OUT_DIR.mkdir(parents=True, exist_ok=True)

p(SEP)
p("EVAL: sft_patch_grpo_v2 on balanced_eval.json (469 videos)")
p(f"Base model : {BASE_MODEL}")
p(f"GRPO path  : {GRPO_PATH}")
p(f"SFT v2 path: {SFT_V2_PATH}")
p(f"Eval data  : {EVAL_DATA}")
p(SEP)

# ── Load eval data ──────────────────────────────────────────────────────────
with open(EVAL_DATA) as f:
    eval_data = json.load(f)
p(f"Loaded {len(eval_data)} eval entries.")

# ── Load model ──────────────────────────────────────────────────────────────
p("\nLoading processor ...")
proc = AutoProcessor.from_pretrained(BASE_MODEL)
if hasattr(proc, "image_processor") and hasattr(proc.image_processor, "max_image_tiles"):
    proc.image_processor.max_image_tiles = 1
    p("  max_image_tiles=1")

p(f"\nLoading base model ...")
base = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL, dtype=torch.bfloat16, device_map={"": "cuda:0"})

p(f"Merging GRPO adapter from {GRPO_PATH} ...")
merged = PeftModel.from_pretrained(base, GRPO_PATH).merge_and_unload()

p(f"Applying sft_patch_grpo_v2 adapter from {SFT_V2_PATH} ...")
model = PeftModel.from_pretrained(merged, SFT_V2_PATH).merge_and_unload()
model.eval()
p("Model ready.\n")

# ── STEP 1: Inference ───────────────────────────────────────────────────────
p(SEP)
p(f"STEP 1 — INFERENCE ({len(eval_data)} videos)")
p(SEP)

# Resume support
if OUT_INFER.exists():
    with open(OUT_INFER) as f:
        results = json.load(f)
    done_ids = {r["video_id"] for r in results}
    p(f"Resuming: {len(results)} already done.")
else:
    results = []
    done_ids = set()

t0 = time.time()
for i, entry in enumerate(eval_data):
    vid = entry["video_id"]
    if vid in done_ids:
        continue

    ref = entry.get("blv_description", entry.get("original_caption", ""))

    if "keyframe_paths" not in entry:
        p(f"  [{i+1}/{len(eval_data)}] {vid}: SKIP (no keyframe_paths — synthetic text-only entry)")
        results.append({"video_id": vid, "reference": ref, "generated": ""})
        done_ids.add(vid)
        continue

    kf_paths = entry["keyframe_paths"][:4]

    try:
        frames = [Image.open(p_).convert("RGB") for p_ in kf_paths]
    except Exception as e:
        p(f"  [{i+1}/{len(eval_data)}] {vid}: SKIP (frame error: {e})")
        results.append({"video_id": vid, "reference": ref, "generated": ""})
        done_ids.add(vid)
        continue

    n = len(frames)
    messages = [{"role": "user", "content":
                 [{"type": "image"}] * n + [{"type": "text", "text": INSTRUCTION}]}]
    text_in = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs  = proc(text=text_in, images=frames, return_tensors="pt")
    inputs  = {k: v.to("cuda:0") for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.3,
        )
    generated = proc.decode(out[0][prompt_len:], skip_special_tokens=True).strip()

    results.append({"video_id": vid, "reference": ref, "generated": generated})
    done_ids.add(vid)

    elapsed = time.time() - t0
    rate = (i + 1) / elapsed
    remaining = (len(eval_data) - i - 1) / max(rate, 1e-9)
    p(f"  [{i+1}/{len(eval_data)}] {vid} | gen_len={len(generated.split())}w | "
      f"ETA {remaining/60:.1f}m")

    if (i + 1) % CKPT_EVERY == 0:
        with open(OUT_INFER, "w") as f:
            json.dump(results, f, indent=2)
        p(f"  [ckpt] Saved {len(results)} to {OUT_INFER}")

with open(OUT_INFER, "w") as f:
    json.dump(results, f, indent=2)
p(f"\nInference complete. Saved {len(results)} to {OUT_INFER}")

# Free GPU memory
del model, merged, base
gc.collect()
torch.cuda.empty_cache()
p("GPU memory freed.\n")

# ── STEP 2: LLM Judge ───────────────────────────────────────────────────────
p(SEP)
p(f"STEP 2 — LLM JUDGE ({JUDGE_MODEL})")
p(SEP)

JUDGE_PROMPT_TMPL = """\
You are an expert evaluator of audio descriptions for blind and low-vision (BLV) users.

Score the following generated description against the reference on these 8 dimensions.
Return ONLY valid JSON with integer scores 1-10.

Reference: {reference}
Generated: {generated}

Dimensions:
- spatial_orientation: Does it describe spatial layout, directions, distances?
- social_interaction: Does it note people, their actions, social context?
- action_events: Does it describe what is happening / dynamic events?
- ambience: Does it convey environment type, lighting, atmosphere?
- descriptiveness: Is it detailed and informative?
- objectivity: Is it factual without unnecessary subjective commentary?
- accuracy: Does the generated match the reference scene?
- clarity: Is it clear and easy to understand?

Return JSON like:
{{"spatial_orientation":7,"social_interaction":5,"action_events":6,"ambience":8,"descriptiveness":7,"objectivity":9,"accuracy":6,"clarity":8}}
"""

def judge_one(ref, gen, retries=3):
    prompt = JUDGE_PROMPT_TMPL.format(reference=ref, generated=gen)
    for attempt in range(retries):
        try:
            resp = requests.post(
                JUDGE_URL,
                json={"model": JUDGE_MODEL, "prompt": prompt, "stream": False},
                timeout=120,
            )
            resp.raise_for_status()
            text = resp.json().get("response", "")
            m = re.search(r'\{[^}]+\}', text, re.DOTALL)
            if m:
                scores = json.loads(m.group())
                if all(d in scores for d in JUDGE_DIMS):
                    return scores
        except Exception as e:
            p(f"    judge attempt {attempt+1} failed: {e}")
        time.sleep(2)
    return {d: 0 for d in JUDGE_DIMS}

# Resume support
if OUT_JUDGE.exists():
    with open(OUT_JUDGE) as f:
        judge_results = json.load(f)
    judged_ids = {r["video_id"] for r in judge_results}
    p(f"Resuming judge: {len(judge_results)} already scored.")
else:
    judge_results = []
    judged_ids = set()

ref_map = {r["video_id"]: r for r in results}

t0 = time.time()
for i, entry in enumerate(results):
    vid = entry["video_id"]
    if vid in judged_ids:
        continue

    ref = entry["reference"]
    gen = entry["generated"]

    if not gen.strip():
        scores = {d: 0 for d in JUDGE_DIMS}
    else:
        scores = judge_one(ref, gen)

    mcf = sum(scores[d] for d in ["spatial_orientation","social_interaction","action_events","ambience"]) / 4
    naf = sum(scores[d] for d in ["descriptiveness","objectivity","accuracy","clarity"]) / 4

    judge_results.append({
        "video_id": vid,
        "scores": scores,
        "MCF_Score": round(mcf, 4),
        "NAF_Score": round(naf, 4),
    })
    judged_ids.add(vid)

    elapsed = time.time() - t0
    rate = (i + 1) / max(elapsed, 1e-9)
    remaining = (len(results) - i - 1) / max(rate, 1e-9)
    p(f"  [{i+1}/{len(results)}] {vid} | MCF={mcf:.2f} NAF={naf:.2f} | ETA {remaining/60:.1f}m")

    if (i + 1) % CKPT_EVERY == 0:
        with open(OUT_JUDGE, "w") as f:
            json.dump(judge_results, f, indent=2)
        p(f"  [ckpt] Saved {len(judge_results)} judge scores")

with open(OUT_JUDGE, "w") as f:
    json.dump(judge_results, f, indent=2)
p(f"\nJudge complete. Saved {len(judge_results)} to {OUT_JUDGE}")

# ── STEP 3: NLP Metrics ─────────────────────────────────────────────────────
p(SEP)
p("STEP 3 — NLP METRICS")
p(SEP)

import sacrebleu
from rouge_score import rouge_scorer
import nltk
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
from nltk.translate.meteor_score import meteor_score as _meteor
from nltk.tokenize import word_tokenize

hypotheses = [r["generated"] for r in results]
references = [r["reference"]  for r in results]

# BLEU-1
bleu1 = sacrebleu.BLEU(max_ngram_order=1).corpus_score(
    hypotheses, [references]).score

# BLEU-4
bleu4 = sacrebleu.BLEU(max_ngram_order=4).corpus_score(
    hypotheses, [references]).score

# ROUGE-L
rscorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
rouge_scores = [rscorer.score(ref, hyp)["rougeL"].fmeasure
                for ref, hyp in zip(references, hypotheses)]
rouge_l = sum(rouge_scores) / max(len(rouge_scores), 1) * 100

# METEOR
meteor_scores = []
for ref, hyp in zip(references, hypotheses):
    ref_tok  = word_tokenize(ref.lower())
    hyp_tok  = word_tokenize(hyp.lower())
    if hyp_tok:
        meteor_scores.append(_meteor([ref_tok], hyp_tok))
    else:
        meteor_scores.append(0.0)
meteor = sum(meteor_scores) / max(len(meteor_scores), 1) * 100

# CIDEr (simple TF-IDF approximation)
try:
    from collections import Counter
    import math

    def tokenize_simple(s):
        return s.lower().split()

    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    def cider_score(hyps, refs):
        n = 4
        scores = []
        corpus_refs = [tokenize_simple(r) for r in refs]
        corpus_hyps = [tokenize_simple(h) for h in hyps]
        doc_freq = Counter()
        for ref_tok in corpus_refs:
            for ng in set(get_ngrams(ref_tok, n)):
                doc_freq[ng] += 1
        N = len(refs)
        for hyp_tok, ref_tok in zip(corpus_hyps, corpus_refs):
            hyp_ng  = Counter(get_ngrams(hyp_tok, n))
            ref_ng  = Counter(get_ngrams(ref_tok, n))
            if not hyp_ng or not ref_ng:
                scores.append(0.0)
                continue
            num = den_h = den_r = 0.0
            all_ng = set(hyp_ng) | set(ref_ng)
            for ng in all_ng:
                idf = math.log((N + 1) / (doc_freq.get(ng, 0) + 1))
                h_w = hyp_ng.get(ng, 0) * idf
                r_w = ref_ng.get(ng, 0) * idf
                num   += h_w * r_w
                den_h += h_w ** 2
                den_r += r_w ** 2
            if den_h > 0 and den_r > 0:
                scores.append(num / (math.sqrt(den_h) * math.sqrt(den_r)))
            else:
                scores.append(0.0)
        return sum(scores) / max(len(scores), 1)

    cider = cider_score(hypotheses, references)
except Exception as e:
    p(f"CIDEr error: {e}")
    cider = 0.0

nlp_metrics = {
    "n_samples": len(results),
    "BLEU-1":   round(bleu1,  4),
    "BLEU-4":   round(bleu4,  4),
    "ROUGE-L":  round(rouge_l, 4),
    "METEOR":   round(meteor,  4),
    "CIDEr":    round(cider,   4),
}
with open(OUT_NLP, "w") as f:
    json.dump(nlp_metrics, f, indent=2)

p(f"NLP metrics: {nlp_metrics}")
p(f"Saved to {OUT_NLP}")

# ── STEP 4: Comparison Table ────────────────────────────────────────────────
p(SEP)
p("STEP 4 — COMPARISON TABLE")
p(SEP)

# Load existing baselines
baseline_file = OUT_DIR / "nlp_metrics_ABCD.json"
if not baseline_file.exists():
    baseline_file = OUT_DIR / "nlp_metrics_ABC.json"

baselines = {}
if baseline_file.exists():
    with open(baseline_file) as f:
        raw = json.load(f)
    # Support both dict-of-dicts and other formats
    if isinstance(raw, dict):
        baselines = raw

label_map = {
    "A": "base",
    "B": "SFT_v2",
    "C": "DPO",
    "D": "RLAIF",
}

# Judge averages for sft_patch_v2
if judge_results:
    avg_mcf = sum(r["MCF_Score"] for r in judge_results) / len(judge_results)
    avg_naf = sum(r["NAF_Score"] for r in judge_results) / len(judge_results)
else:
    avg_mcf = avg_naf = 0.0

p(f"\n{'Condition':<20} {'BLEU-1':>7} {'BLEU-4':>7} {'ROUGE-L':>8} {'METEOR':>8} {'CIDEr':>7} {'MCF':>6} {'NAF':>6}")
p("-" * 75)

for key, label in label_map.items():
    if key in baselines:
        m = baselines[key]
        b1 = m.get("BLEU-1", m.get("bleu1", 0))
        b4 = m.get("BLEU-4", m.get("bleu4", 0))
        rl = m.get("ROUGE-L", m.get("rouge_l", 0))
        mt = m.get("METEOR", m.get("meteor", 0))
        cd = m.get("CIDEr", m.get("cider", 0))
        mcf_v = m.get("MCF_Score", m.get("mcf", "-"))
        naf_v = m.get("NAF_Score", m.get("naf", "-"))
        mcf_s = f"{mcf_v:.2f}" if isinstance(mcf_v, float) else str(mcf_v)
        naf_s = f"{naf_v:.2f}" if isinstance(naf_v, float) else str(naf_v)
        p(f"{label:<20} {b1:>7.2f} {b4:>7.2f} {rl:>8.2f} {mt:>8.2f} {cd:>7.4f} {mcf_s:>6} {naf_s:>6}")

p(f"{'sft_patch_grpo_v2':<20} "
  f"{nlp_metrics['BLEU-1']:>7.2f} "
  f"{nlp_metrics['BLEU-4']:>7.2f} "
  f"{nlp_metrics['ROUGE-L']:>8.2f} "
  f"{nlp_metrics['METEOR']:>8.2f} "
  f"{nlp_metrics['CIDEr']:>7.4f} "
  f"{avg_mcf:>6.2f} "
  f"{avg_naf:>6.2f}")

p(SEP)
p("=== ALL DONE ===")
