#!/usr/bin/env python3
"""
run_eval.py — Full LLM-judge ablation evaluation: Conditions A, B, C
Saves all outputs to ~/p16_blv/results/analysis/
Designed to run inside tmux; crash-safe (resumes from checkpoints).

Conditions:
  A — SmolVLM2-500M baseline (no fine-tuning)
  B — SmolVLM2-500M + SFT LoRA  (sft_checkpoint/)
  C — SmolVLM2-500M + SFT merged + DPO LoRA  (dpo_checkpoint/)

Judge: gpt-oss:20b via Ollama API (already running on localhost:11434)
"""

import os, sys, json, random, time, logging, traceback, requests
from pathlib import Path
from datetime import datetime

# ── CUDA must be set before torch import ──────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["HF_HOME"] = "/usershome/cs671_user2/p16_blv/models/.hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/usershome/cs671_user2/p16_blv/models/.hf_cache"

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR   = Path("/usershome/cs671_user2/p16_blv/results/analysis")
CAPTIONS_FILE = Path("/usershome/cs671_user2/p16_blv/data/generated/all_captions.json")
DPO_PAIRS_FILE= Path("/usershome/cs671_user2/p16_blv/data/generated/dpo_pairs.json")
BASE_MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
HF_CACHE      = "/usershome/cs671_user2/p16_blv/models/.hf_cache"
SFT_PATH      = "/usershome/cs671_user2/p16_blv/models/student/sft_checkpoint"
DPO_PATH      = "/usershome/cs671_user2/p16_blv/models/student/dpo_checkpoint"
OLLAMA_URL    = "http://localhost:11434/api/generate"
OLLAMA_MODEL  = "gpt-oss:20b"

N_CHARADES    = 500
N_AVCAPS      = 200
SEED          = 42
MAX_NEW_TOKENS= 256
DEVICE        = "cuda:0"

# ── Output files ──────────────────────────────────────────────────────────────
TEST_SET_FILE      = RESULTS_DIR / "held_out_test_set.json"
GROUND_TRUTH_FILE  = RESULTS_DIR / "ground_truth.json"
PROGRESS_LOG       = RESULTS_DIR / "progress.log"
SANITY_LOG         = RESULTS_DIR / "sanity_check_log.txt"
NLP_METRICS_FILE   = RESULTS_DIR / "nlp_metrics.json"
FINAL_TABLE_FILE   = RESULTS_DIR / "final_results_table.json"
SUMMARY_FILE       = RESULTS_DIR / "summary.txt"

COND_OUTPUTS = {
    "A": RESULTS_DIR / "condition_A_outputs.json",
    "B": RESULTS_DIR / "condition_B_outputs.json",
    "C": RESULTS_DIR / "condition_C_outputs.json",
}
JUDGE_SCORES = {
    "A": RESULTS_DIR / "judge_scores_condition_A.json",
    "B": RESULTS_DIR / "judge_scores_condition_B.json",
    "C": RESULTS_DIR / "judge_scores_condition_C.json",
}

# ── Sanity check ranges (from paper Tables 6-7, Condition A baseline) ─────────
SANITY_RANGES = {
    "spatial_orientation": (2.9, 3.6),
    "social_interaction":  (3.2, 3.3),
    "action_events":       (1.9, 2.6),
    "ambience":            (3.5, 4.9),
    "descriptiveness":     (2.8, 3.6),
    "objectivity":         (4.9, 5.2),
    "accuracy":            (3.0, 3.5),
    "clarity":             (3.1, 3.9),
}

DIMS = ["spatial_orientation", "social_interaction", "action_events", "ambience",
        "descriptiveness", "objectivity", "accuracy", "clarity"]

# ── Logging ───────────────────────────────────────────────────────────────────
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROGRESS_LOG),
    ],
)
log = logging.getLogger(__name__)


def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def plog(msg):
    """Progress log with timestamp."""
    log.info(msg)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Build held-out test set
# ─────────────────────────────────────────────────────────────────────────────

def build_test_set():
    if TEST_SET_FILE.exists():
        plog(f"[STEP 2] Loading existing test set from {TEST_SET_FILE}")
        with open(TEST_SET_FILE) as f:
            return json.load(f)

    plog("[STEP 2] Building held-out test set (500 Charades + 200 AVCaps)...")
    with open(CAPTIONS_FILE) as f:
        all_caps = json.load(f)
    with open(DPO_PAIRS_FILE) as f:
        dpo_data = json.load(f)

    # Collect all video_ids used in DPO training (train + val splits)
    used = set()
    for split in ("train", "val"):
        for item in dpo_data.get(split, []):
            if isinstance(item, dict) and "video_id" in item:
                used.add(item["video_id"])
    plog(f"  DPO-excluded IDs: {len(used)}")

    charades, avcaps = [], []
    for entry in all_caps:
        if entry["video_id"] in used:
            continue
        # Require at least one existing keyframe and a ground truth description
        kf_ok = [p for p in entry.get("keyframe_paths", []) if Path(p).exists()]
        if not kf_ok:
            continue
        if not entry.get("blv_description", "").strip():
            continue
        row = dict(entry)
        row["_kf"] = kf_ok
        if row["dataset"] == "charades":
            charades.append(row)
        else:
            avcaps.append(row)

    plog(f"  Eligible held-out: {len(charades)} Charades, {len(avcaps)} AVCaps")

    rng = random.Random(SEED)
    n_ch = min(N_CHARADES, len(charades))
    n_av = min(N_AVCAPS,   len(avcaps))
    selected = rng.sample(charades, n_ch) + rng.sample(avcaps, n_av)
    rng.shuffle(selected)
    plog(f"  Selected: {n_ch} Charades + {n_av} AVCaps = {len(selected)} total")

    # Save without the ephemeral _kf field
    saveable = [{k: v for k, v in e.items() if k != "_kf"} for e in selected]
    with open(TEST_SET_FILE, "w") as f:
        json.dump(saveable, f, indent=2)
    plog(f"  Saved: {TEST_SET_FILE}")
    return selected


def hydrate_kf(test_set):
    """Re-attach _kf (existing keyframe paths) after loading from disk."""
    for entry in test_set:
        entry["_kf"] = [p for p in entry.get("keyframe_paths", []) if Path(p).exists()]
    return test_set


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Ground truth
# ─────────────────────────────────────────────────────────────────────────────

def build_ground_truth(test_set):
    if GROUND_TRUTH_FILE.exists():
        plog(f"[STEP 3] Ground truth already exists at {GROUND_TRUTH_FILE}")
        with open(GROUND_TRUTH_FILE) as f:
            return json.load(f)

    plog("[STEP 3] Building ground truth from existing Qwen2.5-VL-7B blv_descriptions...")
    gt = {}
    missing = 0
    for entry in test_set:
        desc = entry.get("blv_description", "").strip()
        if desc:
            gt[entry["video_id"]] = desc
        else:
            missing += 1
    plog(f"  Ground truth: {len(gt)} entries, {missing} missing")
    with open(GROUND_TRUTH_FILE, "w") as f:
        json.dump(gt, f, indent=2)
    plog(f"  Saved: {GROUND_TRUTH_FILE}")
    return gt


# ─────────────────────────────────────────────────────────────────────────────
# STEPS 4-6: Model inference (Conditions A / B / C)
# ─────────────────────────────────────────────────────────────────────────────

AD_PROMPT = (
    "You are an audio description assistant for blind and low-vision users. "
    "Describe this video following VideoA11y AD guidelines:\n"
    "1. State the scene environment first (indoor/outdoor, room type, lighting).\n"
    "2. Describe people: appearance, position relative to the viewer, distance.\n"
    "3. Describe actions in chronological sequence.\n"
    "4. Call out navigational hazards: steps, obstacles, moving objects.\n"
    "5. Use directional language: left, right, ahead, behind, approximately N metres.\n"
    "6. Be objective and factual — no assumptions or interpretations.\n"
    "Provide a concise, clear description of 2-4 sentences."
)


def load_images_for_inference(kf_paths):
    from PIL import Image
    imgs = []
    for p in kf_paths:
        try:
            img = Image.open(p).convert("RGB")
            if max(img.size) > 364:
                img.thumbnail((364, 364), Image.LANCZOS)
            imgs.append(img)
        except Exception as e:
            log.warning("Cannot load frame %s: %s", p, e)
    return imgs


def run_smolvlm(model, processor, images):
    import torch
    if not images:
        return "[ERROR: no keyframes]"
    messages = [{
        "role": "user",
        "content": (
            [{"type": "image", "image": img} for img in images]
            + [{"type": "text", "text": AD_PROMPT}]
        ),
    }]
    text   = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors="pt").to(DEVICE)
    if "pixel_values" in inputs and inputs["pixel_values"] is not None:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    n_in = inputs["input_ids"].shape[1]
    gen  = out_ids[0][n_in:]
    if len(gen) <= 2:
        return "[COLLAPSED]"
    return processor.decode(gen, skip_special_tokens=True).strip()


def load_condition(cond):
    import torch
    from transformers import AutoProcessor, SmolVLMForConditionalGeneration
    from peft import PeftModel

    plog(f"  Loading condition {cond} model...")
    base = SmolVLMForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map={"": 0},
        cache_dir=HF_CACHE,
    )

    if cond == "A":
        model     = base
        processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, cache_dir=HF_CACHE)

    elif cond == "B":
        model     = PeftModel.from_pretrained(base, SFT_PATH, torch_dtype=torch.float16)
        processor = AutoProcessor.from_pretrained(SFT_PATH)

    elif cond == "C":
        plog("  Merging SFT LoRA into base weights...")
        from peft import PeftModel
        sft    = PeftModel.from_pretrained(base, SFT_PATH, torch_dtype=torch.float16)
        merged = sft.merge_and_unload()
        plog("  Applying DPO LoRA...")
        model     = PeftModel.from_pretrained(merged, DPO_PATH, torch_dtype=torch.float16)
        processor = AutoProcessor.from_pretrained(DPO_PATH)

    model.eval()
    plog(f"  Condition {cond} ready.")
    return model, processor


def unload_model(model):
    import torch
    del model
    torch.cuda.empty_cache()


def generate_condition(cond, test_set, label):
    out_file = COND_OUTPUTS[cond]
    plog(f"[{label}] Generating outputs for condition {cond}...")

    existing = {}
    if out_file.exists():
        with open(out_file) as f:
            existing = json.load(f)
        plog(f"  Resuming: {len(existing)} already done.")

    todo = [e for e in test_set if e["video_id"] not in existing]
    if not todo:
        plog(f"  All {len(existing)} outputs already present for condition {cond}.")
        return existing

    model, processor = load_condition(cond)
    outputs = dict(existing)

    for i, entry in enumerate(todo, 1):
        vid = entry["video_id"]
        try:
            images = load_images_for_inference(entry["_kf"])
            desc   = run_smolvlm(model, processor, images)
        except Exception as e:
            log.error("  [%d/%d] %s FAILED: %s", i, len(todo), vid, e)
            desc = f"[ERROR: {e}]"
        outputs[vid] = desc
        if i % 10 == 0 or i == len(todo):
            with open(out_file, "w") as f:
                json.dump(outputs, f, indent=2)
            plog(f"  [{i}/{len(todo)}] checkpoint saved.")

    unload_model(model)
    with open(out_file, "w") as f:
        json.dump(outputs, f, indent=2)
    plog(f"  Condition {cond} outputs saved: {out_file}")
    return outputs


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: LLM Judge scoring
# ─────────────────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = (
    "You are an expert BLV accessibility evaluator following the VideoA11y methodology. "
    "Score a model-generated video description against a ground truth reference on 8 dimensions "
    "(1-10 integer scale). Evaluate strictly from the perspective of a blind user depending on "
    "this description for navigation and safety."
)

JUDGE_USER_TMPL = """Ground Truth: {gt}
Model Output: {hyp}

Score on these 8 dimensions (1-10 each):

MULTI-CONTEXT BLV FRAMEWORK:
1. spatial_orientation — Location descriptions, directional cues (left/right/ahead), relative positioning, environmental layout for mental mapping.
2. social_interaction — Person identification, interpersonal dynamics, emotional expressions, social context.
3. action_events — Temporal sequence clarity, activity description completeness, causal relationships between events.
4. ambience — Mood, lighting conditions, environmental atmosphere, sensory details.

NAVIGATIONAL ASSISTANCE FRAMEWORK:
5. descriptiveness — Spatial layout detail, hazard identification, environmental features (obstacles, pathways, boundaries).
6. objectivity — Factual reporting without assumptions, no subjective interpretations.
7. accuracy — Precision in spatial relationships, object positions, distance estimations.
8. clarity — Information organized for sequential navigation, logical flow, unambiguous directional references.

Respond ONLY with JSON:
{{"spatial_orientation": <int>, "social_interaction": <int>, "action_events": <int>, "ambience": <int>, "descriptiveness": <int>, "objectivity": <int>, "accuracy": <int>, "clarity": <int>}}"""


def call_judge(gt, hyp, retries=3):
    """Call gpt-oss:20b via Ollama API. Returns dict of scores or None.

    gpt-oss:20b is a thinking/reasoning model: thinking goes in the
    "thinking" field; the final JSON answer goes in "response".
    Do NOT set num_predict (cuts off thinking -> empty response).
    Do NOT use format=json (forces all output into response, slower).
    timeout=600 handles complex prompts that need >120s of reasoning.
    """
    prompt = JUDGE_USER_TMPL.format(gt=gt, hyp=hyp)
    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "system": JUDGE_SYSTEM,
        "stream": False,
        "options": {"temperature": 0},
    }
    for attempt in range(retries):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=600)
            resp.raise_for_status()
            raw = resp.json()
            # Primary: response field; fallback: thinking (rare edge case)
            text = raw.get("response", "") or raw.get("thinking", "") or ""
            # Extract JSON from response (model may add surrounding text)
            start = text.find("{")
            end   = text.rfind("}") + 1
            if start == -1 or end == 0:
                raise ValueError(f"No JSON in response: {text[:200]}")
            scores = json.loads(text[start:end])
            # Validate all 8 keys are present and in [1,10]
            for dim in DIMS:
                v = scores.get(dim)
                if v is None:
                    raise ValueError(f"Missing dim: {dim}")
                scores[dim] = max(1, min(10, int(v)))
            return scores
        except Exception as e:
            log.warning("  Judge attempt %d/%d failed: %s", attempt + 1, retries, e)
            time.sleep(2 * (attempt + 1))
    return None


def score_condition_judge(cond, outputs, ground_truth):
    scores_file = JUDGE_SCORES[cond]
    plog(f"[STEP 7] Scoring condition {cond} with LLM judge...")

    existing = {}
    if scores_file.exists():
        with open(scores_file) as f:
            existing = json.load(f)
        plog(f"  Resuming judge: {len(existing)} already scored.")

    all_scores = dict(existing)
    vids_to_score = [
        vid for vid in outputs
        if vid not in existing
        and vid in ground_truth
        and outputs[vid]
        and not outputs[vid].startswith("[")
    ]

    skipped = sum(
        1 for vid in outputs
        if outputs[vid].startswith("[") or vid not in ground_truth
    )
    if skipped:
        plog(f"  Skipping {skipped} error/missing outputs.")

    for i, vid in enumerate(vids_to_score, 1):
        scores = call_judge(ground_truth[vid], outputs[vid])
        if scores is None:
            log.error("  Judge failed for %s — skipping.", vid)
            continue
        # Add MCF and NAF
        scores["MCF_Score"] = round(
            (scores["spatial_orientation"] + scores["social_interaction"] +
             scores["action_events"] + scores["ambience"]) / 4, 4)
        scores["NAF_Score"] = round(
            (scores["descriptiveness"] + scores["objectivity"] +
             scores["accuracy"] + scores["clarity"]) / 4, 4)
        all_scores[vid] = scores

        if i % 20 == 0 or i == len(vids_to_score):
            with open(scores_file, "w") as f:
                json.dump(all_scores, f, indent=2)
            plog(f"  Judge [{i}/{len(vids_to_score)}] checkpoint saved.")

    with open(scores_file, "w") as f:
        json.dump(all_scores, f, indent=2)
    plog(f"  Condition {cond} judge scores saved: {scores_file}")
    return all_scores


def compute_averages(scores_dict):
    if not scores_dict:
        return {}
    avgs = {}
    for dim in DIMS + ["MCF_Score", "NAF_Score"]:
        vals = [v[dim] for v in scores_dict.values() if dim in v]
        avgs[dim] = round(sum(vals) / len(vals), 4) if vals else None
    avgs["n"] = len(scores_dict)
    return avgs


def sanity_check(avgs, cond):
    plog(f"[SANITY] Checking condition {cond} averages against paper ranges...")
    issues = []
    for dim, (lo, hi) in SANITY_RANGES.items():
        v = avgs.get(dim)
        if v is None:
            issues.append(f"  MISSING: {dim}")
        elif not (lo <= v <= hi):
            issues.append(f"  OUT OF RANGE: {dim} = {v:.2f} (expected {lo}-{hi})")
        else:
            plog(f"  OK: {dim} = {v:.2f} [{lo}-{hi}]")

    if issues:
        msg = f"[{ts()}] SANITY FAIL for condition {cond}:\n" + "\n".join(issues) + "\n"
        plog(f"SANITY CHECK FAILED:\n" + "\n".join(issues))
        with open(SANITY_LOG, "a") as f:
            f.write(msg)
        return False
    else:
        plog("  All dimensions within expected ranges.")
        with open(SANITY_LOG, "a") as f:
            f.write(f"[{ts()}] Condition {cond} PASSED sanity check.\n")
        return True


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8: NLP Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_nlp_metrics(test_set, ground_truth, all_outputs):
    plog("[STEP 8] Computing NLP metrics (pycocoevalcap)...")

    if NLP_METRICS_FILE.exists():
        plog(f"  NLP metrics already exist: {NLP_METRICS_FILE}")
        with open(NLP_METRICS_FILE) as f:
            return json.load(f)

    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.spice.spice import Spice
    except ImportError as e:
        plog(f"  pycocoevalcap not available: {e} — skipping NLP metrics.")
        return {}

    results = {}
    for cond, out_file in COND_OUTPUTS.items():
        if not out_file.exists():
            plog(f"  Skipping condition {cond} NLP metrics — outputs not found.")
            continue
        with open(out_file) as f:
            outputs = json.load(f)

        gts  = {}
        hyps = {}
        for entry in test_set:
            vid = entry["video_id"]
            gt  = ground_truth.get(vid, "")
            hyp = outputs.get(vid, "")
            if not gt or not hyp or hyp.startswith("["):
                continue
            gts[vid]  = [gt]
            hyps[vid] = [hyp]

        if not gts:
            plog(f"  No valid pairs for condition {cond}.")
            continue

        plog(f"  Scoring {len(gts)} pairs for condition {cond}...")
        cond_scores = {}

        try:
            scorer = Bleu(4)
            score, _ = scorer.compute_score(gts, hyps)
            cond_scores["BLEU-1"] = round(score[0], 4)
            cond_scores["BLEU-4"] = round(score[3], 4)
        except Exception as e:
            log.warning("  BLEU failed: %s", e)

        try:
            scorer = Meteor()
            score, _ = scorer.compute_score(gts, hyps)
            cond_scores["METEOR"] = round(score, 4)
        except Exception as e:
            log.warning("  METEOR failed: %s", e)

        try:
            scorer = Rouge()
            score, _ = scorer.compute_score(gts, hyps)
            cond_scores["ROUGE-L"] = round(score, 4)
        except Exception as e:
            log.warning("  ROUGE-L failed: %s", e)

        try:
            scorer = Cider()
            score, _ = scorer.compute_score(gts, hyps)
            cond_scores["CIDEr"] = round(score, 4)
        except Exception as e:
            log.warning("  CIDEr failed: %s", e)

        try:
            scorer = Spice()
            score, _ = scorer.compute_score(gts, hyps)
            cond_scores["SPICE"] = round(score, 4)
        except Exception as e:
            log.warning("  SPICE failed: %s", e)

        cond_scores["n"] = len(gts)
        results[cond] = cond_scores
        plog(f"  Condition {cond} NLP: {cond_scores}")

    with open(NLP_METRICS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    plog(f"  NLP metrics saved: {NLP_METRICS_FILE}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9: Final results table
# ─────────────────────────────────────────────────────────────────────────────

def save_final_results(judge_avgs, nlp_metrics):
    plog("[STEP 9] Saving final results table...")
    table = {}
    for cond in ("A", "B", "C"):
        table[cond] = {
            "judge": judge_avgs.get(cond, {}),
            "nlp":   nlp_metrics.get(cond, {}),
        }
    with open(FINAL_TABLE_FILE, "w") as f:
        json.dump(table, f, indent=2)
    plog(f"  Final table saved: {FINAL_TABLE_FILE}")

    # Human-readable summary
    lines = [
        "=" * 70,
        "  BLV ABLATION EVALUATION — FINAL RESULTS",
        f"  Generated: {ts()}",
        "  A: SmolVLM2-500M (no fine-tuning)",
        "  B: SmolVLM2-500M + SFT",
        "  C: SmolVLM2-500M + SFT + DPO",
        "=" * 70,
        "",
        "── LLM JUDGE SCORES (gpt-oss:20b, 1-10 scale) ──",
        f"  {'Dimension':<22} {'A':>8} {'B':>8} {'C':>8}  {'Δ B→C':>8}",
        "  " + "-" * 56,
    ]
    for dim in DIMS:
        va = judge_avgs.get("A", {}).get(dim)
        vb = judge_avgs.get("B", {}).get(dim)
        vc = judge_avgs.get("C", {}).get(dim)
        delta = f"{vc - vb:+.2f}" if vb is not None and vc is not None else "  N/A"
        lines.append(
            f"  {dim:<22} "
            f"{str(round(va,2)) if va else 'N/A':>8} "
            f"{str(round(vb,2)) if vb else 'N/A':>8} "
            f"{str(round(vc,2)) if vc else 'N/A':>8}  {delta:>8}"
        )

    for label, key in [("MCF_Score (avg)", "MCF_Score"), ("NAF_Score (avg)", "NAF_Score")]:
        va = judge_avgs.get("A", {}).get(key)
        vb = judge_avgs.get("B", {}).get(key)
        vc = judge_avgs.get("C", {}).get(key)
        delta = f"{vc - vb:+.2f}" if vb is not None and vc is not None else "  N/A"
        lines.append(
            f"  {label:<22} "
            f"{str(round(va,2)) if va else 'N/A':>8} "
            f"{str(round(vb,2)) if vb else 'N/A':>8} "
            f"{str(round(vc,2)) if vc else 'N/A':>8}  {delta:>8}"
        )

    lines += ["", "── NLP METRICS ──",
              f"  {'Metric':<12} {'A':>8} {'B':>8} {'C':>8}", "  " + "-" * 40]
    for metric in ["BLEU-1", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr", "SPICE"]:
        va = nlp_metrics.get("A", {}).get(metric)
        vb = nlp_metrics.get("B", {}).get(metric)
        vc = nlp_metrics.get("C", {}).get(metric)
        lines.append(
            f"  {metric:<12} "
            f"{str(round(va,4)) if va else 'N/A':>8} "
            f"{str(round(vb,4)) if vb else 'N/A':>8} "
            f"{str(round(vc,4)) if vc else 'N/A':>8}"
        )

    lines += ["", "=" * 70]
    summary = "\n".join(lines)
    with open(SUMMARY_FILE, "w") as f:
        f.write(summary)
    print(summary)
    plog(f"  Summary saved: {SUMMARY_FILE}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    plog("=" * 60)
    plog(f"[START] Full eval run started at {ts()}")
    plog(f"  CUDA_VISIBLE_DEVICES=4  |  GPU for inference: {DEVICE}")
    plog(f"  Ollama judge: {OLLAMA_MODEL} @ {OLLAMA_URL}")
    plog("=" * 60)

    # ── Install pycocoevalcap if needed ───────────────────────────────────────
    try:
        import pycocoevalcap  # noqa
        plog("[SETUP] pycocoevalcap already installed.")
    except ImportError:
        plog("[SETUP] Installing pycocoevalcap...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "pycocoevalcap", "--break-system-packages", "-q"])
        plog("[SETUP] pycocoevalcap installed.")

    # ── Step 2: Test set ──────────────────────────────────────────────────────
    test_set = build_test_set()
    test_set = hydrate_kf(test_set)

    # ── Step 3: Ground truth ──────────────────────────────────────────────────
    ground_truth = build_ground_truth(test_set)

    # ── Steps 4-6: Inference for A, B, C ─────────────────────────────────────
    cond_labels = {"A": "STEP 4", "B": "STEP 5", "C": "STEP 6"}
    all_outputs = {}
    for cond in ("A", "B", "C"):
        all_outputs[cond] = generate_condition(cond, test_set, cond_labels[cond])

    # ── Step 7: LLM Judge scoring ─────────────────────────────────────────────
    judge_avgs    = {}
    sanity_passed = True

    for cond in ("A", "B", "C"):
        outputs = all_outputs[cond]
        if not outputs:
            plog(f"  No outputs for condition {cond} — skipping judge.")
            continue
        scores = score_condition_judge(cond, outputs, ground_truth)
        avgs   = compute_averages(scores)
        judge_avgs[cond] = avgs
        plog(f"  Condition {cond} averages: {json.dumps(avgs, indent=2)}")

        # Sanity check after Condition A
        if cond == "A":
            passed = sanity_check(avgs, "A")
            if not passed:
                plog("  WARNING: Condition A failed sanity check. "
                     "Logged to sanity_check_log.txt. Continuing to B and C.")
                sanity_passed = False

    # ── Step 8: NLP metrics ───────────────────────────────────────────────────
    nlp_metrics = compute_nlp_metrics(test_set, ground_truth, all_outputs)

    # ── Step 9: Final results ─────────────────────────────────────────────────
    save_final_results(judge_avgs, nlp_metrics)

    plog(f"[DONE] {ts()}")
    with open(PROGRESS_LOG, "a") as f:
        f.write(f"DONE {ts()}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        plog(f"[FATAL] {e}")
        traceback.print_exc()
        sys.exit(1)
