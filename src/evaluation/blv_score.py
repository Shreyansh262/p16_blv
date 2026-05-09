#!/usr/bin/env python3
"""
blv_score.py — Ablation evaluation: Condition A (baseline) vs B (SFT) vs C (SFT+DPO)

Implements all four evaluation approaches from docs/08_EVALUATION.md:
  1. Multi-Context BLV Score  (spatial, social, action events, ambience)
  2. Navigational Assistance Score (obstacles, directions, hazards, distances, step-changes)
  3. BLEU-4 (sacrebleu, vs blv_description reference)
  4. ROUGE-L (rouge_score, vs blv_description reference)

Conditions:
  A — SmolVLM2-500M baseline (no fine-tuning)
  B — SmolVLM2-500M + SFT Stage 2 (LoRA)
  C — SmolVLM2-500M + SFT Stage 2 merged + DPO LoRA

Test set: held-out from all_captions.json, excludes every video_id used in
          dpo_pairs.json train+val splits.  Saved to logs/eval_results/test_set.json
          so all conditions run on the identical set.

Usage:
    python src/evaluation/blv_score.py --condition all --n 100 --gpu 6
    python src/evaluation/blv_score.py --condition A --gpu 6
    python src/evaluation/blv_score.py --condition C --no-generate   # score only
    python src/evaluation/blv_score.py --no-generate                 # re-score all saved outputs
"""

import os, sys, json, random, argparse, logging
from pathlib import Path

# ── Arg parse (CUDA must be set before torch) ─────────────────────────────────
parser = argparse.ArgumentParser(description="BLV Ablation Evaluation")
parser.add_argument("--condition",   choices=["A", "B", "C", "all"], default="all",
                    help="Which condition(s) to run (default: all)")
parser.add_argument("--n",           type=int, default=100,
                    help="Test-set size (default 100; -1 = all eligible)")
parser.add_argument("--gpu",         type=int, default=6,
                    help="GPU index (default 6)")
parser.add_argument("--seed",        type=int, default=42)
parser.add_argument("--results-dir", type=str,
                    default="/usershome/cs671_user2/p16_blv/logs/eval_results")
parser.add_argument("--no-generate", action="store_true",
                    help="Skip inference; score already-saved outputs only")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ["HF_HOME"] = "/usershome/cs671_user2/p16_blv/models/.hf_cache"

import torch
from PIL import Image
from transformers import AutoProcessor, SmolVLMForConditionalGeneration
from peft import PeftModel
import sacrebleu as _sacrebleu
from rouge_score import rouge_scorer as _rouge_module

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
HF_CACHE      = "/usershome/cs671_user2/p16_blv/models/.hf_cache"
SFT_PATH      = "/usershome/cs671_user2/p16_blv/models/student/sft_stage2_checkpoint"
DPO_PATH      = "/usershome/cs671_user2/p16_blv/models/student/dpo_checkpoint"
CAPTIONS_FILE = "/usershome/cs671_user2/p16_blv/data/generated/all_captions.json"
DPO_PAIRS_FILE= "/usershome/cs671_user2/p16_blv/data/generated/dpo_pairs.json"

DEVICE         = "cuda:0"
MAX_SIDE       = 364
MAX_NEW_TOKENS = 256
PROMPT         = "Describe this video for a blind user."

RESULTS_DIR   = Path(args.results_dir)
TEST_SET_FILE = RESULTS_DIR / "test_set.json"

# ── BLV keyword banks ─────────────────────────────────────────────────────────

_SPATIAL = [
    "left", "right", "ahead", "behind", "above", "below", "near", "far",
    "meters", "feet", "distance", "position", "next to", "beside",
    "in front", "to your", "on your", "at approximately", "straight",
    "forward", "close", "nearby", "across", "steps away", "directly",
    "on the left", "on the right",
]
_SOCIAL = [
    "person", "people", "someone", "man", "woman", "child", "individual",
    "walking toward", "approaching", "facing", "seated", "standing near",
    "figure", "human", "group", "crowd", "pedestrian", "user", "you",
]
_ACTION = [
    "picks up", "walking", "sitting", "standing", "opens", "closes",
    "carries", "reaches", "bends", "turns", "moves", "steps", "places",
    "puts", "takes", "holds", "sets", "lifts", "pushes", "pulls",
    "reaches for", "grabs", "uses", "cooking", "eating", "running",
]
_AMBIENCE = [
    "indoor", "outdoor", "kitchen", "bedroom", "living room", "street",
    "office", "hallway", "bathroom", "dining", "bright", "dim", "dark",
    "light", "morning", "evening", "room", "space", "area", "environment",
    "floor", "ceiling", "wall", "natural light", "artificial", "carpeted",
    "tiled", "wooden", "outside", "inside",
]
_OBSTACLE = [
    "obstacle", "furniture", "table", "chair", "door", "wall", "step",
    "curb", "barrier", "counter", "desk", "sofa", "couch", "bed",
    "cabinet", "shelf", "stair", "column", "pillar", "railing", "fence",
    "appliance", "box", "bin",
]
_STEP = [
    "step up", "step down", "stairs", "steps", "curb", "ramp", "uneven",
    "elevation", "threshold", "ledge", "staircase", "raised", "lowered",
    "slope", "gradient",
]
_DIRECTION = [
    "left", "right", "straight", "forward", "behind", "turn", "head toward",
    "proceed", "continue", "go to", "move toward", "face", "clockwise",
    "counterclockwise", "bear left", "bear right",
]
_MOVING_HAZARD = [
    "moving", "approaching", "vehicle", "car", "bicycle", "bike",
    "coming toward", "walking toward", "running toward", "rushing",
    "swinging", "opening door", "closing door", "falling", "rolling",
]
_DISTANCE = [
    "meters", "feet", "close", "nearby", "far", "approximately", "about",
    "distance", "roughly", "within", "less than", "more than",
    "steps away", "meters away", "a few",
]


def _has(text: str, kws: list) -> float:
    t = text.lower()
    return 1.0 if any(kw in t for kw in kws) else 0.0


def score_blv(text: str) -> dict:
    """Approach 1: Multi-Context BLV Score."""
    dims = {
        "spatial":  _has(text, _SPATIAL),
        "social":   _has(text, _SOCIAL),
        "action":   _has(text, _ACTION),
        "ambience": _has(text, _AMBIENCE),
    }
    dims["mean"] = sum(dims.values()) / 4.0
    return dims


def score_nav(text: str) -> dict:
    """Approach 2: Navigational Assistance Score."""
    dims = {
        "obstacles":      _has(text, _OBSTACLE),
        "step_changes":   _has(text, _STEP),
        "directions":     _has(text, _DIRECTION),
        "moving_hazards": _has(text, _MOVING_HAZARD),
        "distances":      _has(text, _DISTANCE),
    }
    dims["mean"] = sum(dims.values()) / 5.0
    return dims


# ── Inference ─────────────────────────────────────────────────────────────────

def load_images(paths: list) -> list:
    imgs = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            continue
        try:
            img = Image.open(path).convert("RGB")
            if max(img.size) > MAX_SIDE:
                img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
            imgs.append(img)
        except Exception as e:
            log.warning("Cannot load %s: %s", p, e)
    return imgs


def run_inference(model, processor, images: list) -> str:
    if not images:
        return "[ERROR: no keyframes]"
    messages = [{
        "role": "user",
        "content": (
            [{"type": "image", "image": img} for img in images]
            + [{"type": "text",  "text":  PROMPT}]
        ),
    }]
    text   = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors="pt").to(DEVICE)
    if "pixel_values" in inputs and inputs["pixel_values"] is not None:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    n_in      = inputs["input_ids"].shape[1]
    generated = out_ids[0][n_in:]
    if len(generated) <= 2:
        return "[COLLAPSED]"
    return processor.decode(generated, skip_special_tokens=True).strip()


# ── Model loading ─────────────────────────────────────────────────────────────

def load_condition(cond: str):
    """
    Load (model, processor) for a given condition.

    A: BASE only
    B: BASE + SFT LoRA (sft_stage2_checkpoint)
    C: BASE + SFT LoRA merged + DPO LoRA (dpo_checkpoint)
       — mirrors the exact lineage used during DPO training
    """
    log.info("Loading condition %s...", cond)
    base = SmolVLMForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map={"": 0},
        cache_dir=HF_CACHE,
    )

    if cond == "A":
        processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, cache_dir=HF_CACHE)
        model     = base

    elif cond == "B":
        model     = PeftModel.from_pretrained(base, SFT_PATH, torch_dtype=torch.float16)
        processor = AutoProcessor.from_pretrained(SFT_PATH, cache_dir=HF_CACHE)

    elif cond == "C":
        # Step 1: apply SFT LoRA and merge (same as 07_dpo_train.py)
        log.info("  Merging SFT LoRA into base weights...")
        sft    = PeftModel.from_pretrained(base, SFT_PATH, torch_dtype=torch.float16)
        merged = sft.merge_and_unload()
        if hasattr(merged, "peft_config"):
            delattr(merged, "peft_config")
        # Step 2: apply DPO LoRA on top of merged weights
        log.info("  Applying DPO LoRA...")
        model     = PeftModel.from_pretrained(merged, DPO_PATH, torch_dtype=torch.float16)
        processor = AutoProcessor.from_pretrained(DPO_PATH, cache_dir=HF_CACHE)

    model.eval()
    log.info("  Condition %s ready.", cond)
    return model, processor


def unload(model):
    del model
    torch.cuda.empty_cache()


# ── Test-set construction ─────────────────────────────────────────────────────

def build_test_set(n: int, seed: int) -> list:
    """
    Sample a held-out test set from all_captions.json, excluding every
    video_id that appeared in dpo_pairs.json (train + val).

    Target ratio: ~5:2 Charades:AVCaps (matches dataset proportions).
    """
    log.info("Building held-out test set...")
    with open(CAPTIONS_FILE)  as f: all_caps = json.load(f)
    with open(DPO_PAIRS_FILE) as f: dpo_data = json.load(f)

    used = {item["video_id"]
            for split in ("train", "val")
            for item in dpo_data.get(split, [])}
    log.info("  DPO-used IDs excluded: %d", len(used))

    charades, avcaps = [], []
    for entry in all_caps:
        if entry["video_id"] in used:
            continue
        kf_ok = [p for p in entry.get("keyframe_paths", []) if Path(p).exists()]
        if not kf_ok or not entry.get("blv_description", "").strip():
            continue
        row = dict(entry)
        row["_existing_kf"] = kf_ok
        (charades if row["dataset"] == "charades" else avcaps).append(row)

    log.info("  Eligible held-out: %d Charades, %d AVCaps", len(charades), len(avcaps))

    rng = random.Random(seed)
    if n == -1:
        selected = charades + avcaps
    else:
        n_ch = min(round(n * 5 / 7), len(charades))
        n_av = min(n - n_ch, len(avcaps))
        selected = rng.sample(charades, n_ch) + rng.sample(avcaps, n_av)
    rng.shuffle(selected)

    n_ch = sum(1 for e in selected if e["dataset"] == "charades")
    n_av = len(selected) - n_ch
    log.info("  Test set: %d total (%d Charades, %d AVCaps)", len(selected), n_ch, n_av)
    return selected


# ── Generation ────────────────────────────────────────────────────────────────

def generate_outputs(cond: str, test_set: list) -> dict:
    """
    Run inference for one condition on the test set.
    Saves to logs/eval_results/outputs_{cond}.json after every video
    (crash-safe; resumes from last saved point).
    """
    out_file = RESULTS_DIR / ("outputs_%s.json" % cond)

    existing = {}
    if out_file.exists():
        with open(out_file) as f:
            existing = json.load(f)
        log.info("  Resuming condition %s: %d already done.", cond, len(existing))

    todo = [e for e in test_set if e["video_id"] not in existing]
    if not todo:
        log.info("  All outputs present for condition %s.", cond)
        return existing

    model, processor = load_condition(cond)
    outputs = dict(existing)

    for i, entry in enumerate(todo, 1):
        vid    = entry["video_id"]
        images = load_images(entry["_existing_kf"])
        log.info("  [%d/%d] %s -> %s (%d frames)", i, len(todo), cond, vid, len(images))
        outputs[vid] = run_inference(model, processor, images)
        with open(out_file, "w") as f:
            json.dump(outputs, f, indent=2)

    unload(model)
    log.info("  Condition %s: all outputs saved to %s", cond, out_file)
    return outputs


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_condition(cond: str, outputs: dict, test_set: list) -> dict:
    """Compute all four metric families for one condition."""
    refs_by_id = {e["video_id"]: e["blv_description"] for e in test_set}
    rouge      = _rouge_module.RougeScorer(["rougeL"], use_stemmer=True)

    blv_rows, nav_rows = [], []
    hyps, refs_list    = [], []
    rouge_vals         = []
    skipped            = 0

    for entry in test_set:
        vid = entry["video_id"]
        hyp = outputs.get(vid, "")
        ref = refs_by_id.get(vid, "")
        if not hyp or hyp.startswith("["):
            skipped += 1
            continue
        blv_rows.append(score_blv(hyp))
        nav_rows.append(score_nav(hyp))
        hyps.append(hyp)
        refs_list.append(ref)
        rouge_vals.append(rouge.score(ref, hyp)["rougeL"].fmeasure)

    n = len(blv_rows)
    if skipped:
        log.warning("  Condition %s: %d outputs skipped (collapsed/error)", cond, skipped)
    if n == 0:
        log.error("  Condition %s: no valid outputs to score.", cond)
        return {"n": 0}

    bleu4 = _sacrebleu.corpus_bleu(hyps, [refs_list]).score

    def avg(rows, key):
        return sum(r[key] for r in rows) / len(rows)

    return {
        "n":                  n,
        "blv_spatial":        avg(blv_rows, "spatial"),
        "blv_social":         avg(blv_rows, "social"),
        "blv_action":         avg(blv_rows, "action"),
        "blv_ambience":       avg(blv_rows, "ambience"),
        "blv_mean":           avg(blv_rows, "mean"),
        "nav_obstacles":      avg(nav_rows, "obstacles"),
        "nav_step_changes":   avg(nav_rows, "step_changes"),
        "nav_directions":     avg(nav_rows, "directions"),
        "nav_moving_hazards": avg(nav_rows, "moving_hazards"),
        "nav_distances":      avg(nav_rows, "distances"),
        "nav_mean":           avg(nav_rows, "mean"),
        "bleu4":              bleu4,
        "rouge_l":            sum(rouge_vals) / len(rouge_vals),
    }


# ── Results table ─────────────────────────────────────────────────────────────

def print_results(all_scores: dict):
    order   = ["A", "B", "C"]
    present = [c for c in order if c in all_scores and all_scores[c].get("n", 0) > 0]
    if not present:
        print("No scores to display.")
        return

    CW = 16
    labels = {
        "A": "Baseline (A)",
        "B": "SFT (B)",
        "C": "SFT+DPO (C)",
    }

    print()
    print("=" * 80)
    print("  BLV ABLATION EVALUATION RESULTS")
    print("  A: SmolVLM2-500M baseline (no fine-tuning)")
    print("  B: SmolVLM2-500M + SFT Stage 2")
    print("  C: SmolVLM2-500M + SFT Stage 2 + DPO")
    print("=" * 80)

    # Header row
    hdr = "  %-30s" % "Metric"
    for c in present:
        hdr += ("%-*s" % (CW, "%s (n=%d)" % (labels[c], all_scores[c]["n"])))
    if "B" in present and "C" in present:
        hdr += "  Δ B->C"
    elif "A" in present and "B" in present:
        hdr += "  Δ A->B"
    print(hdr)
    print("  " + "-" * 76)

    # Metric rows  (key, display_label, fmt)
    ROWS = [
        (None,               "Multi-Context BLV Score",    None),
        ("blv_spatial",      "  Spatial orientation",       "pct"),
        ("blv_social",       "  Social interaction",        "pct"),
        ("blv_action",       "  Action events",             "pct"),
        ("blv_ambience",     "  Ambience",                  "pct"),
        ("blv_mean",         "  ** MEAN **",                "pct"),
        (None,               "Navigational Assistance",     None),
        ("nav_obstacles",      "  Obstacles",               "pct"),
        ("nav_step_changes",   "  Step changes",            "pct"),
        ("nav_directions",     "  Directions",              "pct"),
        ("nav_moving_hazards", "  Moving hazards",          "pct"),
        ("nav_distances",      "  Distances",               "pct"),
        ("nav_mean",           "  ** MEAN **",              "pct"),
        (None,               "Text Similarity",             None),
        ("bleu4",            "  BLEU-4",                    "bleu"),
        ("rouge_l",          "  ROUGE-L",                   "pct"),
    ]

    for key, label, fmt in ROWS:
        if key is None:
            print()
            print("  %s" % label)
            continue
        row  = "  %-30s" % label
        vals = {}
        for c in present:
            v = all_scores[c].get(key, 0.0)
            vals[c] = v
            if fmt == "pct":
                row += "%-*s" % (CW, "%5.1f%%" % (v * 100))
            else:
                row += "%-*s" % (CW, "%5.2f" % v)
        # Delta column
        if "B" in present and "C" in present:
            d = vals.get("C", 0) - vals.get("B", 0)
        elif "A" in present and "B" in present:
            d = vals.get("B", 0) - vals.get("A", 0)
        else:
            d = None
        if d is not None:
            row += "  %+.1f%%" % (d * 100) if fmt == "pct" else "  %+.2f" % d
        print(row)

    print()
    print("=" * 80)
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load or build fixed test set
    if TEST_SET_FILE.exists():
        with open(TEST_SET_FILE) as f:
            test_set = json.load(f)
        for entry in test_set:
            entry["_existing_kf"] = [
                p for p in entry.get("keyframe_paths", []) if Path(p).exists()
            ]
        log.info("Loaded existing test set: %d videos from %s", len(test_set), TEST_SET_FILE)
    else:
        test_set = build_test_set(args.n, args.seed)
        saveable = [{k: v for k, v in e.items() if k != "_existing_kf"} for e in test_set]
        with open(TEST_SET_FILE, "w") as f:
            json.dump(saveable, f, indent=2)
        log.info("Test set saved: %s", TEST_SET_FILE)

    conditions = ["A", "B", "C"] if args.condition == "all" else [args.condition]
    all_scores = {}

    for cond in conditions:
        out_file = RESULTS_DIR / ("outputs_%s.json" % cond)

        if args.no_generate:
            if not out_file.exists():
                log.error("--no-generate set but %s missing. Skipping.", out_file)
                continue
            with open(out_file) as f:
                outputs = json.load(f)
            log.info("Loaded %d saved outputs for condition %s.", len(outputs), cond)
        else:
            outputs = generate_outputs(cond, test_set)

        log.info("Scoring condition %s...", cond)
        scores = score_condition(cond, outputs, test_set)
        all_scores[cond] = scores

        score_file = RESULTS_DIR / ("scores_%s.json" % cond)
        with open(score_file, "w") as f:
            json.dump(scores, f, indent=2)
        log.info("Scores saved: %s", score_file)

    # Save combined
    with open(RESULTS_DIR / "scores_all.json", "w") as f:
        json.dump(all_scores, f, indent=2)

    print_results(all_scores)


if __name__ == "__main__":
    main()
