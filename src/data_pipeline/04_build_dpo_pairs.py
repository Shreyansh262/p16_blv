"""
04_build_dpo_pairs.py
Builds chosen/rejected DPO pairs from:
  - Chosen: all_captions.json (Charades 7985 + AVCaps 1661 = 9646 total, teacher-generated, AD-compliant)
  - Rejected: original_caption field (raw dataset labels)
Usage:
    python 04_build_dpo_pairs.py
"""

import json, logging, random
from pathlib import Path
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


def load_config():
    config_path = (Path(__file__).parent.parent.parent
                   / "config" / "paths_config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def is_valid_chosen(text):
    """Check if description meets minimum BLV quality bar."""
    if not text or len(text.strip()) < 30:
        return False
    text_lower = text.lower()
    spatial_words = [
        "left", "right", "ahead", "front", "behind",
        "near", "far", "meter", "center", "forward"
    ]
    has_spatial = any(w in text_lower for w in spatial_words)
    return has_spatial


def is_valid_rejected(text):
    """Reject captions that are too short or are action codes."""
    if not text or len(text.strip()) < 3:
        return False
    # Charades action codes look like 'c001 c045' — reject those
    # but keep them as rejected examples if they're just bad captions
    if text.strip().startswith("c0") and len(text.strip()) < 10:
        return False
    return True


def build_dpo_pairs(captions_path, out_path):
    with open(captions_path) as f:
        captions = json.load(f)

    log.info(f"Loaded {len(captions)} teacher captions")

    pairs   = []
    skipped = 0

    for entry in captions:
        chosen   = entry.get("blv_description", "").strip()
        rejected = entry.get("original_caption", "").strip()

        if not is_valid_chosen(chosen):
            skipped += 1
            continue

        if not is_valid_rejected(rejected):
            # Use a generic fallback rejected caption
            rejected = "A person in a scene."

        # Build prompt (same format used at training time)
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are an assistive technology for blind and "
                    "low-vision users. Describe video scenes following "
                    "professional Audio Description standards."
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": kf
                    }
                    for kf in entry.get("keyframe_paths", [])
                ] + [
                    {
                        "type": "text",
                        "text": "Describe this video for a blind user."
                    }
                ]
            }
        ]

        pairs.append({
            "video_id":       entry["video_id"],
            "dataset":        entry["dataset"],
            "keyframe_paths": entry.get("keyframe_paths", []),
            "prompt":         prompt,
            "chosen":         chosen,
            "rejected":       rejected,
            "rag_context":    entry.get("rag_context", ""),
        })

        # Also add multiturn pairs if available
        for turn in entry.get("multiturn", []):
            if (turn.get("assistant")
                    and is_valid_chosen(turn["assistant"])):
                pairs.append({
                    "video_id":       entry["video_id"] + "_mt",
                    "dataset":        entry["dataset"],
                    "keyframe_paths": entry.get("keyframe_paths", []),
                    "prompt": prompt + [
                        {
                            "role": "assistant",
                            "content": chosen
                        },
                        {
                            "role": "user",
                            "content": turn["user"]
                        }
                    ],
                    "chosen":   turn["assistant"],
                    "rejected": "I cannot see anything clearly.",
                    "rag_context": entry.get("rag_context", ""),
                })

    random.shuffle(pairs)

    # Split 90/10 train/val
    split       = int(len(pairs) * 0.9)
    train_pairs = pairs[:split]
    val_pairs   = pairs[split:]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"train": train_pairs, "val": val_pairs}, f,
                  indent=2)

    log.info(f"DPO pairs built — train: {len(train_pairs)} "
             f"| val: {len(val_pairs)} | skipped: {skipped}")
    log.info(f"Saved to: {out_path}")
    return len(train_pairs), len(val_pairs)


if __name__ == "__main__":
    cfg           = load_config()
    captions_path = Path(cfg["data"]["qwen_captions"])
    out_path      = Path(cfg["data"]["dpo_pairs"])

    if not captions_path.exists():
        log.error(
            f"Captions file not found: {captions_path}\n"
            f"Run 03_generate_teacher_captions.py first."
        )
        raise SystemExit(1)

    build_dpo_pairs(captions_path, out_path)
