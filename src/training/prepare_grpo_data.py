import json, os, random

CAPTIONS_PATH = "data/generated/all_captions_gemma.json"
KEYFRAMES_DIR = "data/keyframes"
OUTPUT_TRAIN  = "data/generated/grpo_train.json"
OUTPUT_VAL    = "data/generated/grpo_val.json"
VAL_SIZE      = 500
SEED          = 42

FEW_SHOT_PROMPT = (
    "You are an audio description system for blind and low vision users.\n\n"
    "Examples of correct output:\n"
    "Example 1: CAUTION: Kitchen environment, wet floor 1 meter ahead. "
    "Man in blue shirt moves left at 2 meters. Child stands center at 1.5 meters. "
    "Path right is clear.\n"
    "Example 2: Living room. Person in white shirt seated 2 meters ahead center. "
    "Coffee table at 1 meter right — potential trip hazard. Path left is clear.\n\n"
    "Now describe this scene following the exact same format:"
)

def main():
    random.seed(SEED)
    with open(CAPTIONS_PATH) as f:
        captions = json.load(f)
    print(f"Loaded {len(captions)} captions")

    samples = []
    missing = 0
    for entry in captions:
        video_id  = str(entry.get("video_id", entry.get("id", "")))
        reference = entry.get("blv_description", "")
        if not reference.strip():
            continue
        keyframe_path = None
        kfps = entry.get("keyframe_paths", [])
        if kfps and os.path.exists(kfps[0]):
            keyframe_path = kfps[0]
        if keyframe_path is None:
            missing += 1
            continue
        samples.append({
            "sample_id":       video_id,
            "keyframe_path":   keyframe_path,
            "gemma_reference": reference,
            "prompt":          FEW_SHOT_PROMPT,
            "has_caution":     "CAUTION" in reference.upper()
        })

    print(f"Valid samples: {len(samples)} | Missing keyframes: {missing}")
    caution_count = sum(1 for s in samples if s["has_caution"])
    print(f"Samples with CAUTION: {caution_count} ({100*caution_count/len(samples):.1f}%)")

    if len(samples) < 5000:
        print("ERROR: fewer than 5000 valid samples — check keyframe paths and stop")
        return

    random.shuffle(samples)
    val_data   = samples[:VAL_SIZE]
    train_data = samples[VAL_SIZE:]
    os.makedirs("data/generated", exist_ok=True)
    with open(OUTPUT_TRAIN, "w") as f:
        json.dump(train_data, f, indent=2)
    with open(OUTPUT_VAL, "w") as f:
        json.dump(val_data, f, indent=2)
    print(f"Saved {len(train_data)} train -> {OUTPUT_TRAIN}")
    print(f"Saved {len(val_data)} val   -> {OUTPUT_VAL}")

if __name__ == "__main__":
    main()
