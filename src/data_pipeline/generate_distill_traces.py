"""
generate_distill_traces.py
Generates distillation traces (reasoning + description) using Qwen2-VL-7B
for held-out videos not seen in SFT or DPO training.

Usage:
    python generate_distill_traces.py --gpu 6 --max_videos 2000
"""

import os, json, logging, argparse, re
from pathlib import Path
from PIL import Image
import torch
import yaml
from tqdm import tqdm

# ── logging ─────────────────────────────────────────────────────────────────
LOG_PATH = Path.home() / "p16_blv/logs/distill_traces.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_PATH)),
    ]
)
log = logging.getLogger(__name__)

# ── prompt ──────────────────────────────────────────────────────────────────
DISTILLATION_PROMPT = """You are a professional audio describer for blind and low-vision users.

First, reason step by step about this scene:
- What type of environment is this? (bedroom, kitchen, corridor, outdoor, etc.)
- Where are the key objects spatially? (left, right, center, near, far)
- What are the navigation hazards? (obstacles, steps, wet surfaces, moving people)
- What is the person doing and where are they moving?

Then write the final BLV description.

Format your response EXACTLY as:
<reasoning>
[your step by step spatial reasoning here]
</reasoning>
<description>
[final 3-4 sentence BLV description for a blind user]
</description>"""


def load_config():
    config_path = Path(__file__).parent.parent.parent / "config" / "paths_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_teacher_model(cfg):
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

    model_id = cfg["models"]["teacher_id"]
    hf_cache = cfg["cache"]["hf_cache"]
    os.environ["HF_HOME"] = hf_cache

    log.info(f"Loading teacher: {model_id}")
    log.info("Using 4-bit quantization...")

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_cfg,
        device_map={"": 0},
        cache_dir=hf_cache,
    )
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=hf_cache)
    log.info("Teacher model loaded.")
    return model, processor


def parse_output(text):
    """Extract reasoning and description from model output."""
    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
    description_match = re.search(r"<description>(.*?)</description>", text, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    description = description_match.group(1).strip() if description_match else text.strip()
    return reasoning, description


def generate_trace(model, processor, keyframe_paths):
    MAX_SIDE = 640
    images = []
    for p in keyframe_paths:
        if Path(p).exists():
            img = Image.open(p).convert("RGB")
            if max(img.size) > MAX_SIDE:
                img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
            images.append(img)

    if not images:
        return None

    messages = [
        {
            "role": "user",
            "content": (
                [{"type": "image", "image": img} for img in images]
                + [{"type": "text", "text": DISTILLATION_PROMPT}]
            ),
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=images, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=1.0,
        )

    input_len = inputs["input_ids"].shape[1]
    new_tokens = output_ids[0][input_len:]
    raw_output = processor.decode(new_tokens, skip_special_tokens=True).strip()

    reasoning, description = parse_output(raw_output)
    return reasoning, description


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--max_videos", type=int, default=2000)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cfg = load_config()
    out_path = Path(cfg["data"]["generated_dir"]) / "distill_traces.json"
    captions_path = Path(cfg["data"]["qwen_captions"])
    dpo_path = Path(cfg["data"]["dpo_pairs"])

    # ── identify held-out videos ─────────────────────────────────────────────
    log.info("Identifying held-out videos...")
    dpo = json.load(open(dpo_path))
    train_ids = set()
    for split in ["train", "val"]:
        if split in dpo:
            for p in dpo[split]:
                train_ids.add(p.get("video_id", ""))

    captions = json.load(open(captions_path))
    captions_by_id = {c["video_id"]: c for c in captions}
    held_out_ids = set(captions_by_id.keys()) - train_ids
    log.info(
        f"Total: {len(captions_by_id)} | In training: {len(train_ids)} "
        f"| Held out: {len(held_out_ids)}"
    )

    # ── resume support ───────────────────────────────────────────────────────
    results = []
    done_ids = set()
    if out_path.exists():
        with open(out_path) as f:
            results = json.load(f)
        done_ids = {r["video_id"] for r in results}
        log.info(f"Resuming: {len(done_ids)} already processed.")

    to_process = [
        vid for vid in list(held_out_ids)[:args.max_videos]
        if vid not in done_ids
    ]
    log.info(f"Videos to process this run: {len(to_process)}")

    if not to_process:
        log.info("Nothing to do. Exiting.")
        raise SystemExit(0)

    # ── load model ───────────────────────────────────────────────────────────
    model, processor = load_teacher_model(cfg)

    # ── main loop ────────────────────────────────────────────────────────────
    for i, video_id in enumerate(tqdm(to_process, desc="Distill traces")):
        entry = captions_by_id[video_id]
        keyframe_paths = entry.get("keyframe_paths", [])

        existing_kf = [p for p in keyframe_paths if Path(p).exists()]
        if not existing_kf:
            log.warning(f"No keyframes found for {video_id} — skipping.")
            continue

        try:
            result = generate_trace(model, processor, keyframe_paths)
            if result is None:
                log.warning(f"No images loaded for {video_id} — skipping.")
                continue
            reasoning, description = result
            results.append({
                "video_id": video_id,
                "keyframe_paths": keyframe_paths,
                "reasoning": reasoning,
                "description": description,
            })
        except Exception as e:
            log.error(f"Error on {video_id}: {e}")
            continue

        if len(results) % 50 == 0:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            log.info(f"Checkpoint: {len(results)} traces saved.")

    # ── final save ───────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Done. {len(results)} distillation traces saved to {out_path}")
