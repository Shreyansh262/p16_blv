"""
03_generate_teacher_captions.py
Generates BLV-compliant captions using Qwen2-VL-7B teacher model.
Reads keyframe manifest from 01_extract_keyframes.py output.
Usage:
    python 03_generate_teacher_captions.py --dataset charades --max_videos 10
"""

import os, json, logging, argparse
from pathlib import Path
from PIL import Image
import torch
import yaml
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            Path.home() /
            "p16_blv/logs/training_logs/teacher_captions.log"
        )
    ]
)
log = logging.getLogger(__name__)

AD_SYSTEM_PROMPT = """You are a professional audio describer for \
blind and low-vision (BLV) audiences, following ITC and Netflix \
Audio Description standards.

Describe the video content using these professional AD guidelines:
- Use present tense throughout
- Lead with the most safety-critical visual information first
- Specify exact spatial positions: left, right, center, near, far,
  foreground, background with approximate distances where visible
- Describe people by observable features only: clothing color,
  position, direction of movement
- Describe the environment type in the first sentence
- Mention surface conditions, obstacles, level changes
- Use active voice: "A person opens the door" not "The door is opened"
- Do not describe what is audible, only what is visible
- Be concise, every word serves the BLV user

Produce a description a BLV person could use to safely navigate
or understand this scene in real time. Maximum 4 sentences."""



def load_config():
    config_path = (Path(__file__).parent.parent.parent
                   / "config" / "paths_config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_teacher_model(cfg):
    from transformers import Qwen2VLForConditionalGeneration
    from transformers import AutoProcessor

    model_id = cfg["models"]["teacher_id"]
    hf_cache = cfg["cache"]["hf_cache"]
    os.environ["HF_HOME"] = hf_cache

    log.info(f"Loading teacher: {model_id}")
    log.info("Using 4-bit quantization to save VRAM...")

    from transformers import BitsAndBytesConfig
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_cfg,
        device_map={"": 0},
        cache_dir=hf_cache,
    )
    processor = AutoProcessor.from_pretrained(
        model_id, cache_dir=hf_cache
    )
    log.info("Teacher model loaded.")
    return model, processor


def generate_description(model, processor, keyframe_paths,
                         rag_context=""):
    images = []
    MAX_SIDE = 640  # cap resolution to avoid CUDA OOM
    for p in keyframe_paths:
        if Path(p).exists():
            img = Image.open(p).convert("RGB")
            if max(img.size) > MAX_SIDE:
                img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
            images.append(img)

    if not images:
        return {"description": ""}

    context_line = (f"\nSCENE CONTEXT: {rag_context}\n"
                    if rag_context else "")
    user_text = (f"{context_line}Describe this video scene "
                 f"for a blind user.")

    messages = [
        {"role": "system", "content": AD_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                [{"type": "image", "image": img} for img in images]
                + [{"type": "text", "text": user_text}]
            )
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=images,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            temperature=1.0,
        )

    input_len   = inputs["input_ids"].shape[1]
    new_tokens  = output_ids[0][input_len:]
    description = processor.decode(
        new_tokens, skip_special_tokens=True
    ).strip()

    return {"description": description}


def load_rag_db(cfg):
    """Load RAG embedder and db path for context retrieval."""
    from sentence_transformers import SentenceTransformer
    import sqlite3, numpy as np

    db_path  = Path(cfg["data"]["rag_db"])
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def get_context(query_text):
        if not db_path.exists():
            return ""
        qemb = embedder.encode(
            [query_text], normalize_embeddings=True
        )[0]
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT category, description, embedding "
            "FROM scene_library"
        ).fetchall()
        conn.close()
        best_score, best_desc = -1, ""
        for _, desc, emb_bytes in rows:
            stored = np.frombuffer(emb_bytes, dtype=np.float32)
            score  = float(np.dot(qemb, stored))
            if score > best_score:
                best_score = score
                best_desc  = desc
        return best_desc

    return get_context


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        choices=["charades", "avcaps"],
                        default="charades")
    parser.add_argument("--max_videos",   type=int, default=10)
    parser.add_argument("--gpu",          type=int, default=0)
    parser.add_argument("--output_file",  type=str, default=None,
                        help="Override output JSON path (default: from config)")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cfg        = load_config()
    out_path   = (Path(args.output_file) if args.output_file
                  else Path(cfg["data"]["qwen_captions"]))
    kf_dir     = Path(cfg["data"]["keyframes_dir"])

    # Load manifest from keyframe extraction step
    manifest_path = (kf_dir /
                     f"{args.dataset}_luv_manifest.json")
    if not manifest_path.exists():
        log.error(
            f"Manifest not found: {manifest_path}\n"
            f"Run 01_extract_keyframes.py first."
        )
        raise SystemExit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    log.info(f"Loaded {len(manifest)} entries from manifest")

    model, processor = load_teacher_model(cfg)
    get_context      = load_rag_db(cfg)

    results   = []
    to_process = manifest[:args.max_videos]

    for entry in tqdm(to_process, desc="Generating captions"):
        try:
            context = get_context(
                entry.get("original_caption", "")
            )
            output  = generate_description(
                model, processor,
                entry["keyframe_paths"],
                rag_context=context,
            )
            results.append({
                "video_id":          entry["video_id"],
                "dataset":           entry["dataset"],
                "keyframe_paths":    entry["keyframe_paths"],
                "original_caption":  entry["original_caption"],
                "blv_description":   output["description"],
                "rag_context":       context,
                "teacher_model":     cfg["models"]["teacher_id"],
            })
        except Exception as e:
            log.error(
                f"Error on {entry['video_id']}: {e}"
            )
            continue

        # Save checkpoint every 50 videos
        if len(results) % 50 == 0:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            log.info(f"Checkpoint saved: {len(results)} done")

    # Final save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"Done. {len(results)} captions saved to {out_path}")
