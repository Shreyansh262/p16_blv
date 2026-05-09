"""
03_generate_gemma_captions.py
Generates BLV captions using Gemma 4 hosted via ngrok API.
Runs on college server — keyframes read locally, sent as base64 to remote API.
Usage:
    python 03_generate_gemma_captions.py --dataset charades --max_videos 10 \
        --api_url https://45bc-5-195-0-145.ngrok-free.app \
        --api_key suryavansidgxapi123123
"""

import os, json, logging, argparse, base64, time, uuid
from pathlib import Path
from PIL import Image
import yaml
import requests
from io import BytesIO
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            Path.home() / "p16_blv/logs/training_logs/gemma_captions.log"
        )
    ]
)
log = logging.getLogger(__name__)

AD_SYSTEM_PROMPT = """You are a professional audio describer for blind and low-vision (BLV) audiences, following ITC and Netflix Audio Description standards.

STRICT RULES — every description MUST follow all of these:
1. FIRST SENTENCE: Name the environment type and any immediate hazard within 2 meters (hot surfaces, steps, wet floor, moving people). If a hazard exists, say CAUTION explicitly.
2. SPATIAL POSITIONS: Every object and person must have a direction (left/right/center/ahead/behind) AND an approximate distance in meters.
3. PEOPLE: Describe by clothing color, position, and direction of movement only.
4. HAZARDS: Explicitly flag steps, ramps, wet floors, hot surfaces, obstacles at foot level, and moving people or vehicles.
5. DISTANCES: Use approximate meters — approximately 2 meters ahead, 1 meter to your left.
6. Present tense, active voice only.
7. Maximum 4 sentences. Every word must serve navigation.
IMPORTANT: Write ONE single unified description across all provided images as if they are frames from one single video. Do NOT write separate descriptions per image. Do NOT reference any previous scenes. Do NOT use headers like Image 1 or Image 2. Output only one paragraph of maximum 4 sentences describing this specific video only."""


def load_config():
    config_path = (Path(__file__).parent.parent.parent
                   / "config" / "paths_config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def image_to_base64(img: Image.Image, max_side: int = 640) -> str:
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def generate_description(api_url: str, api_key: str,
                         keyframe_paths: list, rag_context: str = "",
                         max_retries: int = 3) -> dict:
    images = []
    for p in keyframe_paths:
        if Path(p).exists():
            img = Image.open(p).convert("RGB")
            images.append(image_to_base64(img))

    if not images:
        log.warning("No valid keyframes found — skipping.")
        return {"description": ""}

    context_line = (f"\nSCENE CONTEXT: {rag_context}\n"
                    if rag_context else "")
    user_text = (f"{context_line}Describe this video scene "
                 f"for a blind user.")

    # Force fresh context — no memory of previous videos
    content = []
    for b64 in images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })
    content.append({"type": "text", "text": user_text})

    payload = {
        "model": "google/gemma-4-31B-it",
        "messages": [
            {"role": "system", "content": AD_SYSTEM_PROMPT},
            {"role": "user",   "content": content}
        ],
        "max_tokens": 300,
        "temperature": 0.0,
        "user": str(uuid.uuid4())
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                f"{api_url.rstrip('/')}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=120
            )
            resp.raise_for_status()
            data = resp.json()
            description = (data["choices"][0]["message"]["content"]
                           .strip())
            return {"description": description}

        except requests.exceptions.Timeout:
            log.warning(f"Attempt {attempt}/{max_retries} timed out.")
            time.sleep(2 ** attempt)
        except requests.exceptions.HTTPError as e:
            log.error(f"HTTP {e.response.status_code}: "
                      f"{e.response.text[:200]}")
            if e.response.status_code in (401, 403):
                raise
            time.sleep(2 ** attempt)
        except Exception as e:
            log.error(f"Attempt {attempt} failed: {e}")
            time.sleep(2 ** attempt)

    log.error("All retries exhausted — skipping.")
    return {"description": ""}


def load_rag_db(cfg):
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
    parser.add_argument("--max_videos",  type=int, default=10)
    parser.add_argument("--api_url",     type=str,
                        default="https://45bc-5-195-0-145.ngrok-free.app")
    parser.add_argument("--api_key",     type=str,
                        default="suryavansidgxapi123123")
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    cfg      = load_config()
    out_path = (Path(args.output_file) if args.output_file
                else Path(cfg["data"]["keyframes_dir"]).parent
                     / "generated" / "gemma_captions.json")
    kf_dir   = Path(cfg["data"]["keyframes_dir"])

    manifest_path = kf_dir / f"{args.dataset}_luv_manifest.json"
    if not manifest_path.exists():
        log.error(f"Manifest not found: {manifest_path}")
        raise SystemExit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)
    log.info(f"Loaded {len(manifest)} entries from manifest")

    log.info("Testing API connection...")
    try:
        test = requests.get(
            f"{args.api_url.rstrip('/')}/v1/models",
            headers={"Authorization": f"Bearer {args.api_key}"},
            timeout=15
        )
        log.info(f"API OK: {test.status_code}")
    except Exception as e:
        log.error(f"API unreachable: {e}")
        raise SystemExit(1)

    get_context = load_rag_db(cfg)

    results, done_ids = [], set()
    if out_path.exists():
        with open(out_path) as f:
            results = json.load(f)
        done_ids = {e["video_id"] for e in results}
        log.info(f"Resuming — {len(results)} already done.")

    to_process = [e for e in manifest[:args.max_videos]
                  if e["video_id"] not in done_ids]
    log.info(f"{len(to_process)} videos to process.")

    for entry in tqdm(to_process, desc="Gemma captions"):
        try:
            context = get_context(entry.get("original_caption", ""))
            output  = generate_description(
                args.api_url, args.api_key,
                entry["keyframe_paths"],
                rag_context=context,
            )
            results.append({
                "video_id":         entry["video_id"],
                "dataset":          entry["dataset"],
                "keyframe_paths":   entry["keyframe_paths"],
                "original_caption": entry.get("original_caption", ""),
                "blv_description":  output["description"],
                "rag_context":      context,
                "teacher_model":    "google/gemma-4-31B-it",
            })
        except Exception as e:
            log.error(f"Error on {entry['video_id']}: {e}")
            continue

        if len(results) % 50 == 0:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            log.info(f"Checkpoint: {len(results)} done")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Done. {len(results)} captions → {out_path}")
