"""
01_extract_keyframes.py
Extracts 4 keyframes per video using LUV or DINO method.
Reads directly from local Charades_v1.zip — no streaming needed.
Usage:
    python 01_extract_keyframes.py --method luv --dataset charades --max_videos 10
    python 01_extract_keyframes.py --method luv --dataset avcaps --max_videos 10
"""

import os, json, argparse, logging, io, zipfile
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm
import yaml
import tempfile

def load_config():
    config_path = (Path(__file__).parent.parent.parent
                   / "config" / "paths_config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            Path.home() /
            "p16_blv/logs/training_logs/keyframe_extraction.log"
        )
    ]
)
log = logging.getLogger(__name__)

# ── LUV method ─────────────────────────────────────────────────────────────────
def extract_keyframes_luv(frames, n=4):
    if len(frames) <= n:
        return list(range(len(frames)))
    luv_frames = []
    for f in frames:
        if len(f.shape) == 3 and f.shape[2] == 4:
            f = f[:, :, :3]
        luv = cv2.cvtColor(f, cv2.COLOR_RGB2Luv).astype(np.float32)
        luv_frames.append(luv)
    diffs = []
    for i in range(1, len(luv_frames)):
        diff = np.mean(np.abs(luv_frames[i] - luv_frames[i-1]))
        diffs.append((i, diff))
    diffs.sort(key=lambda x: x[1], reverse=True)
    top_indices = sorted([0] + [d[0] for d in diffs[:n-1]])
    return top_indices

# ── DINO method ────────────────────────────────────────────────────────────────
_dino_model = None
_dino_processor = None

def get_dino():
    global _dino_model, _dino_processor
    if _dino_model is None:
        log.info("Loading DINOv2-base...")
        from transformers import AutoImageProcessor, AutoModel
        _dino_processor = AutoImageProcessor.from_pretrained(
            "facebook/dinov2-base"
        )
        _dino_model = AutoModel.from_pretrained("facebook/dinov2-base")
        _dino_model.eval()
        if torch.cuda.is_available():
            _dino_model = _dino_model.cuda()
    return _dino_model, _dino_processor

def extract_keyframes_dino(frames, n=4):
    if len(frames) <= n:
        return list(range(len(frames)))
    model, processor = get_dino()
    device = next(model.parameters()).device
    embeddings = []
    for f in frames:
        pil = Image.fromarray(f)
        inputs = processor(images=pil, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = model(**inputs).last_hidden_state[:, 0, :]
        embeddings.append(emb.cpu().squeeze(0))
    diffs = []
    for i in range(1, len(embeddings)):
        dist = 1.0 - F.cosine_similarity(
            embeddings[i].unsqueeze(0),
            embeddings[i-1].unsqueeze(0)
        ).item()
        diffs.append((i, dist))
    diffs.sort(key=lambda x: x[1], reverse=True)
    top_indices = sorted([0] + [d[0] for d in diffs[:n-1]])
    return top_indices

# ── Video loading ──────────────────────────────────────────────────────────────
def load_video_from_local_zip(zip_path, video_id):
    """Read one video from local Charades zip — fast, no download."""
    mp4_name = f"{video_id}.mp4"
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Find the file inside zip
            matches = [n for n in zf.namelist()
                       if Path(n).name == mp4_name]
            if not matches:
                return None
            return zf.read(matches[0])
    except Exception as e:
        log.error(f"Zip read error for {video_id}: {e}")
        return None

def video_bytes_to_frames(video_bytes, max_frames=64):
    with tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    cap   = cv2.VideoCapture(tmp_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step  = max(1, total // max_frames)
    frames = []
    idx    = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frames.append(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )
        idx += 1
    cap.release()
    os.unlink(tmp_path)
    return frames

def save_keyframes(frames, indices, out_dir, video_id):
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for rank, idx in enumerate(indices):
        path = (out_dir /
                f"{video_id}_kf{rank:02d}_frame{idx:04d}.jpg")
        img  = Image.fromarray(frames[idx])
        img.save(path, "JPEG", quality=85)
        saved_paths.append(str(path))
    return saved_paths

# ── Charades metadata loader ───────────────────────────────────────────────────
def load_charades_metadata(cfg):
    """
    Stream just metadata from HuggingFace — no video bytes.
    Returns list of dicts with video_id, caption, scene, objects.
    """
    from datasets import load_dataset
    hf_cache = cfg["cache"]["hf_cache"]
    log.info("Loading Charades metadata only (streaming)...")
    ds = load_dataset(
        "HuggingFaceM4/charades",
        streaming=True,
        split="train",
        cache_dir=hf_cache
    )
    metadata = []
    for sample in ds:
        metadata.append({
            "video_id": sample["video_id"],
            "caption":  (sample.get("script") or
                         sample.get("descriptions", [""])[0]),
            "scene":    sample.get("scene", ""),
            "objects":  sample.get("objects", []),
        })
    log.info(f"Loaded metadata for {len(metadata)} videos")
    return metadata

# ── Main loop ──────────────────────────────────────────────────────────────────
def process_charades(method, max_videos, cfg):
    zip_path = Path(cfg["data"]["raw_dir"]) / "Charades_v1.zip"
    out_base = Path(cfg["data"]["keyframes_dir"])

    if not zip_path.exists():
        log.error(
            f"Charades zip not found at {zip_path}\n"
            f"Run: wget -c 'https://ai2-public-datasets.s3-us-west-2"
            f".amazonaws.com/charades/Charades_v1.zip' "
            f"-O {zip_path}"
        )
        raise SystemExit(1)

    log.info(f"Using local zip: {zip_path} "
             f"({zip_path.stat().st_size / 1e9:.1f} GB)")

    # Load metadata from HuggingFace (text only, fast)
    metadata = load_charades_metadata(cfg)
    to_process = metadata[:max_videos]

    manifest  = []
    errors    = []
    processed = 0

    for entry in tqdm(to_process,
                      desc=f"charades/{method}"):
        video_id = entry["video_id"]
        try:
            video_bytes = load_video_from_local_zip(
                zip_path, video_id
            )
            if video_bytes is None:
                log.warning(f"Video not found in zip: {video_id}")
                errors.append(video_id)
                continue

            frames = video_bytes_to_frames(video_bytes)
            if len(frames) < 2:
                continue

            indices = (extract_keyframes_luv(frames)
                       if method == "luv"
                       else extract_keyframes_dino(frames))

            saved = save_keyframes(
                frames, indices,
                out_base / "charades", video_id
            )

            manifest.append({
                "video_id":         video_id,
                "dataset":          "charades",
                "keyframe_paths":   saved,
                "frame_indices":    indices,
                "original_caption": entry["caption"],
                "scene":            entry["scene"],
                "objects":          entry["objects"],
                "method":           method,
            })
            processed += 1

            # Save checkpoint every 200 videos
            if processed % 200 == 0:
                mp = (out_base /
                      f"charades_{method}_manifest.json")
                with open(mp, "w") as f:
                    json.dump(manifest, f, indent=2)
                log.info(f"Checkpoint: {processed} done")

        except Exception as e:
            log.error(f"Error on {video_id}: {e}")
            errors.append(video_id)

    manifest_path = (out_base /
                     f"charades_{method}_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    log.info(f"Done — processed: {processed} "
             f"| errors: {len(errors)}")
    log.info(f"Manifest: {manifest_path}")
    return str(manifest_path)


def process_avcaps(method, max_videos, cfg):
    """AVCaps: downloads zip+captions from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download
    hf_cache = cfg["cache"]["hf_cache"]
    raw_dir  = Path(cfg["data"]["raw_dir"])
    out_base = Path(cfg["data"]["keyframes_dir"])

    # Download captions JSON (small, fast)
    log.info("Downloading AVCaps train_captions.json...")
    captions_path = hf_hub_download(
        "TUT-ARG/AVCaps", "train_captions.json",
        repo_type="dataset", cache_dir=hf_cache
    )
    with open(captions_path) as f:
        captions = json.load(f)
    log.info(f"Loaded captions for {len(captions)} videos")

    # Download videos zip if not present
    zip_path = raw_dir / "avcaps_train_videos.zip"
    if not zip_path.exists():
        log.info("Downloading AVCaps train_videos.zip (~8.5GB)...")
        tmp_path = hf_hub_download(
            "TUT-ARG/AVCaps", "train_videos.zip",
            repo_type="dataset", cache_dir=hf_cache
        )
        import shutil
        shutil.copy(tmp_path, str(zip_path))
        log.info(f"Saved to {zip_path}")
    else:
        log.info(f"Using cached zip: {zip_path}")

    video_ids = list(captions.keys())[:max_videos]
    manifest  = []
    errors    = []
    processed = 0

    for video_id in tqdm(video_ids, desc=f"avcaps/{method}"):
        try:
            video_bytes = load_video_from_local_zip(
                zip_path, video_id
            )
            if video_bytes is None:
                log.warning(f"Video not in zip: {video_id}")
                errors.append(video_id)
                continue

            frames = video_bytes_to_frames(video_bytes)
            if len(frames) < 2:
                continue

            indices = (extract_keyframes_luv(frames)
                       if method == "luv"
                       else extract_keyframes_dino(frames))

            saved = save_keyframes(
                frames, indices,
                out_base / "avcaps", video_id
            )

            caps = captions[video_id]
            caption = (caps.get("GPT_AV_captions",
                       caps.get("visual_captions", [""]))[0])

            manifest.append({
                "video_id":         video_id,
                "dataset":          "avcaps",
                "keyframe_paths":   saved,
                "frame_indices":    indices,
                "original_caption": str(caption),
                "method":           method,
            })
            processed += 1

            if processed % 200 == 0:
                mp = out_base / f"avcaps_{method}_manifest.json"
                with open(mp, "w") as f:
                    json.dump(manifest, f, indent=2)
                log.info(f"Checkpoint: {processed} done")

        except Exception as e:
            log.error(f"Error on {video_id}: {e}")
            errors.append(video_id)

    manifest_path = out_base / f"avcaps_{method}_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    log.info(f"Done — processed: {processed} | errors: {len(errors)}")
    return str(manifest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method",
                        choices=["luv", "dino"], default="luv")
    parser.add_argument("--dataset",
                        choices=["charades", "avcaps"],
                        default="charades")
    parser.add_argument("--max_videos", type=int, default=10)
    args = parser.parse_args()

    cfg = load_config()
    log.info(f"Starting: dataset={args.dataset} "
             f"method={args.method} "
             f"max_videos={args.max_videos}")

    if args.dataset == "charades":
        process_charades(args.method, args.max_videos, cfg)
    else:
        process_avcaps(args.method, args.max_videos, cfg)
