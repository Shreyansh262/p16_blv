#!/usr/bin/env python3
"""RTC latency benchmark on Charades keyframes."""
import time, json, os, torch, numpy as np
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
CKPT       = "/usershome/cs671_user2/p16_blv/models/student/final_merged"
FRAMES_DIR = Path("/usershome/cs671_user2/p16_blv/data/keyframes/charades")
OUT        = Path("rtc_latency_results.json")
PROMPT     = ("Describe what is happening in this scene for a blind user. "
              "Focus on navigation hazards and spatial layout.")

if not Path(CKPT).exists():
    raise FileNotFoundError(f"Checkpoint not found: {CKPT}")
if not FRAMES_DIR.exists():
    raise FileNotFoundError(f"Frames dir not found: {FRAMES_DIR}")

processor = AutoProcessor.from_pretrained(CKPT)
model = AutoModelForImageTextToText.from_pretrained(
    CKPT, torch_dtype=torch.bfloat16, device_map={"":0}
).eval()

exts   = {".jpg", ".jpeg", ".png"}
frames = sorted([p for p in FRAMES_DIR.rglob("*") if p.suffix.lower() in exts])
print(f"Found {len(frames)} test frames", flush=True)
if len(frames) == 0:
    raise RuntimeError("No frames found. Check FRAMES_DIR path.")

print("Warming up (3 frames)...", flush=True)
for p in frames[:min(3, len(frames))]:
    img  = Image.open(p).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    ti   = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp  = processor(text=ti, images=img, return_tensors="pt").to("cuda")
    with torch.no_grad(): model.generate(**inp, max_new_tokens=128, do_sample=False)
torch.cuda.synchronize()
print("Warm-up done. Starting timed runs...", flush=True)

latencies, per_frame = [], []
for i, fp in enumerate(frames):
    img  = Image.open(fp).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    ti   = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp  = processor(text=ti, images=img, return_tensors="pt").to("cuda")
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.no_grad(): ids = model.generate(**inp, max_new_tokens=128, do_sample=False)
    torch.cuda.synchronize(); t1 = time.perf_counter()
    lat_ms = (t1 - t0) * 1000
    pred   = processor.decode(ids[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    latencies.append(lat_ms)
    per_frame.append({"frame": str(fp), "latency_ms": round(lat_ms, 2), "pred": pred})
    if i % 50 == 0: print(f"[{i}/{len(frames)}] {lat_ms:.1f} ms  |  {pred[:60]}", flush=True)

la = np.array(latencies)
summary = {"checkpoint": CKPT, "n_frames": len(frames),
           "mean_ms": round(float(la.mean()), 2),
           "p95_ms":  round(float(np.percentile(la, 95)), 2),
           "fps":     round(1000.0 / float(la.mean()), 3),
           "per_frame": per_frame}
OUT.write_text(json.dumps(summary, indent=2))
print(f"\n=== RTC Latency Results ===")
print(f"Frames : {len(frames)}")
print(f"Mean   : {summary['mean_ms']} ms")
print(f"P95    : {summary['p95_ms']} ms")
print(f"FPS    : {summary['fps']}")
