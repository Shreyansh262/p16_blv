# 02 — LUV Keyframe Extraction (Updated)

> **Related:** [[00_AI_READ_THIS_FIRST]] | [[04_TEACHER_DATA_GEN]] | [[15_MENTOR_UPDATES]]
> **Owner:** Data Team | **Status:** See [[14_PROGRESS_TRACKER]]
> ⚠️ Updated: DINO features added as alternative/complement to LUV per mentor meeting 2

---

## What This Step Does

Selects 3-4 maximally informative frames from each video instead of feeding all frames to the AI model. Fewer frames = faster inference = lower latency on mobile (mentor's #1 priority).

---

## Two Methods (Original + New)

### Method 1: LUV Color Space Differencing (Original — From Your Paper)

Convert frames to LUV color space (matches human perception of visual difference), compute inter-frame differences, select frames with highest difference scores.

**Why LUV:** Better than RGB differencing because LUV is perceptually uniform — a large LUV distance = a meaningful visual change a human would notice.

```python
import cv2, numpy as np

def luv_keyframes(video_path, n=4):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()
    
    if len(frames) <= n:
        return frames
    
    luv = [cv2.cvtColor(f, cv2.COLOR_BGR2Luv).astype(float) for f in frames]
    diffs = [(i, np.mean(np.abs(luv[i] - luv[i-1]))) for i in range(1, len(luv))]
    diffs.sort(key=lambda x: x[1], reverse=True)
    indices = sorted([0] + [d[0] for d in diffs[:n-1]])
    return [frames[i] for i in indices]
```

---

### Method 2: DINOv2 Feature Distance (New — Mentor Meeting 2, Point 8)

**What DINO is:** DINOv2 (Meta AI) is a self-supervised vision model that produces rich semantic embeddings of images. Two frames with very different DINOv2 embeddings are semantically different — not just pixel-different.

**Why better than LUV for keyframe selection:** LUV measures pixel-level color change. DINO measures semantic content change. Example: a person slowly walking across a frame has small LUV diffs (gradual movement) but large DINO diffs (the spatial relationship changes significantly). For BLV descriptions, semantic change matters more than pixel change.

**Two uses of DINO in this project:**

**Use A — Keyframe selection (drop-in for LUV):**
```python
from transformers import AutoImageProcessor, AutoModel
import torch
from PIL import Image

processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
dino = AutoModel.from_pretrained('facebook/dinov2-base').eval().cuda()

def dino_keyframes(video_path, n=4):
    frames = extract_all_frames(video_path)  # as PIL images
    
    embeddings = []
    for frame in frames:
        inputs = processor(images=frame, return_tensors="pt").to("cuda")
        with torch.no_grad():
            emb = dino(**inputs).last_hidden_state[:, 0, :]  # CLS token
        embeddings.append(emb.cpu())
    
    # Compute cosine distance between consecutive frames
    diffs = []
    for i in range(1, len(embeddings)):
        dist = 1 - F.cosine_similarity(embeddings[i], embeddings[i-1])
        diffs.append((i, dist.item()))
    
    diffs.sort(key=lambda x: x[1], reverse=True)
    indices = sorted([0] + [d[0] for d in diffs[:n-1]])
    return [frames[i] for i in indices]
```

**Use B — Visual backbone for student model:**
If the student model (Qwen2.5-VL-3B or alternative) allows swapping visual encoders, use DINOv2-base as the vision backbone. DINOv2's spatial feature maps are particularly good at encoding "where things are" — directly useful for BLV spatial descriptions.

*Note: This is architecturally more complex. Evaluate Use A first (keyframe selection). Use B is a stretch goal.*

---

## LUV vs DINO Comparison Experiment

Before committing to one method, run this quick experiment on 50 videos:

1. Extract keyframes with LUV method
2. Extract keyframes with DINO method
3. Feed both sets to Teacher model, generate descriptions
4. Score with Multi-Context BLV Framework
5. Pick method with higher average spatial coverage score

Report this comparison in paper — it's a small but novel ablation.

---

## Mobile Latency Consideration

DINOv2-base (86M params) adds ~50-100ms per video for keyframe selection on phone. LUV is pure math — essentially 0ms overhead.

If latency is tight (mentor's #1 priority), use LUV at inference time on mobile and DINO only during training data generation (where speed doesn't matter). Best of both worlds.

---

## Checklist

- [ ] Install OpenCV (LUV method)
- [ ] Install DINOv2: `pip install transformers` (already installed)
- [ ] Test LUV extraction on 10 sample videos
- [ ] Test DINO extraction on same 10 videos
- [ ] Visual comparison: do DINO keyframes look more semantically distinct?
- [ ] Run 50-video comparison experiment — score with BLV framework
- [ ] Log LUV vs DINO decision in [[13_DECISIONS_AND_BLOCKERS]]
- [ ] Run selected method on full Charades + AVCaps
- [ ] Benchmark keyframe selection latency (ms) for paper

---

*Next: [[03_RAG_CONTEXT]] | Context: [[15_MENTOR_UPDATES]]*
