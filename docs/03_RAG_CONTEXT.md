# 03 — RAG Context Retrieval (Updated)

> **Related:** [[00_AI_READ_THIS_FIRST]] | [[04_TEACHER_DATA_GEN]] | [[15_MENTOR_UPDATES]]
> **Owner:** Data Team | **Status:** See [[14_PROGRESS_TRACKER]]
> ⚠️ Updated: Mobile-optimized RAG is now a hard requirement per mentor meeting 1, points 6, 7, 8

---

## What Changed

**Mentor (Meeting 1, points 6, 7, 8):**
- *"Find a way to effectively use RAG in the model for less latency and storage on mobile deployment"*
- *"Main agenda is to reduce the latency and effective use in the deployed edge device"*
- *"Find the optimal way for the vector storage"*

The original plan used standard FAISS with float32 embeddings — fine for a server, impractical on a phone. The updated plan requires RAG that is specifically optimized for:
- Minimal storage (phone has limited disk space)
- Low latency (retrieval must be < 100ms to not block inference)
- Low RAM usage (phone RAM is shared with OS and other apps)

---

## What RAG Does (Reminder)

Before generating a description, the model retrieves relevant scene context from a pre-built knowledge library. For a video of a crowded indoor space, RAG retrieves: "Indoor cafeteria: food counter typically at far end, tables and chairs throughout, trays and queuing area..." This grounds the description in realistic spatial context.

---

## Mobile-Optimized RAG Design

### Problem With Standard FAISS
- float32 embeddings: 384 dimensions × 4 bytes × 1000 entries = ~1.5MB just for embeddings
- Retrieval requires loading full index into RAM — slow on cold start
- General-purpose — not tuned for the small, fixed scene vocabulary you actually need

### Solution 1: Quantized Embeddings + FAISS IVF

```python
import faiss
import numpy as np

# Instead of IndexFlatL2 (brute force), use IVF with PQ compression
# IVF = Inverted File Index (clusters vectors for fast lookup)
# PQ = Product Quantization (compresses vector size ~8-32x)

dimension = 384  # all-MiniLM-L6-v2 output size
n_clusters = 8   # small because your library has only 20-30 scenes

# Product Quantization: compresses 384 float32 dims → 8 bytes per vector
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFPQ(
    quantizer, 
    dimension,
    n_clusters,   # number of clusters
    8,            # number of sub-quantizers (bytes per vector)
    8             # bits per sub-quantizer
)
index.train(embeddings)
index.add(embeddings)

# Result: 30 scene descriptions × 8 bytes = ~240 bytes total index
# vs original: 30 × 384 × 4 = ~46KB
# 190x smaller, negligible on mobile
```

### Solution 2: Flat Lookup Table (Best for Mobile)

Because your scene vocabulary is small (~20-30 categories, fixed), you don't need a full vector database at all. Use a **flat lookup table**:

```python
# Pre-compute: for each video in your deployment dataset,
# decide its scene category ONCE during training/indexing.
# At inference on mobile: simple dictionary lookup, zero retrieval overhead.

SCENE_LOOKUP = {
    "indoor_kitchen": "Kitchen environment: counter surfaces, appliances...",
    "outdoor_street": "Urban street: road, curb, pedestrians...",
    "indoor_corridor": "Hallway/corridor: linear path, doors on sides...",
    # ... 20-30 categories
}

def get_context_mobile(predicted_scene_class):
    return SCENE_LOOKUP.get(predicted_scene_class, "")
```

**At mobile inference:** Run a tiny scene classifier (MobileNetV3-Small, ~2MB) to predict scene category from first keyframe → look up in dictionary. Total overhead: ~50ms for classification, 0ms for retrieval.

### Solution 3: Baked-In Context (Most Latency-Efficient)

For the most latency-critical deployment, pre-compute and store the RAG context for every video in the deployment dataset. At inference, retrieve by video ID — no embedding computation at all.

This only works for a pre-indexed video library, not for live camera feeds. For live use, Solution 2 (scene classifier + lookup table) is the right approach.

---

## Optimal Vector Storage (Mentor Point 8)

For the A5000 training pipeline: use standard FAISS IndexFlatL2 (simple, exact, fast on GPU).

For mobile deployment: use one of these in order of preference:

| Option | Storage | Latency | Use When |
|--------|---------|---------|----------|
| Flat lookup table | < 1KB | ~0ms | Fixed scene categories (recommended) |
| FAISS IVF+PQ | ~1KB | ~10ms | Need semantic flexibility |
| usearch (mobile-optimized FAISS alternative) | ~5KB | ~20ms | Need approximate NN search |
| Standard FAISS IndexFlatL2 | ~46KB | ~100ms | Server only — too slow for mobile |

**Recommendation:** Flat lookup table for mobile deployment. It perfectly matches your use case (fixed 20-30 scene categories, pre-defined descriptions).

---

## Scene Library (Unchanged, ~20-30 Categories)

Build entries for: outdoor urban street, crosswalk, park, parking lot, bus stop, stairs/escalator, indoor corridor, elevator lobby, automatic door, ramp, kitchen, living room, bedroom, bathroom, office, cafeteria, hospital, shop, library, classroom.

For each: write a 2-3 sentence BLV-relevant spatial description following AD guidelines (see [[04_TEACHER_DATA_GEN]]).

---

## Checklist

- [ ] Write 20-30 scene descriptions (AD guidelines compliant)
- [ ] Build standard FAISS index for training pipeline
- [ ] Build flat lookup table for mobile deployment
- [ ] Train MobileNetV3-Small scene classifier on scene categories (~2MB model)
- [ ] Test scene classifier accuracy on 50 sample videos
- [ ] Benchmark lookup latency on phone (target: < 50ms total)
- [ ] Integrate into Qwen prompt pipeline ([[04_TEACHER_DATA_GEN]])
- [ ] Integrate lightweight version into mobile inference ([[09_DEPLOYMENT]])
- [ ] Measure: does RAG improve BLV scores? (ablation in [[07_ABLATION]])

---

*Next: [[04_TEACHER_DATA_GEN]] | Context: [[15_MENTOR_UPDATES]]*
