import time, json, os
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

EVAL_JSON = "data/augmented/balanced_splits/balanced_eval.json"
OUT_DIR   = "results/benchmark_graph"
MODEL_DIR = "models/baselines/moondream2"
N_SAMPLES = 50; MAX_NEW = 120; SEED = 42; GPU = 0

BLV_PROMPT = (
    "Describe this scene for a blind user. "
    "Mention spatial layout, directions (left/right/ahead), distances, obstacles, and any hazards."
)

def load_images(paths):
    imgs = []
    for p in paths:
        if os.path.exists(p):
            imgs.append(Image.open(p).convert("RGB"))
        else:
            alt = os.path.join("data/keyframes/charades", os.path.basename(p))
            if os.path.exists(alt):
                imgs.append(Image.open(alt).convert("RGB"))
    return imgs[:1]  # moondream takes single image

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = json.load(open(EVAL_JSON))
    rng  = np.random.default_rng(SEED)
    idx  = rng.choice(len(data), size=min(N_SAMPLES, len(data)), replace=False)
    samples = [data[i] for i in idx]
    print(f"Loaded {len(samples)} samples.", flush=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
    device = "cuda:0"
    print(f"Loading moondream2 from {MODEL_DIR} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    print("Model loaded.", flush=True)

    latencies, outputs = [], []
    for i, s in enumerate(samples):
        imgs = load_images(s.get("keyframe_paths", []))
        if not imgs:
            print(f"  sample {i}: no images, skip", flush=True); continue
        img = imgs[0]
        try:
            enc = model.encode_image(img)
        except Exception as e:
            print(f"  sample {i} encode error: {e}", flush=True); continue

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            out = model.answer_question(enc, BLV_PROMPT, tokenizer, max_new_tokens=MAX_NEW).strip()
        except Exception as e:
            print(f"  sample {i} gen error: {e}", flush=True); continue
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        latencies.append(ms)
        outputs.append({"video_id": s["video_id"], "output": out, "latency_ms": ms})
        if (i+1) % 10 == 0:
            print(f"  [{i+1}/{len(samples)}] avg: {np.mean(latencies):.0f}ms", flush=True)

    json.dump(outputs, open(f"{OUT_DIR}/moondream2_outputs.json","w"), indent=2)
    if not latencies:
        print("ERROR: no successful samples!", flush=True); return

    result = {
        "model_id": "moondream2", "params_b": 1.8, "n": len(latencies),
        "mean_ms": round(np.mean(latencies), 1),
        "p95_ms":  round(np.percentile(latencies, 95), 1),
        "fps":     round(1000 / np.mean(latencies), 3),
        "outputs_file": f"{OUT_DIR}/moondream2_outputs.json",
    }
    print(f"\nDONE -- mean={result['mean_ms']}ms | p95={result['p95_ms']}ms | fps={result['fps']}", flush=True)

    results_file = f"{OUT_DIR}/latency_results.json"
    existing = json.load(open(results_file)) if os.path.exists(results_file) else []
    existing = [r for r in existing if r["model_id"] != "moondream2"]
    existing.append(result)
    json.dump(existing, open(results_file,"w"), indent=2)
    print(f"Saved -> {results_file}", flush=True)

if __name__ == "__main__":
    main()
