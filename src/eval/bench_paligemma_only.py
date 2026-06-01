import time, json, os
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

EVAL_JSON = "data/augmented/balanced_splits/balanced_eval.json"
OUT_DIR   = "results/benchmark_graph"
MODEL_DIR = "models/baselines/paligemma2_3b_instruct"
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
    return imgs[:1]  # paligemma takes single image

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = json.load(open(EVAL_JSON))
    rng  = np.random.default_rng(SEED)
    idx  = rng.choice(len(data), size=min(N_SAMPLES, len(data)), replace=False)
    samples = [data[i] for i in idx]
    print(f"Loaded {len(samples)} samples.", flush=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
    print(f"Loading PaliGemma2-3B-instruct from {MODEL_DIR} ...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    print("Model loaded.", flush=True)

    latencies, outputs = [], []
    for i, s in enumerate(samples):
        imgs = load_images(s.get("keyframe_paths", []))
        if not imgs:
            print(f"  sample {i}: no images, skip", flush=True); continue
        img = imgs[0]
        try:
            inputs = processor(text=BLV_PROMPT, images=img, return_tensors="pt").to("cuda")
            input_len = inputs["input_ids"].shape[1]
        except Exception as e:
            print(f"  sample {i} preprocess error: {e}", flush=True); continue

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        latencies.append(ms)
        text_out = processor.decode(out_ids[0][input_len:], skip_special_tokens=True).strip()
        outputs.append({"video_id": s["video_id"], "output": text_out, "latency_ms": ms})
        if (i+1) % 10 == 0:
            print(f"  [{i+1}/{len(samples)}] avg: {np.mean(latencies):.0f}ms", flush=True)

    json.dump(outputs, open(f"{OUT_DIR}/PaliGemma2-3B_outputs.json","w"), indent=2)
    if not latencies:
        print("ERROR: no successful samples!", flush=True); return

    result = {
        "model_id": "PaliGemma2-3B", "params_b": 3.0, "n": len(latencies),
        "mean_ms": round(np.mean(latencies), 1),
        "p95_ms":  round(np.percentile(latencies, 95), 1),
        "fps":     round(1000 / np.mean(latencies), 3),
        "outputs_file": f"{OUT_DIR}/PaliGemma2-3B_outputs.json",
    }
    print(f"\nDONE -- mean={result['mean_ms']}ms | p95={result['p95_ms']}ms | fps={result['fps']}", flush=True)

    results_file = f"{OUT_DIR}/latency_results.json"
    existing = json.load(open(results_file)) if os.path.exists(results_file) else []
    existing = [r for r in existing if r["model_id"] != "PaliGemma2-3B"]
    existing.append(result)
    json.dump(existing, open(results_file,"w"), indent=2)
    print(f"Saved -> {results_file}", flush=True)

if __name__ == "__main__":
    main()
