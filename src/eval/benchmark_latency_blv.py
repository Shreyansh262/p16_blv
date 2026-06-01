"""
Multi-model latency + BLV inference benchmark.
Runs 50 samples from balanced_eval.json on each model, records ms/frame.
Saves outputs for BLV judge in results/benchmark_graph/
"""

import sys, time, json, os, gc, importlib.util, importlib.machinery
import numpy as np

# torchvision mock (exact pattern from run_inference_all_conditions.py)
import enum as _enum
class _InterpolationMode(_enum.Enum):
    NEAREST="nearest"; BILINEAR="bilinear"; BICUBIC="bicubic"
    BOX="box"; HAMMING="hamming"; LANCZOS="lanczos"; NEAREST_EXACT="nearest-exact"
def _mod(n):
    s = importlib.machinery.ModuleSpec(n, None, is_package=True)
    m = importlib.util.module_from_spec(s)
    return m
_tv = _mod("torchvision"); _tr = _mod("torchvision.transforms")
_tr.InterpolationMode = _InterpolationMode; _tv.transforms = _tr
for _s in ["_meta_registrations","datasets","models","ops","utils","extension","io",
           "transforms.v2","transforms.v2.functional"]:
    sys.modules["torchvision."+_s] = _mod("torchvision."+_s)
sys.modules["torchvision"] = _tv; sys.modules["torchvision.transforms"] = _tr
del _enum, _InterpolationMode, _mod, _tv, _tr, _s

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    SmolVLMForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    PaliGemmaForConditionalGeneration,
    AutoModelForCausalLM,
)
from peft import PeftModel

EVAL_JSON = "data/augmented/balanced_splits/balanced_eval.json"
OUT_DIR   = "results/benchmark_graph"
N_SAMPLES = 50
MAX_NEW   = 120
SEED      = 42

AD_SYSTEM = (
    "You are an assistive AI for blind and low-vision users. "
    "Describe the scene for safe navigation: mention directions, distances, hazards, obstacles."
)
BLV_PROMPT = (
    "Describe this scene for a blind user. "
    "Mention spatial layout, directions (left/right/ahead), distances, obstacles, and any hazards."
)

MODELS = [
# {"id":"SmolVLM2-500M-base",    "hf_id":"HuggingFaceTB/SmolVLM2-500M-Video-Instruct", "model_cls":"smolvlm",   "adapter":None,                              "params_b":0.5, "gpu":1},
# {"id":"sft_patch_grpo_v2-ours","hf_id":"HuggingFaceTB/SmolVLM2-500M-Video-Instruct", "model_cls":"smolvlm",   "adapter":"models/student/sft_patch_grpo_v2","params_b":0.5, "gpu":0},
# {"id":"SmolVLM2-2.2B",         "hf_id":"HuggingFaceTB/SmolVLM2-2.2B-Instruct",       "model_cls":"smolvlm",   "adapter":None,                              "params_b":2.2, "gpu":1},
# {"id":"Qwen2.5-VL-3B",         "hf_id":"Qwen/Qwen2.5-VL-3B-Instruct",                "model_cls":"qwen2vl",   "adapter":None,                              "params_b":3.0, "gpu":1},
#     {"id":"PaliGemma2-3B",         "hf_id":"google/paligemma2-3b-pt-224",                 "model_cls":"paligemma", "adapter":None,                              "params_b":3.0, "gpu":1},
    {"id":"moondream2",            "hf_id":"vikhyatk/moondream2",                         "model_cls":"causal",    "adapter":None,                              "params_b":1.8, "gpu":1},
]

CLS_MAP = {
    "smolvlm":   SmolVLMForConditionalGeneration,
    "qwen2vl":   Qwen2_5_VLForConditionalGeneration,
    "paligemma": PaliGemmaForConditionalGeneration,
    "causal":    AutoModelForCausalLM,
}

def load_images(paths):
    imgs = []
    for p in paths:
        if os.path.exists(p):
            imgs.append(Image.open(p).convert("RGB"))
        else:
            alt = os.path.join("data/keyframes/charades", os.path.basename(p))
            if os.path.exists(alt):
                imgs.append(Image.open(alt).convert("RGB"))
    return imgs[:2]

def build_messages(imgs):
    content = [{"type":"image"} for _ in imgs]
    content.append({"type":"text","text":BLV_PROMPT})
    return [{"role":"system","content":AD_SYSTEM},{"role":"user","content":content}]

def run_model(cfg, samples):
    device = f"cuda:{cfg['gpu']}"
    cache  = "models/.hf_cache"
    cls    = CLS_MAP[cfg["model_cls"]]
    print(f"\n{'='*60}", flush=True)
    print(f"Loading {cfg['id']} on GPU {cfg['gpu']} ...", flush=True)
    try:
        processor = AutoProcessor.from_pretrained(cfg["hf_id"], cache_dir=cache, trust_remote_code=True)
        model = cls.from_pretrained(cfg["hf_id"], cache_dir=cache,
            torch_dtype=torch.bfloat16, device_map={"":cfg["gpu"]}, trust_remote_code=True)
        if cfg["adapter"]:
            model = PeftModel.from_pretrained(model, cfg["adapter"])
            model = model.merge_and_unload()
        model.eval()
        print("  Model loaded.", flush=True)
    except Exception as e:
        print(f"  LOAD FAILED: {e}", flush=True)
        return None

    latencies, outputs = [], []
    for i, s in enumerate(samples):
        imgs = load_images(s.get("keyframe_paths", []))
        if not imgs:
            print(f"  sample {i}: no images, skip", flush=True)
            continue
        msgs = build_messages(imgs)
        try:
            text = processor.apply_chat_template(msgs, add_generation_prompt=True)
            inputs = processor(text=text, images=imgs, return_tensors="pt").to(device)
        except Exception as e:
            print(f"  sample {i} preprocess error: {e}", flush=True)
            continue
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False, temperature=None, top_p=None)
        torch.cuda.synchronize(device)
        ms = (time.perf_counter() - t0) * 1000
        latencies.append(ms)
        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = out_ids[0][prompt_len:]
        text_out = processor.decode(gen_ids, skip_special_tokens=True).strip()
        outputs.append({"video_id":s["video_id"],"output":text_out,"latency_ms":ms})
        if (i+1) % 10 == 0:
            print(f"  [{i+1}/{len(samples)}] avg so far: {np.mean(latencies):.0f}ms", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_file = f"{OUT_DIR}/{cfg['id']}_outputs.json"
    json.dump(outputs, open(out_file,"w"), indent=2)
    result = {
        "model_id":cfg["id"], "params_b":cfg["params_b"], "n":len(latencies),
        "mean_ms":round(np.mean(latencies),1) if latencies else 0,
        "p95_ms":round(np.percentile(latencies,95),1) if latencies else 0,
        "fps":round(1000/np.mean(latencies),3) if latencies else 0,
        "outputs_file":out_file,
    }
    print(f"  DONE -- mean={result['mean_ms']}ms | p95={result['p95_ms']}ms | fps={result['fps']}", flush=True)
    del model; torch.cuda.empty_cache(); gc.collect()
    return result

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = json.load(open(EVAL_JSON))
    rng  = np.random.default_rng(SEED)
    idx  = rng.choice(len(data), size=min(N_SAMPLES,len(data)), replace=False)
    samples = [data[i] for i in idx]
    print(f"Loaded {len(samples)} samples.", flush=True)
    results = []
    for cfg in MODELS:
        r = run_model(cfg, samples)
        if r:
            results.append(r)
            json.dump(results, open(f"{OUT_DIR}/latency_results.json","w"), indent=2)
            print(f"\nIntermediate save -> {OUT_DIR}/latency_results.json", flush=True)
    print("\n\n=== FINAL LATENCY TABLE ===")
    print(f"{'Model':<30} {'Params':>7} {'Mean ms':>9} {'P95 ms':>8} {'FPS':>7}")
    print("-"*65)
    for r in results:
        print(f"{r['model_id']:<30} {r['params_b']:>6.1f}B {r['mean_ms']:>9.1f} {r['p95_ms']:>8.1f} {r['fps']:>7.3f}")

if __name__ == "__main__":
    main()
