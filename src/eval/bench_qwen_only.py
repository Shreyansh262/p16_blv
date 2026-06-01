import sys, time, json, os, gc, importlib.util, importlib.machinery, enum as _enum, types
import numpy as np

# ── torchvision mock ─────────────────────────────────────────────────────
class _IM(_enum.Enum):
    NEAREST="nearest"; BILINEAR="bilinear"; BICUBIC="bicubic"
    BOX="box"; HAMMING="hamming"; LANCZOS="lanczos"; NEAREST_EXACT="nearest-exact"

def _make_mod(name):
    s = importlib.machinery.ModuleSpec(name, None, is_package=True)
    return importlib.util.module_from_spec(s)

# Build a smart functional module that handles real ops via PIL/torch
import torch as _torch, numpy as _np
from PIL import Image as _PILImage

class _SmartFunctional(types.ModuleType):
    InterpolationMode = _IM
    def pil_to_tensor(self, pic):
        arr = _np.array(pic)
        if arr.ndim == 2: arr = arr[:,:,None]
        return _torch.from_numpy(arr.transpose(2,0,1)).contiguous()
    def resize(self, img, size, interpolation=None, **kw):
        if isinstance(img, _torch.Tensor):
            # simple bilinear resize via interpolate
            if img.dim() == 3: img = img.unsqueeze(0)
            h, w = (size, size) if isinstance(size, int) else size
            img = _torch.nn.functional.interpolate(img.float(), size=(h,w), mode='bilinear', align_corners=False)
            return img.squeeze(0).to(_torch.uint8)
        elif isinstance(img, _PILImage.Image):
            h, w = (size, size) if isinstance(size, int) else size
            return img.resize((w, h), _PILImage.BILINEAR)
        return img
    def normalize(self, tensor, mean, std, **kw):
        mean = _torch.tensor(mean, dtype=tensor.dtype).view(-1,1,1)
        std  = _torch.tensor(std,  dtype=tensor.dtype).view(-1,1,1)
        return (tensor - mean) / std
    def __getattr__(self, name):
        if name.startswith("__"): raise AttributeError(name)
        # return a no-op for anything else
        def _noop(*a, **kw):
            return a[0] if a else None
        return _noop

_tv2f = _SmartFunctional("torchvision.transforms.v2.functional")

def _make_simple(name):
    m = _make_mod(name)
    m.InterpolationMode = _IM
    return m

_tv   = _make_mod("torchvision")
_tr   = _make_simple("torchvision.transforms")
_tv2  = _make_simple("torchvision.transforms.v2")
_tr.InterpolationMode = _IM
_tv.transforms = _tr

for _s in ["_meta_registrations","datasets","models","ops","utils","extension","io"]:
    sys.modules["torchvision."+_s] = _make_mod("torchvision."+_s)

sys.modules["torchvision"]                          = _tv
sys.modules["torchvision.transforms"]               = _tr
sys.modules["torchvision.transforms.v2"]            = _tv2
sys.modules["torchvision.transforms.v2.functional"] = _tv2f
sys.modules["torchvision.transforms.functional"]    = _tv2f

del _enum, _IM, _make_mod, _make_simple
del _tv, _tr, _tv2, _tv2f, _s, _SmartFunctional
# ── end mock ─────────────────────────────────────────────────────────────

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

EVAL_JSON="data/augmented/balanced_splits/balanced_eval.json"
OUT_DIR="results/benchmark_graph"
N_SAMPLES=50; MAX_NEW=120; SEED=42; GPU=0

BLV_PROMPT="Describe this scene for a blind user. Mention spatial layout, directions (left/right/ahead), distances, obstacles, and any hazards."
AD_SYSTEM="You are an assistive AI for blind and low-vision users. Describe the scene for safe navigation: mention directions, distances, hazards, obstacles."

def load_images(paths):
    imgs=[]
    for p in paths:
        if os.path.exists(p): imgs.append(Image.open(p).convert("RGB"))
        else:
            alt=os.path.join("data/keyframes/charades",os.path.basename(p))
            if os.path.exists(alt): imgs.append(Image.open(alt).convert("RGB"))
    return imgs[:2]

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data=json.load(open(EVAL_JSON))
    rng=np.random.default_rng(SEED)
    idx=rng.choice(len(data),size=min(N_SAMPLES,len(data)),replace=False)
    samples=[data[i] for i in idx]
    print(f"Loaded {len(samples)} samples.", flush=True)

    device=f"cuda:{GPU}"
    cache="models/.hf_cache"
    print(f"Loading Qwen2.5-VL-3B on GPU {GPU} (physical GPU6) ...", flush=True)
    processor=AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct",cache_dir=cache,trust_remote_code=True)
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",cache_dir=cache,
        torch_dtype=torch.bfloat16,device_map={"":GPU},trust_remote_code=True)
    model.eval()
    print("Model loaded.", flush=True)

    latencies,outputs=[],[]
    for i,s in enumerate(samples):
        imgs=load_images(s.get("keyframe_paths",[]))
        if not imgs:
            print(f"  sample {i}: no images, skip", flush=True); continue
        content=[{"type":"image"} for _ in imgs]+[{"type":"text","text":BLV_PROMPT}]
        msgs=[{"role":"system","content":AD_SYSTEM},{"role":"user","content":content}]
        try:
            text=processor.apply_chat_template(msgs,add_generation_prompt=True)
            inputs=processor(text=text,images=imgs,return_tensors="pt").to(device)
        except Exception as e:
            print(f"  sample {i} preprocess error: {e}", flush=True); continue
        torch.cuda.synchronize(device)
        t0=time.perf_counter()
        with torch.no_grad():
            out_ids=model.generate(**inputs,max_new_tokens=MAX_NEW,do_sample=False,temperature=None,top_p=None)
        torch.cuda.synchronize(device)
        ms=(time.perf_counter()-t0)*1000
        latencies.append(ms)
        pl=inputs["input_ids"].shape[1]
        text_out=processor.decode(out_ids[0][pl:],skip_special_tokens=True).strip()
        outputs.append({"video_id":s["video_id"],"output":text_out,"latency_ms":ms})
        if (i+1)%10==0:
            print(f"  [{i+1}/{len(samples)}] avg: {np.mean(latencies):.0f}ms", flush=True)

    json.dump(outputs,open(f"{OUT_DIR}/Qwen2.5-VL-3B_outputs.json","w"),indent=2)
    if not latencies:
        print("ERROR: no successful samples!", flush=True); return
    result={"model_id":"Qwen2.5-VL-3B","params_b":3.0,"n":len(latencies),
        "mean_ms":round(np.mean(latencies),1),"p95_ms":round(np.percentile(latencies,95),1),
        "fps":round(1000/np.mean(latencies),3)}
    print(f"\nDONE -- mean={result['mean_ms']}ms | p95={result['p95_ms']}ms | fps={result['fps']}", flush=True)

    results_file=f"{OUT_DIR}/latency_results.json"
    existing=json.load(open(results_file)) if os.path.exists(results_file) else []
    existing=[r for r in existing if r["model_id"]!="Qwen2.5-VL-3B"]
    existing.append(result)
    json.dump(existing,open(results_file,"w"),indent=2)
    print(f"Saved -> {results_file}", flush=True)

if __name__=="__main__":
    main()
