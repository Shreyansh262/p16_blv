import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.chdir("/usershome/cs671_user2/p16_blv")

import torch, json, glob
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

BASE_MODEL  = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
GRPO_PATH   = "models/student/grpo/best"
OUTPUT_JSON = "results/inference/comparisons/raw_captions_template.json"

TEMPLATE_PROMPT = (
    "You are a BLV navigation assistant.\n"
    "Format you must follow:\n"
    "[ENVIRONMENT TYPE]. [CAUTION: hazard + distance if present]. "
    "[Person description + distance + direction]. [Key obstacle + distance + direction].\n"
    "Rules:\n"
    "- ENVIRONMENT: state only what you see (Kitchen/Hallway/Living room/Bedroom/Outdoor)\n"
    "- CAUTION: only if real hazard visible — wet floor, steps, moving obstacle\n"
    "- All distances in meters\n"
    "- Present tense, 4 sentences max"
)

TARGET_IDS = ["46GP8", "N11GT", "0IH69", "IA2O6", "PFOD8"]
KF_DIR     = "data/keyframes/charades"

video_frames = {}
for vid in TARGET_IDS:
    paths = sorted(glob.glob(f"{KF_DIR}/{vid}_kf*.jpg"))[:4]
    assert len(paths) == 4, f"Missing keyframes for {vid}: {paths}"
    video_frames[vid] = paths

with open("data/generated/all_captions_gemma.json") as f:
    all_data = json.load(f)
ref_map = {e["video_id"]: e.get("blv_description", "") for e in all_data}

Path("outputs").mkdir(exist_ok=True)

SEP  = "=" * 70
THIN = "-" * 50

def p(msg=""):
    print(msg, flush=True)

p(SEP)
p("GRPO_best — TEMPLATE PROMPT — 5 videos")
p(f"Videos : {TARGET_IDS}")
p(f"Prompt :\n{TEMPLATE_PROMPT}")
p(SEP)

p(f"\nLoading GRPO best ({GRPO_PATH}) ...")
proc = AutoProcessor.from_pretrained(BASE_MODEL)
base = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL, dtype=torch.bfloat16, device_map={"": "cuda:0"})
grpo = PeftModel.from_pretrained(base, GRPO_PATH).merge_and_unload()
grpo.eval()
p("GRPO best ready.\n")

results = {}

for vid in TARGET_IDS:
    frames = [Image.open(path).convert("RGB") for path in video_frames[vid]]
    n = len(frames)

    messages   = [{"role": "user", "content":
                   [{"type": "image"}] * n + [{"type": "text", "text": TEMPLATE_PROMPT}]}]
    text_in    = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs     = proc(text=text_in, images=frames, return_tensors="pt")
    inputs     = {k: v.to("cuda:0") for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = grpo.generate(**inputs, max_new_tokens=300, do_sample=False)

    caption = proc.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
    results[vid] = {"gemma_ref": ref_map.get(vid, ""), "grpo_template": caption}

    p(f"\n{THIN}")
    p(f"VIDEO: {vid}")
    p(THIN)
    p(f"[GEMMA REF]\n  {ref_map.get(vid, '')}")
    p(f"\n[GRPO_BEST — template]\n  {caption}")

output = {
    "prompt_type": "template",
    "prompt": TEMPLATE_PROMPT,
    "model": GRPO_PATH,
    "videos": TARGET_IDS,
    "results": {
        vid: {
            "keyframes":     video_frames[vid],
            "gemma_ref":     results[vid]["gemma_ref"],
            "grpo_template": results[vid]["grpo_template"],
        }
        for vid in TARGET_IDS
    }
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(output, f, indent=2)

p(f"\nSaved → {OUTPUT_JSON}")
p("=== DONE ===")
