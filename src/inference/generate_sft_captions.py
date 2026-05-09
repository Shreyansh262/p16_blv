import json, os, sys, torch
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, SmolVLMForConditionalGeneration
from peft import PeftModel

os.environ.setdefault('HF_HOME', '/usershome/cs671_user2/p16_blv/models/.hf_cache')

BLV_PROMPT = (
    "You are an audio description system for blind and low vision users. "
    "Follow these rules strictly: "
    "1. First sentence: name the environment and say CAUTION if any hazard is within 2 meters. "
    "2. Every object and person must have a direction (left/right/center/ahead) and distance in meters. "
    "3. People: describe by clothing color, position, and movement direction only. "
    "4. Present tense, active voice. Maximum 4 sentences. Every word must serve navigation. "
    "Describe this video scene for a blind user."
)

CHECKPOINT  = "models/student/sft_v2/best/"
BASE_MODEL  = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
DPO_PAIRS   = "data/generated/dpo_pairs.json"
OUTPUT_FILE = "data/generated/sft_v2_captions.json"
MAX_SIDE    = 364
MODEL_DTYPE = torch.bfloat16


def load_model():
    print("Loading processor...", flush=True)
    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    print("Loading base model...", flush=True)
    base = SmolVLMForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=MODEL_DTYPE,
        device_map={"": 0},
    )
    print("Applying LoRA adapter...", flush=True)
    model = PeftModel.from_pretrained(base, CHECKPOINT)
    model.eval()
    print(f"Model ready. Device: {next(model.parameters()).device}", flush=True)
    return model, processor


def generate_caption(model, processor, keyframe_paths):
    images = []
    for p in keyframe_paths:
        if os.path.exists(p):
            img = Image.open(p).convert("RGB")
            if max(img.size) > MAX_SIDE:
                img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
            images.append(img)
    if not images:
        return ""

    # Pass actual PIL Image objects in the content dict — this is what works
    # with SmolVLMForConditionalGeneration + apply_chat_template
    messages = [{
        "role": "user",
        "content": (
            [{"type": "image", "image": img} for img in images]
            + [{"type": "text", "text": BLV_PROMPT}]
        ),
    }]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    raw_inputs = processor(text=[prompt], images=images, return_tensors="pt")

    # Cast pixel_values to model dtype (bfloat16); leave int tensors as-is
    inputs = {
        k: (v.to(device=model.device, dtype=MODEL_DTYPE)
            if v.dtype == torch.float32 else v.to(model.device))
        for k, v in raw_inputs.items()
    }

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            temperature=None,
            top_p=None,
            repetition_penalty=1.3,
        )

    prompt_len = inputs["input_ids"].shape[1]
    generated  = out_ids[0][prompt_len:]
    return processor.decode(generated, skip_special_tokens=True).strip()


def main():
    results = {}
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            results = json.load(f)
        print(f"Resuming from {len(results)} existing captions", flush=True)

    with open(DPO_PAIRS) as f:
        data = json.load(f)
    all_pairs = data["train"] + data["val"]

    seen, unique_pairs = set(), []
    for p in all_pairs:
        if p["video_id"] not in seen and p["video_id"] not in results:
            seen.add(p["video_id"])
            unique_pairs.append(p)

    print(f"Videos to process: {len(unique_pairs)}", flush=True)

    model, processor = load_model()

    for i, pair in enumerate(unique_pairs):
        vid = pair["video_id"]
        try:
            caption = generate_caption(model, processor, pair["keyframe_paths"])
            results[vid] = caption
        except Exception as e:
            print(f"Error on {vid}: {e}", flush=True)
            results[vid] = ""

        if (i + 1) % 50 == 0:
            with open(OUTPUT_FILE, "w") as f:
                json.dump(results, f, indent=2)
            print(f"[{i+1}/{len(unique_pairs)}] checkpoint saved — last: {vid}", flush=True)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Done. Total captions: {len(results)}", flush=True)


if __name__ == "__main__":
    main()
