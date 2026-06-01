import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.chdir("/usershome/cs671_user2/p16_blv")

import torch, json, gc
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, SmolVLMForConditionalGeneration
from peft import PeftModel

BASE_MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
SFT_V2     = "models/student/sft_v2/best"
GRPO_BEST  = "models/student/grpo/best"
PATCH_V2   = "models/student/sft_patch_grpo_v2"
V1_DIR     = "models/student/final_merged"
V2_DIR     = "models/student/final_merged_v2"
SEP        = "=" * 72

# Test sample: 3YY88, 4 keyframes
SAMPLE_ID = "3YY88"
KF_DIR    = "data/keyframes/charades"
kf_paths  = sorted(Path(KF_DIR).glob(SAMPLE_ID + "_kf*.jpg"))
assert kf_paths, "No keyframes for " + SAMPLE_ID
frames    = [Image.open(p).convert("RGB") for p in kf_paths]
print("Loaded", len(frames), "frames for", SAMPLE_ID, ":", [p.name for p in kf_paths], flush=True)

with open("data/generated/grpo_val.json") as f:
    val_data = json.load(f)
sample = next(s for s in val_data if s["sample_id"] == SAMPLE_ID)
PROMPT = sample["prompt"]
REF    = sample["gemma_reference"]

# Step 1: Build final_merged_v2 on CPU if needed
if not Path(V2_DIR).exists():
    print("\n" + SEP, flush=True)
    print("Building final_merged_v2 (base -> sft_v2 -> grpo -> sft_patch) on CPU ...", flush=True)
    print(SEP, flush=True)

    m = SmolVLMForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cpu")
    proc = AutoProcessor.from_pretrained(SFT_V2)

    print("  Merging sft_v2/best ...", flush=True)
    m = PeftModel.from_pretrained(m, SFT_V2).merge_and_unload()

    print("  Merging grpo/best ...", flush=True)
    m = PeftModel.from_pretrained(m, GRPO_BEST).merge_and_unload()

    print("  Merging sft_patch_grpo_v2 ...", flush=True)
    m = PeftModel.from_pretrained(m, PATCH_V2).merge_and_unload()

    print("  Saving to " + V2_DIR + " ...", flush=True)
    m.save_pretrained(V2_DIR, safe_serialization=True)
    proc.save_pretrained(V2_DIR)
    del m
    gc.collect()
    print("  final_merged_v2 saved.", flush=True)
else:
    print("final_merged_v2 already exists -- skipping build.", flush=True)

# Inference helper
def infer(model_dir, frames, prompt):
    proc  = AutoProcessor.from_pretrained(model_dir)
    model = SmolVLMForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    n = len(frames)
    messages = [{"role": "user", "content":
                 [{"type": "image"}] * n + [{"type": "text", "text": prompt}]}]
    text_in    = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs     = proc(text=text_in, images=frames, return_tensors="pt")
    inputs     = {k: v.to("cuda") for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    result = proc.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
    del model, proc
    torch.cuda.empty_cache()
    return result

# Run both models
print("\n" + SEP, flush=True)
print("INFERENCE -- sample: " + SAMPLE_ID + "  |  GPU: 2  |  frames: " + str(len(frames)), flush=True)
print(SEP, flush=True)

print("\n>>> final_merged    (v1: base -> grpo -> sft_patch) ...", flush=True)
v1_out = infer(V1_DIR, frames, PROMPT)
print("    done.", flush=True)

print(">>> final_merged_v2 (v2: base -> sft_v2 -> grpo -> sft_patch) ...", flush=True)
v2_out = infer(V2_DIR, frames, PROMPT)
print("    done.", flush=True)

# Side-by-side output
print("\n" + SEP)
print("SAMPLE    : " + SAMPLE_ID)
print("KEYFRAMES : " + str([p.name for p in kf_paths]))
print("PROMPT    : " + PROMPT[:100].strip() + "...")
print(SEP)
print()
print("[GEMMA REFERENCE]")
print(REF)
print()
print("[V1 -- final_merged   (base -> grpo -> sft_patch)]")
print(v1_out)
print()
print("[V2 -- final_merged_v2 (base -> sft_v2 -> grpo -> sft_patch)]")
print(v2_out)
print()
print(SEP)
