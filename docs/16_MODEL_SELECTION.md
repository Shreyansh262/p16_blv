# 16 — Model Selection (Final)

> **Related:** [[00_AI_READ_THIS_FIRST]] | [[15_MENTOR_UPDATES]]

---

## Student Model — DECIDED

**SmolVLM2-500M — keep.**

Mentor said "similar parameter model." Team: ≤1B strictly.
Your paper already has 500M and 2.2B numbers — direct comparison.
Going to 3B destroys mobile latency (15-25s vs 3-5s inference).

HuggingFace: HuggingFaceTB/SmolVLM2-500M-Video-Instruct

---

## Teacher Model — TEST FIRST

Current plan: Qwen2-VL-7B-Instruct

Test InternVL2-20B:
```bash
export CUDA_VISIBLE_DEVICES=6
python -c "
from transformers import AutoModel, BitsAndBytesConfig
import torch, time
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
t0 = time.time()
model = AutoModel.from_pretrained('OpenGVLab/InternVL2-20B',
    quantization_config=bnb, device_map='auto', trust_remote_code=True)
print(f'Load time: {time.time()-t0:.1f}s')
import subprocess
r = subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],
    capture_output=True, text=True)
print('VRAM used:', r.stdout.strip(), 'MiB')
"
```

Decision rule:
- Fits in VRAM AND <8s per video → use InternVL2-20B
- Does not fit OR >8s → stay with Qwen2-VL-7B

Rationale: Quality gain 7B→20B is marginal when student is 500M params.
Timeline risk of 2-3x slower generation not worth it.

---

## Multi-Turn — Resolved

Mentor requested multi-turn. Team rejected.
Reason: compounds latency on mobile with every turn.
Single-turn with complete AD description is the correct approach.
Logged in [[13_DECISIONS_AND_BLOCKERS]].


---

## Final Model Selection and Deployment (May 8, 2026)

### Selected: SmolVLM2-500M with GRPO
- 500M params, LoRA-adaptable, multimodal
- Training: GRPO, 9,076 steps, May 8
- Base: SFT v2 (MCF=4.38, NAF=4.10)
- Result: NAF improved to 4.20

### All Conditions
- A (Base): MCF=3.90, NAF=3.96
- B (SFT v2): MCF=4.38, NAF=4.10 - DEPLOY (best)
- C (DPO v2): MCF=3.89
- D (RLAIF-V): MCF=3.88
- E (SFT v3): MCF=4.33
- F (GRPO): NAF=4.20 - RL effective
- G (SimPO): pending
- H (KTO): pending

### Deployment Package
- Model: 481 MB (290 LM + 191 mmproj)
- Backend: llama-mtmd-cli on Termux
- Latency: 504 ms TTFT, 17.71 tok/s, 44.2s per-frame
- Memory: 619 MB peak RAM
- Device: Samsung Galaxy A55 5G
- Status: Production-ready, no crashes