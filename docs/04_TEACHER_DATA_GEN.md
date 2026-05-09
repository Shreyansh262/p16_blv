# 04 — Teacher Data Generation

> **Related:** [[00_AI_READ_THIS_FIRST]] | [[02_LUV_KEYFRAME_EXTRACTION]] | [[03_RAG_CONTEXT]] | [[15_MENTOR_UPDATES]]
> **Owner:** Data Team | **Status:** See [[17_CURRENT_PROGRESS]]

---

## What Changed From Original Plan

| Original | Final |
|----------|-------|
| Custom BLV bullet prompt | AD guidelines prompt |
| Multi-turn conversations | REMOVED — hurts latency |
| Reasoning traces | REMOVED — more tokens = worse latency |
| Single-turn only | CONFIRMED |
| OpenRouter parallel generation | ADDED |
| LLaVA-558K | 10-20K filtered subset only |
| 7B teacher | Test 20B first, likely stay 7B |

---

## Teacher Model

Current plan: Qwen2-VL-7B-Instruct (~8GB in 4-bit on A6000)
Test InternVL2-20B — if VRAM fits AND generation <8s per video, upgrade.
Decision: Log in [[13_DECISIONS_AND_BLOCKERS]] before generating any data.

---

## System Prompt — AD Guidelines Version

Download before finalizing prompt:
- ITC Guidance on Standards for Audio Description
- Netflix Audio Description Style Guide  
- https://www.acb.org/adp/guidelines.html

```
You are a professional audio describer for blind and low-vision (BLV)
audiences, following ITC and Netflix Audio Description standards.

Describe video content using these professional AD guidelines:
- Use present tense throughout
- Lead with the most safety-critical visual information first
- Specify exact spatial positions: left, right, center, near, far with distances
- Describe people by observable features only: clothing, position, movement direction
- Name the environment type in the first sentence
- Mention surface conditions, obstacles, steps, ramps, wet floors
- Active voice: "A person opens the door" not "The door is opened"
- Do not describe audio, only visual content
- Be concise — every word serves the BLV user. Maximum 4 sentences.

Produce a description a BLV person could use to safely navigate this scene.
```

---

## Generation Sources

### Track 1: Local Teacher (A6000)
Run Qwen2-VL-7B on keyframes + RAG context. ~10 hours for full dataset.
Script: src/data_pipeline/03_generate_teacher_captions.py

### Track 2: OpenRouter (Free, Parallel)
Llama-3.3-70B free tier. Text-only — use rough caption as input, rewrite to BLV format.
Sign up: openrouter.ai

### Track 3: LLaVA-558K (10-20K filtered)
Scene description conversations only. SFT warm-up, not main training data.

---

## Output Format (Single-Turn Only — No multiturn field)

```json
{
  "video_id": "charades_001",
  "dataset": "charades",
  "generation_source": "local_teacher",
  "teacher_model": "Qwen2-VL-7B-Instruct",
  "keyframe_paths": ["frame_000.jpg", "frame_047.jpg"],
  "rag_context": "Kitchen environment...",
  "blv_description": "A person in a grey t-shirt stands at a kitchen counter..."
}
```

---

## Running It

```bash
export CUDA_VISIBLE_DEVICES=6
cd ~/p16_blv

# Test 10 videos first
python src/data_pipeline/03_generate_teacher_captions.py --dataset charades --max_videos 10

# Check quality manually
cat data/generated/qwen_captions.json | python -m json.tool | head -50

# Full run
python src/data_pipeline/03_generate_teacher_captions.py --dataset charades --max_videos 10000
```

---

## Checklist

- [ ] Download and read AD guidelines
- [ ] Write and test new system prompt on 20 videos
- [ ] Decide teacher model — log in [[13_DECISIONS_AND_BLOCKERS]]
- [ ] Set up OpenRouter API key
- [ ] Test run: 10 videos, check quality
- [ ] Full Charades generation (~10k)
- [ ] AVCaps generation (~2k)
- [ ] Filter LLaVA-558K for 10-20K scene conversations

*Prerequisite: [[02_LUV_KEYFRAME_EXTRACTION]] must complete first*
*Next: [[05_SFT_TRAINING]]*


---

## Final Caption Statistics (Training Complete)

### Qwen2-VL-7B Captions
- Total: 10,107 videos
- Used for: SFT v2 (MCF 4.38), DPO pairs

### Gemma-2-9B Captions
- Total: 10,107 videos
- Used for: SFT v3 (MCF 4.33), RLAIF-V DPO, KTO

### DPO Pairs
- File: data/generated/dpo_pairs.json
- Count: 7,842 pairs (7,057 train / 785 val)
- Quality: reward_accuracy 99.6%

### Training Results
- SFT v2 (Qwen): MCF=4.38, NAF=4.10 (best)
- SFT v3 (Gemma): MCF=4.33, NAF=3.98
- GRPO (SFT v2 base): NAF improved to 4.20
- DPO v2/RLAIF/SimPO/KTO all trained; GRPO most effective