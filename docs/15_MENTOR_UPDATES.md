# 15 — Mentor Updates and Acceptance Decisions

> **Read this after [[00_AI_READ_THIS_FIRST]].**
> This file logs every mentor input, whether it was accepted or rejected, and exactly why.
> When inputs conflict with the PS goal (latency-first mobile deployment), they are rejected.

---

## Core Principle Applied to All Decisions

The PS says: *"end application is to be deployed in a mobile device."*
The mentor said: *"Main agenda is to reduce the latency."*
Therefore: **any change that hurts latency is rejected, regardless of quality benefit.**

---

## Meeting 1 — 8 March 2026

### ✅ ACCEPTED — AD Guidelines Prompt
**What:** Replace custom BLV bullet-point system prompt with official Audio Description standards (ITC/Netflix/BBC).
**Why accepted:** Better training data quality. Zero latency impact. More credible academically.
**Action:** Read ITC guidelines PDF. Update system prompt in [[04_TEACHER_DATA_GEN]] before generating any captions.
**Sources:** https://www.acb.org/adp/guidelines.html | Netflix AD Style Guide

---

### 🔴 REJECTED — Multi-Turn Conversations
**What mentor said:** Fine-tune for multi-turn Q&A conversations.
**Why rejected:** Multi-turn = longer context = more tokens at inference = higher latency on phone. The PS is a single-shot task: point camera, get description. A blind person needs one fast, complete description — not a dialogue.
**Decision:** Single-turn only. Make the single description so complete that follow-ups are unnecessary.
**Note:** If mentor insists, the only acceptable implementation is visual feature caching (see [[09_DEPLOYMENT]]) — encode video once, cache embeddings, text-only for follow-ups. But we are not training for this.

---

### ✅ ACCEPTED — Improve Metrics Beyond Paper
**What:** Don't just use paper's metrics — improve and extend them.
**Why accepted:** No latency impact. Better evaluation = more credible paper.
**Action:** See [[08_EVALUATION]] for extended metric plan.

---

### 🔴 REJECTED — Better Student Model (Going Above 1B)
**What mentor said:** Look for a similar parameter better model than SmolVLM2.
**Why rejected:** "Similar parameter" means ≤1B. Candidates like Qwen2.5-VL-3B are 6x heavier — latency jumps from ~5s to ~25s on phone. Unacceptable.
**Decision:** Keep SmolVLM2-500M. If a genuinely better model exists at ≤1B params, test it. Otherwise stay.
**Test to run:** Quick no-finetune BLV score comparison on 10 videos between SmolVLM2-500M and any ≤1B candidate.

---

### ✅ ACCEPTED — Human Annotations + RL
**What:** Human-annotated DPO preference pairs required.
**Why accepted:** Already in original plan. PS requires it. Now confirmed by mentor.
**Action:** See [[06_DPO_TRAINING]] for annotation process.

---

### ✅ ACCEPTED — Mobile-Optimized RAG
**What mentor said (points 6, 7, 8):** RAG must work on mobile with low latency and low storage. Find optimal vector storage.
**Why accepted:** Directly serves latency goal.
**Solution:** SQLite database (~100KB), precomputed embeddings, flat lookup table pattern. See [[03_RAG_CONTEXT]].

---

### 🔴 REJECTED — Teacher Embedding Distillation
**What mentor said:** Use teacher embeddings instead of text SFT.
**Why rejected:** Changes how the student learns, not how fast it runs. Deployed model is identical size regardless of training method. Adds significant implementation complexity (projection layers, dual-encoder loading, more VRAM) for zero latency benefit.
**Decision:** Standard text-based SFT with LoRA. Simpler, proven, same deployed model.

---

### ✅ ACCEPTED — LoRA Fine-Tuning
**What:** Use LoRA/QLoRA for efficient fine-tuning.
**Why accepted:** Already in plan, confirmed by mentor.

---

### 🟡 CONDITIONAL — 20B Teacher Model
**What mentor said:** Maybe look at 20B teacher.
**Decision:** Test InternVL2-20B first. Load in 4-bit on GPU 6, check VRAM, check generation speed (seconds per video). If >8s per video, too slow for our timeline — stick with 7B.
**Test command:**
```bash
python -c "
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch
cfg = BitsAndBytesConfig(load_in_4bit=True)
m = AutoModelForCausalLM.from_pretrained('InternVL2-20B', quantization_config=cfg, device_map='auto')
print('VRAM used:', torch.cuda.memory_allocated()/1e9, 'GB')
"
```

---

## Meeting 2 — 11 March 2026

### ✅ ACCEPTED — Selective Decoding
**What:** Constrain model to generate only BLV-relevant tokens, terminate earlier.
**Why accepted:** Directly reduces inference latency on mobile. This is the highest-priority Meeting 2 item.
**Implementation:** HuggingFace LogitsProcessor to upweight spatial/directional tokens. See [[09_DEPLOYMENT]].

---

### 🟡 STRETCH GOAL — SDPO
**What:** Selective/Span-level DPO — weight preference signal on spatial token spans.
**Decision:** Run standard DPO first and get a working checkpoint. Only attempt SDPO as a second run if time allows. Do not block main timeline on this.
**Why not fully rejected:** Genuinely novel contribution if it works. But high implementation risk.

---

### 🔴 REJECTED — Open-R1 Reasoning Traces
**What:** Train on chain-of-thought reasoning traces.
**Why rejected:** Reasoning traces make the model generate long internal monologue before the answer. That's 2-3x more tokens at inference = directly worse latency on phone. Exact opposite of what we need.

---

### 🟡 CONDITIONAL — LLaVA-558K Instruction Data
**What:** Use LLaVA-558K as SFT warm-up data.
**Decision:** Use a filtered subset of 10-20K relevant scene description conversations only. Do NOT train on the full 558K — most is irrelevant to BLV scenes and it would dominate training.
**Filter criteria:** Keep only conversations where task is describing indoor/outdoor scenes.

---

### ✅ ACCEPTED — OpenRouter for Parallel Generation
**What:** Use OpenRouter free API to generate additional captions in parallel.
**Why accepted:** Zero cost, increases dataset diversity. Text-only on free tier but useful for augmentation.
**Setup:** Sign up at openrouter.ai, use Llama 3.3 70B free tier.

---

### 🔴 REJECTED — Self-Distillation
**What:** Iterative self-improvement loop after initial SFT.
**Why rejected:** Wrong priority. Requires multiple training rounds (each ~10 hours), reliable auto-scoring, complex implementation. Does not improve latency. Time that could be spent on deployment engineering.

---

### 🔴 REJECTED — Diffusion Model
**What:** Use video diffusion model for synthetic training data generation.
**Why rejected:** This is a completely separate research project. You already have VideoA11y (40k videos), Charades (10k), AVCaps (2k). No data scarcity problem. Out of scope for 6-week timeline.
**Note for report:** Mention as future work.

---

### ✅ ACCEPTED — DINO for Keyframe Selection
**What:** Use DINOv2 features for semantically-aware keyframe selection.
**Decision:** DINO at training time (on A6000, speed doesn't matter). LUV at inference on mobile (essentially 0ms overhead). Best of both worlds.
**Why accepted:** Better training data quality, no inference latency cost.

---

## Summary: What Changed From Original Plan

| Component | Original | Final |
|---|---|---|
| System prompt | Custom BLV bullets | AD guidelines (ITC/Netflix) |
| Conversation format | Single-turn | Single-turn (multi-turn dropped) |
| Student model | SmolVLM2-500M | SmolVLM2-500M (≤1B confirmed) |
| Teacher model | Qwen2-VL-7B | Qwen2-VL-7B (test 20B optionally) |
| SFT method | Text imitation | Text imitation (distillation dropped) |
| DPO variant | Standard DPO | Standard DPO (SDPO as stretch) |
| RAG | Standard FAISS | SQLite flat lookup, mobile-optimized |
| Keyframe selection | LUV only | LUV at inference + DINO at training |
| Additional data | None | OpenRouter + LLaVA-558K filtered |
| Reasoning data | None | None (dropped) |
| Deployment focus | Secondary | PRIMARY — latency #1 |

