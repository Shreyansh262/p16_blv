# 🤖 AI CONTEXT — READ THIS FIRST

> **If you are an AI assistant:** Read this file completely, then read [[15_MENTOR_UPDATES]] and [[14_PROGRESS_TRACKER]]. These three files give you full current context. Do not assume anything not written here.

---

## Project Identity

- **Course:** CS671 — Deep Learning | **Project ID:** P16
- **Team:** 9 members | **Mentor:** Sushovan Jena
- **GPU:** 8x NVIDIA RTX A6000 (48GB each) — use GPU 6 by default (most free)
- **Server:** cs671_user2@10.8.1.106 | **Project root:** ~/p16_blv
- **Conda env:** blv | **Timeline:** ~6 weeks

---

## Published Paper This Builds On

- *"Towards Blind and Low-Vision Accessibility of Lightweight VLMs and Custom LLM-Evals"*
- ACL MMLoSo 2025 | https://aclanthology.org/2025.mmloso-1.8/
- What it did: evaluated SmolVLM2 variants on AVCaps + Charades, deployed on phone, introduced Multi-Context BLV Framework and Navigational Assistance Framework
- **This project:** adds RL-based fine-tuning on top of that baseline

---

## One-Line Summary

> Fine-tune SmolVLM2-500M using Teacher-Student pipeline + DPO so it gives detailed, spatially-aware BLV descriptions on a phone — with latency as the #1 priority.

---

## ⚠️ MENTOR INPUT STATUS — ACCEPTED VS REJECTED

Two mentor meetings happened. NOT all inputs were accepted. See [[15_MENTOR_UPDATES]] for full reasoning.

| Mentor Input | Decision | Reason |
|---|---|---|
| AD guidelines prompt | ✅ KEEP | Better training data, zero cost |
| Multi-turn conversations | 🔴 DROPPED | Directly hurts mobile latency |
| Improve metrics beyond paper | ✅ KEEP | No latency impact |
| Better student model | 🔴 STAY ≤1B | Going to 3B destroys latency |
| Human annotations + RL | ✅ KEEP | Already planned, confirmed |
| Mobile RAG redesign | ✅ KEEP | Core to latency goal |
| Latency = #1 priority | ✅ KEEP | Central to PS |
| Teacher embedding distillation | 🔴 DROPPED | Complex, zero latency benefit |
| LoRA fine-tuning | ✅ KEEP | Already planned, confirmed |
| 20B teacher model | 🟡 TEST FIRST | Test if fits and fast enough |
| Selective decoding | ✅ KEEP | Direct latency benefit at inference |
| SDPO | 🟡 STRETCH GOAL | Only after standard DPO works |
| Open-R1 reasoning traces | 🔴 DROPPED | More tokens = worse latency |
| LLaVA-558K | 🟡 FILTERED SUBSET | 10-20K relevant examples only |
| OpenRouter parallel generation | ✅ KEEP | Free, increases data volume |
| Self-distillation | 🔴 DROPPED | Wrong priority, time sink |
| Diffusion model | 🔴 DROPPED | Separate project entirely |
| DINO for keyframes | ✅ KEEP | DINO at training time, LUV at inference |

---

## Final Pipeline

```
Raw Video
    ↓
[[02_LUV_KEYFRAME_EXTRACTION]] — LUV at inference, DINO option at training
    ↓
[[03_RAG_CONTEXT]] — SQLite on-device, flat lookup, <50ms
    ↓
[[04_TEACHER_DATA_GEN]] — AD guidelines, Qwen2-VL-7B, OpenRouter, LLaVA subset
    ↓
[[05_SFT_TRAINING]] — Standard SFT + LoRA (no embedding distillation)
    ↓
[[06_DPO_TRAINING]] — Standard DPO (SDPO as stretch goal)
    ↓
[[07_ABLATION]] — Baseline vs SFT vs SFT+DPO
    ↓
[[08_EVALUATION]] — Paper frameworks + improved metrics
    ↓
[[09_DEPLOYMENT]] — GGUF + selective decoding + latency #1
```

---

## Models

| Role | Model | Decision |
|------|-------|----------|
| Teacher | Qwen2-VL-7B-Instruct | ✅ Default |
| Teacher upgrade | InternVL2-20B | 🟡 Test fit on GPU 6 first |
| Student | SmolVLM2-500M | ✅ Final — do NOT go above 1B params |

---

## Daily Server Commands

```bash
ssh cs671_user2@10.8.1.106
conda activate blv
cd ~/p16_blv
tmux attach -t p16       # if session exists
tmux new -s p16          # if no session
nvidia-smi               # check GPU before running anything
```

---

## File Navigation

- [[00_AI_READ_THIS_FIRST]] ← you are here
- [[01_PROBLEM_EXPLAINED_SIMPLY]]
- [[02_LUV_KEYFRAME_EXTRACTION]]
- [[03_RAG_CONTEXT]]
- [[04_TEACHER_DATA_GEN]]
- [[05_SFT_TRAINING]]
- [[06_DPO_TRAINING]]
- [[07_ABLATION]]
- [[08_EVALUATION]]
- [[09_DEPLOYMENT]]
- [[10_DATASETS]]
- [[11_TEAM_AND_ROLES]]
- [[12_MASTER_TIMELINE]]
- [[13_DECISIONS_AND_BLOCKERS]]
- [[14_PROGRESS_TRACKER]] ← check this for current status
- [[15_MENTOR_UPDATES]] ← read second after this file
- [[16_SERVER_AND_CODE]] ← server setup, paths, all commands

*Last updated: March 18, 2026*
