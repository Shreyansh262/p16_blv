# 01 — The Problem, Explained Simply

> **Related:** [[00_AI_READ_THIS_FIRST]] | [[02_LUV_KEYFRAME_EXTRACTION]] | [[04_TEACHER_DATA_GEN]]

---

## The Real-World Problem

Imagine you are completely blind. You're in an unfamiliar building, trying to navigate to a room. You point your phone camera forward and ask an AI: *"What's in front of me?"*

**What a general AI says today:**
> "A hallway with some doors and people."

**What a blind person actually needs:**
> "A corridor extending approximately 8 meters ahead. There is a partially open door on your left, 3 meters away. A person is walking toward you from the far end. There is a small step down approximately 1.5 meters directly ahead — be careful."

The second description follows **BLV guidelines** — the same guidelines Netflix and Amazon follow when writing audio descriptions for visually impaired viewers. These descriptions are:
- **Spatial** — mentions directions (left, right, ahead, behind)
- **Distance-aware** — gives approximate distances in real units
- **Hazard-flagging** — calls out anything that could cause a fall or collision
- **Objective** — describes what IS there, not what it might mean
- **Sequential** — describes what happens in chronological order

---

## Why Current AI Models Fail at This

All the powerful AI models (GPT-4, Gemini, etc.) were trained on general internet data. The internet is written *for sighted people*. Descriptions like "a beautiful sunset" or "a busy street scene" are perfectly fine for a sighted reader — they just need a rough picture. A blind person navigating physically needs completely different information.

So the failure isn't that AI is stupid. It's that AI was never *trained* to think the way BLV guidelines require.

---

## The Second Problem: Size vs. Intelligence

Here's the cruel irony:

- **Big AI models** (7B, 72B parameters) are smart enough to give detailed BLV descriptions — but they need a powerful computer with a dedicated GPU. They cannot run on a phone.
- **Small AI models** (500M parameters) can run on a phone in real time — but they're not smart enough yet for high-quality BLV descriptions.

A blind person needs something that runs on a *phone* (because that's what they carry). So you're stuck needing small AND smart simultaneously.

---

## The Solution: Teaching a Small Model to Think Like a Big One

This is what **fine-tuning** solves.

You take the small model (SmolVLM2-500M — the "Student") and have it learn from the big model (Qwen2-VL-7B — the "Teacher"). The big model generates perfect BLV descriptions. The small model practices imitating those descriptions thousands of times until it learns the pattern.

Then you add a second stage: the small model is shown pairs of descriptions — a good one and a bad one — and trained to always prefer producing the good kind. This is the **RL (Reinforcement Learning) based fine-tuning** your project title refers to.

After all this training, the small model is still small enough to run on a phone — but it now produces descriptions much closer in quality to what the big model would produce.

---

## What Your Team's Previous Paper Established

Your published paper (ACL MMLoSo 2025) answered the question: *"How bad is SmolVLM2 today at BLV descriptions, and how fast does it run on a phone?"*

It found:
- SmolVLM2 can run on a smartphone (both FP32 and INT8 precision)
- The description quality is poor — too vague, not spatially aware
- The team created two evaluation frameworks to measure BLV quality specifically

**This project's job:** Take those same models, fine-tune them with the Teacher-Student pipeline + DPO, and measure exactly how much better they get — while making sure they're still fast on the phone.

---

## Why This Matters Beyond The Classroom

Over 250 million people worldwide have severe visual impairment. Current smartphone accessibility tools (like Apple's VoiceOver or Android's TalkBack) describe UIs but are not good at describing real-world scenes in the level of detail BLV users actually need for navigation. A fine-tuned, phone-deployable model that follows BLV guidelines is genuinely useful technology — not just an academic exercise.

---

## Quick Glossary

| Term | What It Means |
|------|---------------|
| BLV | Blind and Low Vision — people with severe visual impairment |
| VLM | Vision-Language Model — AI that can understand both video/images and text |
| Fine-tuning | Taking a pre-trained model and specializing it for a specific task |
| SFT | Supervised Fine-Tuning — learning by imitating labeled examples |
| DPO | Direct Preference Optimization — learning from "good vs. bad" pairs |
| LoRA | Low-Rank Adaptation — memory-efficient fine-tuning technique |
| GGUF | File format for running LLMs on phone CPUs/NPUs |
| VRAM | Video RAM — memory on a GPU |
| Parameters | The "knowledge knobs" inside an AI — more = smarter but heavier |
| Keyframes | The most visually important frames extracted from a video |
| RAG | Retrieval-Augmented Generation — AI looks up context before answering |
| LUV | A color space that matches human color perception (used for keyframe selection) |
| Ablation | Running experiments with components removed to prove each one helps |
| CIDEr/BLEU | Standard metrics for evaluating text generation quality |

---

*See also: [[02_LUV_KEYFRAME_EXTRACTION]] | [[03_RAG_CONTEXT]] | [[05_SFT_TRAINING]] | [[06_DPO_TRAINING]]*
