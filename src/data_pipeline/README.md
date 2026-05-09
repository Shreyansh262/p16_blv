# src/data_pipeline/
Sequential data preparation pipeline scripts.

| File | Purpose |
|------|---------|
| 01_extract_keyframes.py | Extract keyframes from videos |
| 02_build_rag_db.py | Build RAG retrieval database |
| 03_generate_gemma_captions.py | Generate captions via Gemma |
| 03_generate_teacher_captions.py | Generate captions via teacher model |
| 04_build_dpo_pairs.py | Build preference pairs for DPO |
| filter_captions.py | Filter captions by BLV score |
| generate_distill_traces.py | Generate distillation traces |
