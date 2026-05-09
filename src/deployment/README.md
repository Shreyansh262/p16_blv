# src/deployment/
Model export and deployment scripts.

| File | Purpose |
|------|---------|
| 11_merge_lora.py | Merge LoRA adapter into base |
| 12_gguf_pipeline.sh | Convert model to GGUF format |
| 13_ollama_bench.sh | Benchmark via Ollama |
| gguf_convert_wrapper.py | Wrapper for GGUF conversion |
| patch_gguf_vision.py | Patch vision projector in GGUF |
