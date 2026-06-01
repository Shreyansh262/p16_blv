#!/usr/bin/env python3
from datasets import load_dataset
ds = load_dataset("echo840/OCRBench", split="test")
ds.save_to_disk("./ocr_bench_test")
print(f"Downloaded {len(ds)} samples")
print(f"Features: {ds.features}")
