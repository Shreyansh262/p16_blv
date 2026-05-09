#!/usr/bin/env bash
# GGUF conversion pipeline for SFT v2 merged model.
# Run inside tmux after 11_merge_lora.py finishes.
# Output: models/gguf/sft_v2_q4km.gguf + models/gguf/sft_v2_mmproj.gguf
#
# NOTE: blv env torchvision is broken (circular import). The LM conversion
# uses src/deployment/gguf_convert_wrapper.py to mock torchvision before
# any import can trigger it. The mmproj path does NOT need the wrapper.

set -e
cd ~/p16_blv

MERGED="models/student/sft_v2_merged"
GGUF_DIR="models/gguf"
LLAMA_DIR="$HOME/p16_blv/tools/llama_cpp"
CONV_PYTHON="/usershome/cs671_user2/miniconda3/envs/blv/bin/python"
WRAPPER="src/deployment/gguf_convert_wrapper.py"

mkdir -p "$GGUF_DIR"

# ── Step 1: Build llama.cpp (CPU, skips if already built) ────────────────────
if [ ! -f "$LLAMA_DIR/build/bin/llama-quantize" ]; then
    echo "=== Building llama.cpp (CPU) ==="
    cd "$LLAMA_DIR"
    cmake -B build -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
    cmake --build build --config Release -j4
    cd ~/p16_blv
    echo "=== llama.cpp build complete ==="
else
    echo "[skip] llama.cpp already built"
fi

# ── Step 2: Convert HF model to F16 GGUF (LM backbone) ──────────────────────
# Uses wrapper to mock broken torchvision before transformers imports it.
# Architecture: llama (VLlama3ForCausalLM handler = SmolVLM2 LM backbone).
# Chat template includes <image> token for multimodal pairing with mmproj.
F16_GGUF="$GGUF_DIR/sft_v2_f16.gguf"
if [ ! -f "$F16_GGUF" ]; then
    echo "=== Converting HF model to GGUF F16 (LM) ==="
    "$CONV_PYTHON" "$WRAPPER" \
        "$LLAMA_DIR/convert_hf_to_gguf.py" \
        "$MERGED" \
        --outtype f16 \
        --outfile "$F16_GGUF"
    echo "F16 GGUF written: $F16_GGUF"
else
    echo "[skip] $F16_GGUF already exists"
fi

# ── Step 3: Generate mmproj (vision encoder + projector) ─────────────────────
# No wrapper needed — mmproj path uses SmolVLMModel handler which does not
# trigger the torchvision → LlamaConfig import chain.
MMPROJ_GGUF="$GGUF_DIR/sft_v2_mmproj.gguf"
if [ ! -f "$MMPROJ_GGUF" ]; then
    echo "=== Generating mmproj GGUF (vision encoder) ==="
    "$CONV_PYTHON" "$LLAMA_DIR/convert_hf_to_gguf.py" \
        "$MERGED" \
        --mmproj \
        --outtype f16 \
        --outfile "$MMPROJ_GGUF"
    echo "mmproj GGUF written: $MMPROJ_GGUF"
else
    echo "[skip] $MMPROJ_GGUF already exists"
fi

# ── Step 4: Quantize LM to Q4_K_M ────────────────────────────────────────────
Q4_GGUF="$GGUF_DIR/sft_v2_q4km.gguf"
if [ ! -f "$Q4_GGUF" ]; then
    echo "=== Quantizing to Q4_K_M ==="
    "$LLAMA_DIR/build/bin/llama-quantize" "$F16_GGUF" "$Q4_GGUF" Q4_K_M
    echo "Q4_K_M GGUF written: $Q4_GGUF"
else
    echo "[skip] $Q4_GGUF already exists"
fi

# ── Step 5: Latency benchmark ─────────────────────────────────────────────────
echo ""
echo "=== Latency Benchmark (llama-bench, CPU, 3 runs) ==="
"$LLAMA_DIR/build/bin/llama-bench" \
    -m "$Q4_GGUF" \
    -p 512 \
    -n 128 \
    -r 3 \
    2>&1 | tee "logs/deployment/bench_cpu.txt"

echo ""
echo "=== File sizes ==="
ls -lh "$GGUF_DIR/"
echo ""
echo "PIPELINE COMPLETE. Files in $GGUF_DIR/"
echo "Deploy: sft_v2_q4km.gguf (LM) + sft_v2_mmproj.gguf (vision) → Android /sdcard/models/"
