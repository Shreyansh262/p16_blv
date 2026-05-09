#!/usr/bin/env bash
# Ollama end-to-end latency benchmark for sft_v2 Q4_K_M GGUF.
# Run after llama-bench completes.
# All files stay inside ~/p16_blv/

set -e
cd ~/p16_blv

OLLAMA="$HOME/p16_blv/bin/ollama"
GGUF_DIR="$HOME/p16_blv/models/gguf"
MODEL_NAME="p16-smolvlm2-sftv2"
OUT="$HOME/p16_blv/logs/deployment/bench_ollama.txt"

# Write Modelfile
cat > "$GGUF_DIR/Modelfile" << 'EOF'
FROM ./sft_v2_q4km.gguf
PARAMETER num_ctx 2048
PARAMETER temperature 0.1
SYSTEM "You are an audio description system for blind and low vision users. Describe video scenes concisely for navigation."
EOF

echo "=== Creating Ollama model: $MODEL_NAME ===" | tee "$OUT"
cd "$GGUF_DIR"
"$OLLAMA" create "$MODEL_NAME" -f Modelfile 2>&1 | tee -a "$OUT"
cd ~/p16_blv

PROMPT="A person walks down an indoor corridor. Describe this scene for a blind user in 2-3 sentences."

echo "" | tee -a "$OUT"
echo "=== Text-only latency (10 runs) ===" | tee -a "$OUT"

TOTAL=0
for i in $(seq 1 10); do
    START=$(date +%s%3N)
    "$OLLAMA" run "$MODEL_NAME" "$PROMPT" > /dev/null 2>&1
    END=$(date +%s%3N)
    MS=$((END - START))
    echo "Run $i: ${MS}ms" | tee -a "$OUT"
    TOTAL=$((TOTAL + MS))
done

AVG=$((TOTAL / 10))
echo "" | tee -a "$OUT"
echo "Average latency: ${AVG}ms" | tee -a "$OUT"
echo "Model size (Q4_K_M): $(du -h $GGUF_DIR/sft_v2_q4km.gguf | cut -f1)" | tee -a "$OUT"
echo "Done. Results in $OUT"
