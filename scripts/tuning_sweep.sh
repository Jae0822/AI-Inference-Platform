#!/bin/bash
# tuning/sweep.sh — Systematic vLLM parameter sweep for AWQ model.
#
# What this script does:
#   For each max-num-seqs value in [64, 128, 256]:
#     1. Print the configuration being tested
#     2. Run Locust at 50 concurrent users for 60 seconds
#     3. Parse and display the Quick Summary
#     4. Wait 30 seconds for system cooldown
#
# Why we only test 50 concurrent users (not 1/10/50/100 like M6)?
#   We already have the full M6 baseline at all concurrency levels.
#   The question for M11 is narrower: "at the concurrency level where
#   FP16 started degrading (50 users), does a higher max-num-seqs
#   with AWQ produce better results?"
#   Testing only 50 users keeps the sweep to ~15 minutes.
#
# Prerequisites:
#   - vLLM running with AWQ model on port 6006 (--quantization awq_marlin)
#   - gateway.py running on port 8000
#   - locust installed

set -e

GATEWAY_URL="http://localhost:8000"
LOCUSTFILE="/root/autodl-tmp/locustfile.py"
RESULTS_DIR="/root/autodl-tmp/tuning/results"
CONCURRENT_USERS=50
SPAWN_RATE=10
RUN_TIME="60s"
COOLDOWN=30

mkdir -p "$RESULTS_DIR"

echo "=================================================="
echo "M11 vLLM Parameter Sweep — AWQ INT4 (Marlin)"
echo "=================================================="
echo "Concurrent users : $CONCURRENT_USERS"
echo "Run time         : $RUN_TIME"
echo "Cooldown         : ${COOLDOWN}s between runs"
echo ""

# Verify gateway is up
if ! curl -s "$GATEWAY_URL/health" > /dev/null 2>&1; then
    echo "ERROR: Gateway not responding at $GATEWAY_URL"
    echo "Make sure gateway.py is running."
    exit 1
fi
echo "Gateway: OK"
echo ""

# ---------------------------------------------------------------------------
# Run one Locust benchmark and capture Quick Summary
# ---------------------------------------------------------------------------
run_locust() {
    local label="$1"
    local csv_prefix="$RESULTS_DIR/${label}"

    echo "--------------------------------------------------"
    echo "Running: $label"
    echo "--------------------------------------------------"

    locust -f "$LOCUSTFILE" \
        --headless \
        --host "$GATEWAY_URL" \
        -u "$CONCURRENT_USERS" \
        -r "$SPAWN_RATE" \
        --run-time "$RUN_TIME" \
        --csv "$csv_prefix" \
        2>&1 | grep -E "Quick Summary|Total requests|TTFT|Total p|TPS|==="

    echo ""
}

# ---------------------------------------------------------------------------
# The three configurations to test
#
# Note: max-num-seqs is a vLLM server parameter, NOT a gateway parameter.
# You must restart vLLM with a different --max-num-seqs for each run.
# This script handles the Locust benchmark part only.
# See instructions below for restarting vLLM between runs.
# ---------------------------------------------------------------------------

echo "=================================================="
echo "IMPORTANT: This script runs the Locust benchmarks."
echo "You must manually restart vLLM with the correct"
echo "--max-num-seqs before each run."
echo ""
echo "Run order:"
echo "  Step 1: Start vLLM with --max-num-seqs 64,  then press Enter"
echo "  Step 2: Start vLLM with --max-num-seqs 128, then press Enter"
echo "  Step 3: Start vLLM with --max-num-seqs 256, then press Enter"
echo "=================================================="
echo ""

# ---------------------------------------------------------------------------
# Config A: max-num-seqs=64
# ---------------------------------------------------------------------------
echo "Step 1: Start vLLM with --max-num-seqs 64 on port 6006."
echo "Command:"
echo "  vllm serve /root/autodl-tmp/models/Qwen2.5-7B-Instruct-AWQ/qwen/Qwen2___5-7B-Instruct-AWQ \\"
echo "    --quantization awq_marlin \\"
echo "    --port 6006 \\"
echo "    --gpu-memory-utilization 0.90 \\"
echo "    --max-model-len 4096 \\"
echo "    --max-num-seqs 64"
echo ""
read -p "Press Enter when vLLM is ready (Uvicorn running on port 6006)..."
echo ""
run_locust "config_A_seqs64_u50"
echo "Cooling down ${COOLDOWN}s..."
sleep "$COOLDOWN"

# ---------------------------------------------------------------------------
# Config B: max-num-seqs=128
# ---------------------------------------------------------------------------
echo "Step 2: Restart vLLM with --max-num-seqs 128."
echo "Command:"
echo "  vllm serve /root/autodl-tmp/models/Qwen2.5-7B-Instruct-AWQ/qwen/Qwen2___5-7B-Instruct-AWQ \\"
echo "    --quantization awq_marlin \\"
echo "    --port 6006 \\"
echo "    --gpu-memory-utilization 0.90 \\"
echo "    --max-model-len 4096 \\"
echo "    --max-num-seqs 128"
echo ""
read -p "Press Enter when vLLM is ready..."
echo ""
run_locust "config_B_seqs128_u50"
echo "Cooling down ${COOLDOWN}s..."
sleep "$COOLDOWN"

# ---------------------------------------------------------------------------
# Config C: max-num-seqs=256
# ---------------------------------------------------------------------------
echo "Step 3: Restart vLLM with --max-num-seqs 256."
echo "Command:"
echo "  vllm serve /root/autodl-tmp/models/Qwen2.5-7B-Instruct-AWQ/qwen/Qwen2___5-7B-Instruct-AWQ \\"
echo "    --quantization awq_marlin \\"
echo "    --port 6006 \\"
echo "    --gpu-memory-utilization 0.90 \\"
echo "    --max-model-len 4096 \\"
echo "    --max-num-seqs 256"
echo ""
read -p "Press Enter when vLLM is ready..."
echo ""
run_locust "config_C_seqs256_u50"

echo ""
echo "=================================================="
echo "All three configurations complete."
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "Summary files:"
ls "$RESULTS_DIR"/*.csv 2>/dev/null || echo "(no CSV files found)"
echo ""
echo "Next: fill in tuning_report.md with the Quick Summary numbers."
echo "=================================================="
