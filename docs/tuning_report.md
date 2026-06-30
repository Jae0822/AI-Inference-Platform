# vLLM Serving Parameter Tuning Report — M11

## Setup

| Item | Value |
|---|---|
| Model | Qwen2.5-7B-Instruct AWQ INT4 |
| Kernel | awq_marlin |
| Concurrency tested | 50 users (saturation zone from M6) |
| Run time per config | 60 seconds |
| Baseline (M6 FP16, 50 users) | TTFT p99=219ms, total p99=1335ms, TPS p50=53.8 |

## Why 50 concurrent users?

M6 showed FP16 saturated between 50-80 users (TTFT p99 jumped from
94ms at 10 users to 219ms at 50 users). 50 users is the stress point
where FP16 started degrading. Testing AWQ at this same load shows
whether the larger KV cache and higher TPS change the picture.

---

## Results

### Config A — max-num-seqs=64

| Metric | Value |
|---|---|
| TTFT p50 (ms) | 24 |
| TTFT p95 (ms) | 503 |
| TTFT p99 (ms) | 506 |
| Total p50 (ms) | 335 |
| Total p99 (ms) | 852 |
| TPS p50 (tok/s) | 157.3 |
| Total requests (60s) | 128 |

### Config B — max-num-seqs=128

| Metric | Value |
|---|---|
| TTFT p50 (ms) | 24 |
| TTFT p95 (ms) | 511 |
| TTFT p99 (ms) | 514 |
| Total p50 (ms) | 339 |
| Total p99 (ms) | 861 |
| TPS p50 (tok/s) | 157.3 |
| Total requests (60s) | 129 |

### Config C — max-num-seqs=256

| Metric | Value |
|---|---|
| TTFT p50 (ms) | 24 |
| TTFT p95 (ms) | 236 |
| TTFT p99 (ms) | 240 |
| Total p50 (ms) | 336 |
| Total p99 (ms) | 585 |
| TPS p50 (tok/s) | 157.1 |
| Total requests (60s) | 129 |

---

## Comparison with M6 FP16 baseline (50 users)

| Config | TTFT p99 (ms) | Total p99 (ms) | TPS p50 | vs FP16 TTFT p99 | vs FP16 Total p99 |
|---|---|---|---|---|---|
| FP16 baseline (M6) | 219 | 1335 | 53.8 | — | — |
| AWQ seqs=64 | 506 | 852 | 157.3 | +131% worse | -36% better |
| AWQ seqs=128 | 514 | 861 | 157.3 | +135% worse | -35% better |
| AWQ seqs=256 | 240 | 585 | 157.1 | +10% (comparable) | -56% better |

---

## Analysis

**Winning configuration: Config C (max-num-seqs=256)**

**Why Config C wins:**
max-num-seqs=64 creates a hidden queue inside vLLM. At 50 concurrent
users, 50 sequences attempt to enter the scheduler simultaneously.
With max-num-seqs=64, most fit — but the last few must wait in the
vLLM internal queue, adding 480ms of invisible wait time to TTFT p99
(from 24ms p50 to 506ms p99, a 21x gap).

With max-num-seqs=256, all 50 concurrent sequences enter the scheduler
immediately. The TTFT p99 drops to 240ms with only a 10x gap from p50
to p99 — still some scheduling variance, but no queue-induced spike.

**Why TPS is identical across all configs (157 tok/s):**
TPS is determined by GPU compute speed and memory bandwidth.
AWQ INT4 with the Marlin kernel saturates the RTX 4090 D compute
at ~157 tok/s regardless of how many sequences are queued.
max-num-seqs affects scheduling fairness, not peak throughput.

**TTFT p50=24ms across all configs:**
The median request (fast path) is unaffected by max-num-seqs.
Only the tail (p95/p99) reveals the queuing effect.
This is why p50 metrics alone are insufficient for production SLOs.

**Recommended production configuration:**
  --quantization awq_marlin
  --max-num-seqs 256
  --gpu-memory-utilization 0.90
  --max-model-len 4096

---

## Full picture: FP16 → AWQ + tuning at 50 concurrent users

| Metric | FP16 (M6) | AWQ + seqs=256 (M11) | Improvement |
|---|---|---|---|
| TTFT p99 (ms) | 219 | 240 | comparable (+10%) |
| Total p99 (ms) | 1335 | 585 | -56% |
| TPS p50 (tok/s) | 53.8 | 157.1 | +192% |
| Model VRAM (GB) | ~14 | 5.29 | -62% |
| KV cache (GB) | 2.65 | 13.93 | +426% |

The system serves 50 concurrent users at 157 tok/s (vs 53.8 FP16)
with 56% lower tail latency, using 62% less VRAM for model weights.

---

## Resume Line

"Post-quantization parameter tuning on vLLM: identified max-num-seqs=256
as optimal for AWQ INT4 model at 50 concurrent users; compared to FP16
baseline — end-to-end p99 latency improved 56% (1335ms → 585ms),
throughput increased 192% (53.8 → 157 tok/s), while maintaining
comparable TTFT p99 (219ms → 240ms); model VRAM reduced 62% freeing
13.93GB for KV cache vs 2.65GB previously."
