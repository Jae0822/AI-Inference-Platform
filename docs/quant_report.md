# Quantization Report — FP16 vs AWQ INT4 (Marlin kernel)

## Model: Qwen2.5-7B-Instruct

| Metric | FP16 | AWQ INT4 (Marlin) | Change |
|---|---|---|---|
| Model weights VRAM | ~14.0 GB | 5.29 GB | -62% |
| KV cache available | 2.65 GB | 13.93 GB | +426% |
| Tokens/sec (single user) | 60.6 | 103.95 | +71% |
| TTFT median (ms) | ~37 | TBD (re-run benchmark) | — |

## Source of numbers

FP16: M6 Locust benchmark + M7 analysis + vLLM startup logs.
AWQ: vLLM startup log (model loading 5.29 GiB, KV cache 13.93 GiB)
     + gateway log tokens_per_sec after awq_marlin kernel enabled.

## KV Cache Headroom

| | FP16 | AWQ INT4 (Marlin) |
|---|---|---|
| GPU VRAM total | 25.76 GB | 25.76 GB |
| Model weights | ~14 GB | 5.29 GB |
| KV cache available | ~2.65 GB | 13.93 GB |
| KV cache multiplier | 1x (baseline) | 5.3x |
| Est. saturation point | ~50-80 users | ~150+ users |

## Why VRAM total looks the same at runtime

vLLM pre-allocates gpu_memory_utilization (90%) of VRAM as a pool.
nvidia-smi shows ~22GB regardless of model size.
The meaningful split is internal:
  FP16:     14.0 GB weights + 2.65 GB KV cache
  AWQ INT4:  5.29 GB weights + 13.93 GB KV cache

## Performance comparison

| Metric | FP16 | AWQ INT4 (awq) | AWQ INT4 (awq_marlin) |
|---|---|---|---|
| Tokens/sec | 60.6 | ~19 | 103.95 |
| Kernel | — | standard AWQ | Marlin fused INT4 |

awq_marlin uses a fused INT4 GEMM kernel that is faster than FP16
because INT4 has 4x lower memory bandwidth requirement, and on RTX 4090 D
memory bandwidth is the primary bottleneck for 7B inference.

## Core result

AWQ INT4 with Marlin kernel delivers:
  - 62% VRAM reduction for model weights (14GB → 5.29GB)
  - 426% more KV cache space (2.65GB → 13.93GB)
  - 71% throughput improvement (60.6 → 103.95 tok/s)
  - Theoretical saturation point moves from ~50-80 to ~150+ concurrent users

## Resume Line

"Applied AWQ INT4 quantization to Qwen2.5-7B using llm-compressor +
vLLM awq_marlin kernel; reduced model VRAM 14GB → 5.29GB (-62%);
expanded KV cache 2.65GB → 13.93GB (+426%); single-user throughput
improved 60 → 104 tok/s (+71%) due to INT4 memory bandwidth advantage;
theoretical concurrent user capacity increased from ~50-80 to ~150+."
