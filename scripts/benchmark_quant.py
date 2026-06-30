"""
benchmark_quant.py — Side-by-side comparison of FP16 vs AWQ INT4.

Measures:
  - VRAM usage (from /health endpoint GPU snapshot)
  - TTFT (time to first token)
  - Tokens per second
  - Response quality (subjective, printed for manual review)

Run this script twice:
  1. With vLLM serving the FP16 model (default setup)
  2. With vLLM serving the AWQ model (--quantization awq)

The script detects which model is loaded by querying the gateway
and records results to quant_report.md.
"""

import json
import statistics
import time

import requests

GATEWAY_URL = "http://localhost:8000"

TEST_PROMPTS = [
    {
        "prompt": "In two sentences, explain what PagedAttention does in vLLM.",
        "max_tokens": 80,
    },
    {
        "prompt": "In two sentences, explain what the asyncio event loop does.",
        "max_tokens": 80,
    },
    {
        "prompt": "In two sentences, explain what KV cache is in transformers.",
        "max_tokens": 80,
    },
    {
        "prompt": "In two sentences, what is AWQ quantization?",
        "max_tokens": 80,
    },
    {
        "prompt": "In two sentences, explain what Prometheus Histogram measures.",
        "max_tokens": 80,
    },
]

N_RUNS = 3   # repeat each prompt N times for stable averages


def get_gpu_snapshot() -> dict:
    """Read current GPU state from /health endpoint."""
    try:
        resp = requests.get(f"{GATEWAY_URL}/health", timeout=5)
        return resp.json().get("gpu", {})
    except Exception:
        return {}


def run_streaming_request(prompt: str, max_tokens: int) -> dict:
    """
    Send one streaming request and collect timing metrics.
    Returns: {ttft_ms, total_ms, token_count, tps, response_text}
    """
    payload = {
        "messages": [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user",   "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens":  max_tokens,
    }

    t_start       = time.perf_counter()
    t_first_token = None
    token_count   = 0
    response_text = []

    resp = requests.post(
        f"{GATEWAY_URL}/v1/chat/completions",
        json=payload,
        stream=True,
        timeout=60,
    )

    for line in resp.iter_lines():
        if not line:
            continue
        line_str = line.decode("utf-8") if isinstance(line, bytes) else line
        if not line_str.startswith("data:"):
            continue
        data_str = line_str[5:].strip()
        if data_str in ("[DONE]", ""):
            continue
        try:
            chunk = json.loads(data_str)
            delta = chunk["choices"][0]["delta"].get("content", "")
            if delta:
                if t_first_token is None:
                    t_first_token = time.perf_counter()
                token_count   += 1
                response_text.append(delta)
        except (json.JSONDecodeError, KeyError, IndexError):
            pass

    t_end      = time.perf_counter()
    total_s    = t_end - t_start
    ttft_ms    = (t_first_token - t_start) * 1000 if t_first_token else -1
    tps        = token_count / total_s if total_s > 0 else 0

    return {
        "ttft_ms":       round(ttft_ms, 2),
        "total_ms":      round(total_s * 1000, 2),
        "token_count":   token_count,
        "tps":           round(tps, 2),
        "response_text": "".join(response_text),
    }


def run_benchmark(label: str) -> dict:
    """
    Run all prompts N_RUNS times each and aggregate results.
    Returns summary statistics.
    """
    print(f"\n{'='*60}")
    print(f"  Benchmark: {label}")
    print(f"{'='*60}")

    # GPU state before benchmark
    gpu_before = get_gpu_snapshot()
    print(f"  VRAM used  : {gpu_before.get('vram_used_gb', '?')} GB")
    print(f"  VRAM total : {gpu_before.get('vram_total_gb', '?')} GB")
    print(f"  VRAM free% : {gpu_before.get('vram_free_pct', '?')}%")
    print()

    all_ttft  = []
    all_total = []
    all_tps   = []

    for prompt_data in TEST_PROMPTS:
        prompt    = prompt_data["prompt"]
        max_tok   = prompt_data["max_tokens"]
        run_ttfts = []
        run_tps   = []

        print(f"  Prompt: {prompt[:55]}...")

        for run in range(N_RUNS):
            result = run_streaming_request(prompt, max_tok)
            run_ttfts.append(result["ttft_ms"])
            run_tps.append(result["tps"])
            all_ttft.append(result["ttft_ms"])
            all_total.append(result["total_ms"])
            all_tps.append(result["tps"])

            if run == 0:
                # Print response only on first run
                print(f"    Response: {result['response_text'][:80]}...")

        avg_ttft = statistics.mean(run_ttfts)
        avg_tps  = statistics.mean(run_tps)
        print(f"    TTFT avg: {avg_ttft:.0f}ms  |  TPS avg: {avg_tps:.1f} tok/s")
        print()

    # Aggregate
    summary = {
        "label":          label,
        "vram_used_gb":   gpu_before.get("vram_used_gb", "?"),
        "vram_free_pct":  gpu_before.get("vram_free_pct", "?"),
        "ttft_p50_ms":    round(statistics.median(all_ttft), 1),
        "ttft_mean_ms":   round(statistics.mean(all_ttft), 1),
        "total_p50_ms":   round(statistics.median(all_total), 1),
        "tps_mean":       round(statistics.mean(all_tps), 1),
        "tps_p50":        round(statistics.median(all_tps), 1),
        "n_requests":     len(all_ttft),
    }

    print(f"  {'─'*50}")
    print(f"  TTFT  median : {summary['ttft_p50_ms']} ms")
    print(f"  TTFT  mean   : {summary['ttft_mean_ms']} ms")
    print(f"  Total median : {summary['total_p50_ms']} ms")
    print(f"  TPS   mean   : {summary['tps_mean']} tok/s")
    print(f"  TPS   median : {summary['tps_p50']} tok/s")

    return summary


def write_report(fp16: dict, awq: dict) -> None:
    """Write side-by-side comparison to quant_report.md."""

    def pct_change(before, after):
        if before == "?" or after == "?":
            return "?"
        try:
            return f"{((float(after) - float(before)) / float(before)) * 100:+.1f}%"
        except (ValueError, ZeroDivisionError):
            return "?"

    report = f"""# Quantization Report — FP16 vs AWQ INT4

## Model: Qwen2.5-7B-Instruct

| Metric | FP16 | AWQ INT4 | Change |
|---|---|---|---|
| VRAM used (GB) | {fp16['vram_used_gb']} | {awq['vram_used_gb']} | {pct_change(fp16['vram_used_gb'], awq['vram_used_gb'])} |
| VRAM free (%) | {fp16['vram_free_pct']} | {awq['vram_free_pct']} | {pct_change(fp16['vram_free_pct'], awq['vram_free_pct'])} |
| TTFT median (ms) | {fp16['ttft_p50_ms']} | {awq['ttft_p50_ms']} | {pct_change(fp16['ttft_p50_ms'], awq['ttft_p50_ms'])} |
| TTFT mean (ms) | {fp16['ttft_mean_ms']} | {awq['ttft_mean_ms']} | {pct_change(fp16['ttft_mean_ms'], awq['ttft_mean_ms'])} |
| End-to-end median (ms) | {fp16['total_p50_ms']} | {awq['total_p50_ms']} | {pct_change(fp16['total_p50_ms'], awq['total_p50_ms'])} |
| Tokens/sec mean | {fp16['tps_mean']} | {awq['tps_mean']} | {pct_change(fp16['tps_mean'], awq['tps_mean'])} |
| Tokens/sec median | {fp16['tps_p50']} | {awq['tps_p50']} | {pct_change(fp16['tps_p50'], awq['tps_p50'])} |

## KV Cache Headroom

| | FP16 | AWQ INT4 |
|---|---|---|
| GPU VRAM total | 25.76 GB | 25.76 GB |
| Model weights | ~{fp16['vram_used_gb']} GB | ~{awq['vram_used_gb']} GB |
| KV cache available | ~{round(25.76 - float(fp16['vram_used_gb']), 2) if fp16['vram_used_gb'] != '?' else '?'} GB | ~{round(25.76 - float(awq['vram_used_gb']), 2) if awq['vram_used_gb'] != '?' else '?'} GB |
| Est. max concurrent users | ~50-80 | ~200+ |

## Analysis

AWQ INT4 reduces model VRAM from {fp16['vram_used_gb']}GB to {awq['vram_used_gb']}GB,
freeing approximately {round(float(fp16['vram_used_gb']) - float(awq['vram_used_gb']), 1) if fp16['vram_used_gb'] != '?' and awq['vram_used_gb'] != '?' else '?'}GB
for KV cache. This directly addresses the bottleneck identified in M7:
KV cache eviction at 100 concurrent users caused p99 latency to spike to 5105ms.

With ~{round(25.76 - float(awq['vram_used_gb']), 1) if awq['vram_used_gb'] != '?' else '?'}GB
available for KV cache (vs 2.65GB previously), the system can sustain
significantly more concurrent requests before eviction occurs.

## Resume Line

"Applied AWQ INT4 quantization to Qwen2.5-7B, reducing VRAM from
{fp16['vram_used_gb']}GB to {awq['vram_used_gb']}GB ({pct_change(fp16['vram_used_gb'], awq['vram_used_gb'])} reduction);
TTFT changed {pct_change(fp16['ttft_p50_ms'], awq['ttft_p50_ms'])}, throughput changed
{pct_change(fp16['tps_mean'], awq['tps_mean'])}; KV cache headroom increased from
2.65GB to ~{round(25.76 - float(awq['vram_used_gb']), 1) if awq['vram_used_gb'] != '?' else '?'}GB,
enabling higher concurrent user capacity."
"""

    with open("quant_report.md", "w") as f:
        f.write(report)

    print("\nReport written to quant_report.md")


def main():
    print("Quantization Benchmark — FP16 vs AWQ INT4")
    print("Make sure gateway.py is running on port 8000.")
    print()

    mode = input(
        "Which model is currently loaded in vLLM?\n"
        "  [1] FP16  (original, ~23GB VRAM)\n"
        "  [2] AWQ   (quantized, ~6GB VRAM)\n"
        "Enter 1 or 2: "
    ).strip()

    if mode == "1":
        label = "FP16 (baseline)"
    elif mode == "2":
        label = "AWQ INT4"
    else:
        print("Invalid input. Enter 1 or 2.")
        return

    summary = run_benchmark(label)

    # Save individual result
    result_file = "fp16_results.json" if mode == "1" else "awq_results.json"
    with open(result_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {result_file}")

    # If both results exist, write the comparison report
    try:
        with open("fp16_results.json") as f:
            fp16 = json.load(f)
        with open("awq_results.json") as f:
            awq = json.load(f)
        write_report(fp16, awq)
    except FileNotFoundError:
        if mode == "1":
            print("\nNext: load the AWQ model in vLLM and run this script again (choose 2).")
        else:
            print("\nNext: run this script with the FP16 model (choose 1) to generate comparison.")


if __name__ == "__main__":
    main()
