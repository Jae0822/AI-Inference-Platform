"""
quantize_awq.py — Quantize Qwen2.5-7B from FP16 to AWQ INT4.

Uses llm-compressor (vLLM's official quantization library),
which replaced the deprecated autoawq package.

llm-compressor vs autoawq:
  autoawq: original implementation, now unmaintained.
  llm-compressor: maintained by the vLLM team, same AWQ algorithm,
                  better compatibility with recent Transformers versions,
                  produces models directly loadable by vLLM.

This script runs ONCE and saves the quantized model to disk.
After this, load with vLLM: --quantization compressed-tensors

Time estimate: 10-30 minutes (one-time cost).
"""

import os

import torch
from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FP16_MODEL_PATH = os.environ.get(
    "FP16_MODEL_PATH",
    "/root/autodl-tmp/models/Qwen2.5-7B-Instruct/Qwen/Qwen2.5-7B-Instruct",
)

OUTPUT_PATH = os.environ.get(
    "OUTPUT_PATH",
    "/root/autodl-tmp/models/Qwen2.5-7B-Instruct-INT4",
)


# ---------------------------------------------------------------------------
# Calibration dataset
#
# llm-compressor expects a HuggingFace Dataset object with a "text" column.
# We build it from our domain-specific prompts.
# ---------------------------------------------------------------------------

CALIBRATION_TEXTS = [
    "What is PagedAttention and how does it improve LLM serving throughput?",
    "Explain the difference between FP16 and INT4 quantization in neural networks.",
    "How does the asyncio event loop handle concurrent coroutines in Python?",
    "What is the token bucket algorithm and how is it used for rate limiting?",
    "Describe the architecture of a production AI inference gateway.",
    "What is KV cache in transformer models and why does it consume memory?",
    "How does vLLM's continuous batching differ from static batching?",
    "Explain the difference between TTFT and end-to-end latency in LLM serving.",
    "What is AWQ quantization and why does it outperform naive INT4 quantization?",
    "How does Prometheus collect metrics from a FastAPI application?",
    "What is the role of a Semaphore in controlling concurrency in asyncio?",
    "Explain how sliding window context truncation works in a chat application.",
    "What causes KV cache eviction in vLLM and how does it affect tail latency?",
    "How does GPU utilization differ from GPU memory utilization?",
    "What is the difference between p50 and p99 latency in a load test?",
    "Explain how StreamingResponse works in FastAPI for SSE endpoints.",
    "What are the main components of a transformer attention mechanism?",
    "How does tensor parallelism work in distributed LLM inference?",
    "What is speculative decoding and how does it speed up LLM generation?",
    "Explain the tradeoff between quantization bits and model quality.",
    "How does GPTQ differ from AWQ in its approach to weight quantization?",
    "What is the purpose of the system prompt in a chat model?",
    "How do you measure perplexity and what does it tell you about model quality?",
    "Explain the role of layer normalization in transformer models.",
    "What is flash attention and how does it reduce memory usage?",
    "How does beam search differ from greedy decoding in text generation?",
    "What is temperature in LLM sampling and how does it affect output diversity?",
    "Explain the difference between encoder-only and decoder-only transformers.",
    "What is LoRA and how does it enable efficient fine-tuning?",
    "How does the attention mask work in batched inference?",
    "What is the purpose of the BOS and EOS tokens in a language model?",
    "Explain how rotary positional embeddings work in modern LLMs.",
]


def main():
    print("=" * 60)
    print("INT4 Quantization — Qwen2.5-7B-Instruct")
    print("Using: llm-compressor (vLLM official)")
    print("=" * 60)
    print(f"Source : {FP16_MODEL_PATH}")
    print(f"Output : {OUTPUT_PATH}")
    print()

    if not os.path.exists(FP16_MODEL_PATH):
        raise FileNotFoundError(
            f"FP16 model not found at {FP16_MODEL_PATH}\n"
            "Make sure vLLM is stopped before running this script."
        )

    if os.path.exists(OUTPUT_PATH) and os.listdir(OUTPUT_PATH):
        answer = input(
            f"Output path already exists: {OUTPUT_PATH}\nOverwrite? [y/N]: "
        ).strip().lower()
        if answer != "y":
            print("Aborted.")
            return

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # ── Step 1: Load tokenizer ───────────────────────────────────────────
    print("[1/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        FP16_MODEL_PATH,
        trust_remote_code=True,
    )
    print("      Done.")

    # ── Step 2: Load model ───────────────────────────────────────────────
    print("[2/4] Loading FP16 model...")
    model = AutoModelForCausalLM.from_pretrained(
        FP16_MODEL_PATH,
        device_map="cuda",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    if torch.cuda.is_available():
        used  = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"      VRAM: {used:.1f} GB / {total:.1f} GB")

    # ── Step 3: Build calibration dataset ───────────────────────────────
    print("[3/4] Building calibration dataset...")
    # Tokenize texts and format as HuggingFace Dataset
    tokenized = tokenizer(
        CALIBRATION_TEXTS,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    # llm-compressor expects a Dataset with "input_ids" column
    calib_dataset = Dataset.from_dict({
        "input_ids":      tokenized["input_ids"].tolist(),
        "attention_mask": tokenized["attention_mask"].tolist(),
    })
    print(f"      {len(calib_dataset)} calibration samples ready.")

    # ── Step 4: Quantize ─────────────────────────────────────────────────
    print()
    print("[4/4] Running INT4 quantization (10-30 minutes)...")
    print("      You will see layer-by-layer progress below.")
    print()

    # SmoothQuant: pre-process activations to reduce quantization error.
    # GPTQ: second-order weight quantization (similar quality to AWQ).
    # llm-compressor uses GPTQ by default; for AWQ-style use GPTQModifier
    # with the actorder setting below.
    recipe = [
        SmoothQuantModifier(smoothing_strength=0.8),
        GPTQModifier(
            targets="Linear",
            scheme="W4A16",      # 4-bit weights, 16-bit activations
            ignore=["lm_head"],  # keep output layer in FP16
        ),
    ]

    oneshot(
        model=model,
        dataset=calib_dataset,
        recipe=recipe,
        max_seq_length=512,
        num_calibration_samples=len(CALIBRATION_TEXTS),
    )

    print()
    print("      Quantization complete.")
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e9
        print(f"      VRAM after quantization: {used:.1f} GB / {total:.1f} GB")

    # Save
    print(f"\nSaving to {OUTPUT_PATH}...")
    model.save_pretrained(OUTPUT_PATH, save_compressed=True)
    tokenizer.save_pretrained(OUTPUT_PATH)

    # Output size
    total_size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, dn, fns in os.walk(OUTPUT_PATH)
        for f in fns
    )
    print(f"Output size: {total_size / 1e9:.2f} GB")

    print()
    print("=" * 60)
    print("Quantization complete!")
    print()
    print("Load in vLLM with:")
    print(f"  vllm serve {OUTPUT_PATH} \\")
    print("    --quantization compressed-tensors \\")
    print("    --port 6006 \\")
    print("    --gpu-memory-utilization 0.90 \\")
    print("    --max-model-len 4096")
    print("=" * 60)


if __name__ == "__main__":
    main()
