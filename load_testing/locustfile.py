"""
locustfile.py — Streaming LLM load test for the AI inference gateway.

Why streaming load testing is different from REST load testing:
  In a standard REST test, each user sends a request and immediately
  gets a complete response. The connection is short-lived.

  In a streaming LLM test, each user sends a request and then holds
  the connection open for the entire duration of the response — which
  can be 3–10 seconds. This means:
    - 50 concurrent users = 50 simultaneous open TCP connections
    - 50 simultaneous vLLM inference tasks consuming GPU memory
    - The KV cache fills up proportionally to active connections
    - TTFT degrades as the GPU scheduler queues more requests

  This is why streaming load testing reveals bottlenecks that a
  standard REST test completely misses.

Test design:
  - Each Locust user sends one streaming request at a time
  - Waits for the full response before sending the next
  - Records: TTFT, total latency, tokens received, tokens/sec
  - Uses a fixed prompt set to make results comparable across runs

Concurrency levels to test:
  Run this script four times with different --users values:
    locust --headless -u 1   -r 1   --run-time 60s
    locust --headless -u 10  -r 10  --run-time 60s
    locust --headless -u 50  -r 10  --run-time 60s
    locust --headless -u 100 -r 10  --run-time 60s
"""

import json
import os
import time

from locust import HttpUser, between, events, task
from locust.runners import MasterRunner

# ---------------------------------------------------------------------------
# Fixed prompt set
#
# Why fixed prompts instead of random?
# Fixed prompts make results comparable across runs and concurrency levels.
# Random prompts introduce variable response length, making it impossible
# to tell whether a throughput drop is due to load or longer responses.
# All prompts are designed to produce ~80-120 token responses.
# ---------------------------------------------------------------------------

PROMPTS = [
    "In two sentences, explain what PagedAttention is in vLLM.",
    "In two sentences, explain what the asyncio event loop does.",
    "In two sentences, explain what a Prometheus Histogram measures.",
    "In two sentences, explain what KV cache is in transformer inference.",
    "In two sentences, explain what continuous batching means in LLM serving.",
]


# ---------------------------------------------------------------------------
# Results storage
# Collected per-request stats, written to CSV at test end.
# ---------------------------------------------------------------------------

_results: list[dict] = []


# ---------------------------------------------------------------------------
# Locust User
# ---------------------------------------------------------------------------

class StreamingLLMUser(HttpUser):
    """
    Simulates one user sending streaming chat requests to the gateway.

    wait_time = between(1, 3):
      After each response completes, the user waits 1-3 seconds before
      sending the next request. This models realistic human think time
      and prevents the test from being purely throughput-limited
      (which would not reflect real usage patterns).

      For a pure throughput test, set wait_time = constant(0).
      For M6 we use between(1, 3) to get a more realistic picture.
    """
    wait_time = between(1, 3)

    # Rotate through prompts so each user sends varied requests
    _prompt_index: int = 0

    def on_start(self):
        """Called once when each simulated user starts."""
        self._prompt_index = 0

    @task
    def streaming_chat(self):
        """
        Send one streaming request and consume the full response.

        Key implementation details:

        1. stream=True on the requests call:
           Tells the requests library NOT to buffer the response body.
           Without this, requests waits for the complete response before
           returning — you would get no streaming behaviour.

        2. Manually call response.iter_lines():
           Reads the SSE stream line by line as chunks arrive.
           Each line is a "data: {...}" JSON fragment containing one token.

        3. Record TTFT at first non-empty chunk:
           time.perf_counter() gives microsecond precision.
           We record the wall-clock time from request send to first
           data chunk — this is the TTFT metric from the user's perspective.

        4. Locust's request_success / request_failure events:
           Locust tracks requests via its internal event system.
           We use self.client.request() with stream=True and manually
           fire success/failure so Locust's stats are accurate for
           streaming (by default Locust stops timing at response headers,
           not at stream end — we override this).
        """
        prompt = PROMPTS[self._prompt_index % len(PROMPTS)]
        self._prompt_index += 1

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a concise technical assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,   # low temperature = deterministic = comparable results
            "max_tokens": 120,
        }

        t_start          = time.perf_counter()
        t_first_token    = None
        token_count      = 0
        total_bytes      = 0
        request_success  = False

        try:
            with self.client.post(
                "/v1/chat/completions",
                json=payload,
                stream=True,
                catch_response=True,    # lets us manually mark success/failure
                timeout=60,
            ) as response:

                if response.status_code != 200:
                    response.failure(
                        f"HTTP {response.status_code}: {response.text[:100]}"
                    )
                    return

                # Consume the SSE stream line by line
                for line in response.iter_lines():
                    if not line:
                        continue

                    line_str = (
                        line.decode("utf-8")
                        if isinstance(line, bytes)
                        else line
                    )

                    if not line_str.startswith("data:"):
                        continue

                    data_str = line_str[5:].strip()

                    if data_str == "[DONE]":
                        break

                    try:
                        chunk    = json.loads(data_str)
                        delta    = (
                            chunk["choices"][0]["delta"].get("content", "")
                        )
                        if delta:
                            total_bytes += len(delta.encode("utf-8"))
                            token_count += 1

                            # Record TTFT at first non-empty token
                            if t_first_token is None:
                                t_first_token = time.perf_counter()

                    except (json.JSONDecodeError, KeyError, IndexError):
                        pass

                # Mark the Locust request as successful
                # response_time is total wall-clock ms (Locust convention)
                t_end          = time.perf_counter()
                total_ms       = (t_end - t_start) * 1000
                ttft_ms        = (
                    (t_first_token - t_start) * 1000
                    if t_first_token else -1
                )
                tps = (
                    token_count / (t_end - t_start)
                    if (t_end - t_start) > 0
                    else 0
                )

                response.success()
                request_success = True

                # Store per-request stats for CSV export
                _results.append({
                    "ttft_ms":    round(ttft_ms, 2),
                    "total_ms":   round(total_ms, 2),
                    "tokens":     token_count,
                    "tps":        round(tps, 2),
                    "prompt":     prompt[:40],
                })

        except Exception:
            if not request_success:
                # Locust will record this as a failure
                pass


# ---------------------------------------------------------------------------
# Event hooks — write CSV at test end
# ---------------------------------------------------------------------------

@events.quit.add_listener
def on_quit(exit_code, **kwargs):
    """
    Called when Locust exits. Write per-request results to CSV.

    Why write our own CSV instead of relying on Locust's built-in stats?
    Locust's built-in stats aggregate by endpoint — they show average
    latency but not TTFT or tokens/sec, which are LLM-specific metrics.
    Our CSV has one row per request with full detail.
    """
    if not _results:
        return

    os.makedirs("results", exist_ok=True)
    filepath = "results/locust_raw.csv"

    with open(filepath, "w") as f:
        f.write("ttft_ms,total_ms,tokens,tps,prompt\n")
        for r in _results:
            f.write(
                f"{r['ttft_ms']},{r['total_ms']},"
                f"{r['tokens']},{r['tps']},{r['prompt']}\n"
            )

    print(f"\nRaw results written to {filepath} ({len(_results)} requests)")

    # Print quick summary to terminal
    if _results:
        ttfts     = sorted(r["ttft_ms"]  for r in _results if r["ttft_ms"] > 0)
        latencies = sorted(r["total_ms"] for r in _results)
        tps_list  = sorted(r["tps"]      for r in _results)

        def pct(lst, p):
            if not lst:
                return 0
            idx = int(len(lst) * p / 100)
            return lst[min(idx, len(lst) - 1)]

        print("\n=== Quick Summary ===")
        print(f"Total requests : {len(_results)}")
        print(f"TTFT  p50/p95/p99 : "
              f"{pct(ttfts,50):.0f} / {pct(ttfts,95):.0f} / {pct(ttfts,99):.0f} ms")
        print(f"Total p50/p95/p99 : "
              f"{pct(latencies,50):.0f} / {pct(latencies,95):.0f} / {pct(latencies,99):.0f} ms")
        print(f"TPS   p50/p95     : "
              f"{pct(tps_list,50):.1f} / {pct(tps_list,95):.1f} tok/s")
        print("====================\n")
