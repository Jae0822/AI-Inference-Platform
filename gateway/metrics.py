"""
metrics.py — Prometheus metric definitions for the AI inference gateway.
M8 adds two instruments for queue depth and rejection tracking.
"""

from prometheus_client import (  # noqa: F401
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# ---------------------------------------------------------------------------
# Bucket boundaries
# ---------------------------------------------------------------------------

_TTFT_BUCKETS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0)
_LATENCY_BUCKETS = (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 30.0)
_TPS_BUCKETS = (5, 10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200)


# ---------------------------------------------------------------------------
# M2 instruments — unchanged
# ---------------------------------------------------------------------------

REQUESTS_TOTAL = Counter(
    name="gateway_http_requests_total",
    documentation="Total HTTP requests received by the gateway.",
    labelnames=["path", "status_code"],
)

REQUEST_LATENCY = Histogram(
    name="gateway_http_request_duration_seconds",
    documentation="End-to-end request duration in seconds.",
    labelnames=["path"],
    buckets=_LATENCY_BUCKETS,
)

TTFT_HISTOGRAM = Histogram(
    name="gateway_llm_ttft_seconds",
    documentation="Time from vLLM request submission to first token received.",
    labelnames=["path"],
    buckets=_TTFT_BUCKETS,
)

TOKENS_PER_SECOND = Histogram(
    name="gateway_llm_tokens_per_second",
    documentation="Tokens per second achieved for completed streaming requests.",
    labelnames=["path"],
    buckets=_TPS_BUCKETS,
)

ACTIVE_REQUESTS = Gauge(
    name="gateway_http_active_requests",
    documentation="Number of streaming requests currently in flight.",
    labelnames=["path"],
)


# ---------------------------------------------------------------------------
# M8 instruments — NEW
#
# QUEUE_ACTIVE: how many requests are currently holding a semaphore slot.
#   This is the concurrency signal from the queue manager's perspective.
#   Distinct from ACTIVE_REQUESTS (which tracks streaming generator state).
#   Both should move together — if they diverge, a slot leak has occurred.
#
# QUEUE_REJECTED_TOTAL: cumulative count of requests turned away with 503.
#   Use rate(gateway_queue_rejected_total[1m]) in Grafana to see the
#   rejection rate per second — the key signal for capacity planning.
# ---------------------------------------------------------------------------

QUEUE_ACTIVE = Gauge(
    name="gateway_queue_active",
    documentation=(
        "Requests currently holding a concurrency slot. "
        "Ceiling is queue_manager.MAX_CONCURRENT."
    ),
)

QUEUE_REJECTED_TOTAL = Counter(
    name="gateway_queue_rejected_total",
    documentation=(
        "Total requests rejected with 503 due to capacity limit. "
        "Use rate()[1m] in Grafana to see rejections per second."
    ),
)
