"""
rate_limiter.py — Per-IP Token Bucket rate limiting.

Core concept: Token Bucket algorithm
  Imagine a bucket with a fixed capacity of N tokens.
  Tokens are added at a constant rate (REFILL_RATE tokens per second).
  Each request consumes 1 token.
  If the bucket is empty, the request is rejected with 429.
  If the bucket has tokens, the request is allowed and 1 token is consumed.

  Key properties:
    - Burst allowance: a bucket starting full allows CAPACITY requests
      in rapid succession before throttling kicks in. This models
      realistic usage where a user legitimately sends a few requests
      at once (opening a page that fires 3 API calls).
    - Sustained rate: over time, throughput is bounded to REFILL_RATE
      requests/second regardless of burst behaviour.
    - Per-IP isolation: each IP has its own bucket. One IP being
      throttled has zero effect on any other IP's bucket.

Why Token Bucket over alternatives?

  Fixed Window (count requests per minute):
    Simple but has a "boundary burst" problem. A user can send
    LIMIT requests at 00:59 and LIMIT more at 01:00 — 2× the
    intended limit in 2 seconds. Token Bucket has no boundary.

  Sliding Window Log (store timestamp of every request):
    Accurate but requires O(requests) memory per IP. Under attack
    (millions of requests/sec), memory blows up. Token Bucket is O(1)
    per IP regardless of request rate.

  Leaky Bucket (queue with fixed drain rate):
    Smooths output to exactly REFILL_RATE req/s. No burst allowed.
    Too strict for a chat API where users legitimately send a few
    messages in quick succession. Token Bucket allows burst up to
    CAPACITY while still bounding the sustained rate.

Token Bucket is the industry standard for API rate limiting.
It is used by AWS API Gateway, Cloudflare, nginx limit_req,
and virtually every production API gateway.

Thread safety:
  asyncio is single-threaded. The event loop never runs two coroutines
  truly in parallel. Therefore, reading and updating the per-IP bucket
  dict is safe without locks — there is no race condition possible
  in a single-threaded event loop.

Memory management:
  Each IP entry is a small dict (~200 bytes). 10,000 distinct IPs
  = ~2 MB. We add a simple TTL cleanup to evict IPs that have been
  idle for CLEANUP_TTL_SECONDS, bounding memory in production.
"""

import time
from typing import Optional

from gateway import logger

# ---------------------------------------------------------------------------
# Configuration
#
# CAPACITY: maximum tokens in the bucket = maximum burst size.
#   Set to 10: a user can send 10 requests in rapid succession
#   before being throttled. Realistic for a chat interface.
#
# REFILL_RATE: tokens added per second.
#   Set to 2: sustained rate of 2 requests/second per IP.
#   A human typing chat messages rarely exceeds 1/s; 2/s gives
#   comfortable headroom for legitimate usage.
#
# CLEANUP_TTL_SECONDS: evict IP entries idle for this long.
#   Set to 600 (10 minutes). Prevents memory growth from IPs
#   that made one request and never returned.
# ---------------------------------------------------------------------------

CAPACITY:             int   = 10
REFILL_RATE:          float = 2.0
CLEANUP_TTL_SECONDS:  int   = 600


# ---------------------------------------------------------------------------
# Per-IP bucket storage
#
# Structure:
#   _buckets = {
#       "192.168.1.1": {
#           "tokens":     9.5,        # current token count (float)
#           "last_refill": 1718000000.0,  # Unix timestamp of last check
#           "last_seen":   1718000000.0,  # for TTL cleanup
#       },
#       ...
#   }
#
# Why float for tokens?
#   Refill is time-based: tokens += elapsed_seconds * REFILL_RATE.
#   elapsed_seconds is a float, so tokens must be float to accumulate
#   fractional tokens correctly between requests.
#   A token is only consumed when >= 1.0 is available.
# ---------------------------------------------------------------------------

_buckets: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def _get_or_create_bucket(ip: str) -> dict:
    """
    Return the bucket for this IP, creating it full if it does not exist.
    A new bucket starts full (CAPACITY tokens) — first-time users get
    the full burst allowance immediately.
    """
    now = time.time()
    if ip not in _buckets:
        _buckets[ip] = {
            "tokens":      float(CAPACITY),
            "last_refill": now,
            "last_seen":   now,
        }
    return _buckets[ip]


def _refill(bucket: dict) -> None:
    """
    Add tokens to the bucket based on elapsed time since last refill.

    Formula:
      elapsed = now - last_refill
      new_tokens = elapsed * REFILL_RATE
      tokens = min(tokens + new_tokens, CAPACITY)

    Why min(..., CAPACITY)?
      Without the cap, a bucket that has been idle for an hour would
      accumulate 7200 tokens (at 2/s), allowing a burst of 7200 requests.
      Capping at CAPACITY ensures the burst allowance never exceeds N
      regardless of how long the IP has been idle.

    This refill happens lazily — only when a request arrives.
    We do NOT run a background task to refill all buckets continuously.
    Lazy refill is O(1) per request and requires no background task.
    """
    now = time.time()
    elapsed = now - bucket["last_refill"]
    new_tokens = elapsed * REFILL_RATE
    bucket["tokens"] = min(bucket["tokens"] + new_tokens, float(CAPACITY))
    bucket["last_refill"] = now
    bucket["last_seen"] = now


def check_rate_limit(ip: str) -> tuple[bool, dict]:
    """
    Check whether this IP is within its rate limit.

    Returns:
      (allowed, info)
      allowed : True if the request should proceed, False if throttled
      info    : dict with current bucket state for response headers

    Side effect:
      If allowed, consumes 1 token from the bucket.
      If not allowed, bucket is unchanged (no token consumed).

    The info dict contains:
      limit     : CAPACITY (max tokens)
      remaining : tokens left after this request (int floor)
      reset_in  : seconds until bucket refills to CAPACITY
    """
    bucket = _get_or_create_bucket(ip)
    _refill(bucket)

    # Calculate how long until the bucket is full again
    tokens_needed_to_fill = CAPACITY - bucket["tokens"]
    reset_in = (tokens_needed_to_fill / REFILL_RATE) if REFILL_RATE > 0 else 0

    info = {
        "limit":     CAPACITY,
        "remaining": max(0, int(bucket["tokens"])),
        "reset_in":  round(reset_in, 1),
    }

    if bucket["tokens"] >= 1.0:
        bucket["tokens"] -= 1.0
        info["remaining"] = max(0, int(bucket["tokens"]))
        return True, info
    else:
        return False, info


# ---------------------------------------------------------------------------
# Memory cleanup
# ---------------------------------------------------------------------------

def cleanup_stale_buckets() -> int:
    """
    Remove IP entries that have been idle for CLEANUP_TTL_SECONDS.

    Called from the background sweep in session_store — we reuse the
    existing sweep infrastructure rather than adding a new background task.

    Returns the number of entries removed.
    """
    now = time.time()
    stale = [
        ip for ip, b in _buckets.items()
        if now - b["last_seen"] > CLEANUP_TTL_SECONDS
    ]
    for ip in stale:
        del _buckets[ip]

    if stale:
        logger.info(
            "rate_limiter_cleanup",
            removed=len(stale),
            remaining=len(_buckets),
        )

    return len(stale)


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def get_status() -> dict:
    """Return current rate limiter state for health checks."""
    return {
        "capacity":       CAPACITY,
        "refill_rate":    REFILL_RATE,
        "tracked_ips":    len(_buckets),
        "cleanup_ttl_s":  CLEANUP_TTL_SECONDS,
    }
