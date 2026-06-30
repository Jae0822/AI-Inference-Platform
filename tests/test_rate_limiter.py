"""
tests/test_rate_limiter.py — Unit tests for Token Bucket rate limiting.

These are pure synchronous tests — rate_limiter has no async code.
No event loop needed, no fixtures beyond reset_rate_limiter.

Test philosophy:
  Each test covers one behaviour of the Token Bucket algorithm.
  Tests are independent: reset_rate_limiter clears state between them.
  Tests do NOT depend on wall-clock time — we manipulate last_refill
  directly to simulate elapsed time without actually sleeping.
"""

import time

import pytest

from gateway import rate_limiter
from gateway.rate_limiter import CAPACITY, REFILL_RATE, check_rate_limit


class TestTokenBucketBasics:
    """Core Token Bucket algorithm correctness."""

    def test_new_ip_starts_with_full_bucket(self, reset_rate_limiter):
        """A first-time IP gets CAPACITY tokens immediately."""
        allowed, info = check_rate_limit("192.168.1.1")
        assert allowed is True
        # After consuming 1 token, remaining = CAPACITY - 1
        assert info["remaining"] == CAPACITY - 1

    def test_bucket_allows_up_to_capacity_requests(self, reset_rate_limiter):
        """Exactly CAPACITY requests succeed before throttling."""
        ip = "10.0.0.1"
        successes = 0
        for _ in range(CAPACITY):
            allowed, _ = check_rate_limit(ip)
            if allowed:
                successes += 1
        assert successes == CAPACITY

    def test_capacity_plus_one_is_rejected(self, reset_rate_limiter):
        """The (CAPACITY + 1)th request is rejected."""
        ip = "10.0.0.2"
        for _ in range(CAPACITY):
            check_rate_limit(ip)   # exhaust the bucket
        allowed, info = check_rate_limit(ip)
        assert allowed is False
        assert info["remaining"] == 0

    def test_rejected_request_does_not_consume_token(self, reset_rate_limiter):
        """
        When a request is rejected, the bucket stays at 0.
        Sending 5 more rejected requests does not push tokens negative.
        """
        ip = "10.0.0.3"
        for _ in range(CAPACITY):
            check_rate_limit(ip)

        for _ in range(5):
            allowed, info = check_rate_limit(ip)
            assert allowed is False
            assert info["remaining"] == 0

    def test_different_ips_have_independent_buckets(self, reset_rate_limiter):
        """
        Exhausting IP A's bucket does not affect IP B.
        This is the core per-IP isolation property.
        """
        ip_a = "1.1.1.1"
        ip_b = "2.2.2.2"

        # Exhaust IP A
        for _ in range(CAPACITY):
            check_rate_limit(ip_a)
        allowed_a, _ = check_rate_limit(ip_a)
        assert allowed_a is False

        # IP B is unaffected
        allowed_b, info_b = check_rate_limit(ip_b)
        assert allowed_b is True
        assert info_b["remaining"] == CAPACITY - 1


class TestTokenRefill:
    """Token refill behaviour (lazy refill on next request)."""

    def test_tokens_refill_over_time(self, reset_rate_limiter):
        """
        After exhausting the bucket, manipulate last_refill to simulate
        elapsed time, then verify tokens are refilled on next request.

        We do NOT use time.sleep() — sleeping makes tests slow and
        fragile. Instead, we directly set last_refill to a past timestamp.
        This tests the refill calculation logic without real time passing.
        """
        ip = "10.0.0.4"

        # Exhaust the bucket
        for _ in range(CAPACITY):
            check_rate_limit(ip)

        # Simulate 2 seconds elapsed by setting last_refill 2s in the past
        rate_limiter._buckets[ip]["last_refill"] = time.time() - 2.0

        # After 2s at REFILL_RATE=2/s, we should have ~4 tokens
        allowed, info = check_rate_limit(ip)
        assert allowed is True
        # Should have refilled ~4 tokens, consumed 1, remaining ~3
        assert info["remaining"] >= 2

    def test_tokens_capped_at_capacity_after_long_idle(self, reset_rate_limiter):
        """
        A bucket idle for a long time should not exceed CAPACITY.
        Without the min(..., CAPACITY) cap, a 1-hour idle would give
        7200 tokens at REFILL_RATE=2/s — a massive unintended burst.
        """
        ip = "10.0.0.5"

        # Exhaust bucket, then simulate 1 hour of idle time
        for _ in range(CAPACITY):
            check_rate_limit(ip)
        rate_limiter._buckets[ip]["last_refill"] = time.time() - 3600.0

        # Consume one token
        check_rate_limit(ip)

        # Remaining should be CAPACITY - 1, not 7200 - 1
        bucket = rate_limiter._buckets[ip]
        assert bucket["tokens"] <= CAPACITY

    def test_partial_refill(self, reset_rate_limiter):
        """
        After 0.5 seconds at REFILL_RATE=2/s, bucket gains 1.0 token.
        This tests that fractional token accumulation works correctly.
        """
        ip = "10.0.0.6"

        # Exhaust the bucket completely
        for _ in range(CAPACITY):
            check_rate_limit(ip)

        # Simulate exactly 0.5 seconds elapsed
        # At REFILL_RATE=2/s: 0.5s × 2 = 1.0 token added
        rate_limiter._buckets[ip]["last_refill"] = time.time() - 0.5

        allowed, info = check_rate_limit(ip)
        assert allowed is True   # 1.0 token available, 1 consumed
        assert info["remaining"] == 0   # consumed the only token


class TestRateLimitInfo:
    """The info dict returned by check_rate_limit."""

    def test_info_contains_required_fields(self, reset_rate_limiter):
        """Every response includes limit, remaining, and reset_in."""
        _, info = check_rate_limit("10.0.0.7")
        assert "limit" in info
        assert "remaining" in info
        assert "reset_in" in info

    def test_limit_equals_capacity(self, reset_rate_limiter):
        """info['limit'] always equals CAPACITY."""
        _, info = check_rate_limit("10.0.0.8")
        assert info["limit"] == CAPACITY

    def test_reset_in_is_positive_when_bucket_empty(self, reset_rate_limiter):
        """When bucket is empty, reset_in tells client how long to wait."""
        ip = "10.0.0.9"
        for _ in range(CAPACITY):
            check_rate_limit(ip)
        _, info = check_rate_limit(ip)
        assert info["reset_in"] > 0


class TestCleanup:
    """Stale bucket eviction."""

    def test_cleanup_removes_stale_buckets(self, reset_rate_limiter):
        """Buckets idle longer than CLEANUP_TTL_SECONDS are evicted."""
        ip = "10.0.0.10"
        check_rate_limit(ip)   # create bucket
        assert ip in rate_limiter._buckets

        # Simulate the bucket being very old
        rate_limiter._buckets[ip]["last_seen"] = (
            time.time() - rate_limiter.CLEANUP_TTL_SECONDS - 1
        )

        removed = rate_limiter.cleanup_stale_buckets()
        assert removed == 1
        assert ip not in rate_limiter._buckets

    def test_active_buckets_not_evicted(self, reset_rate_limiter):
        """Recently accessed buckets are not evicted."""
        ip = "10.0.0.11"
        check_rate_limit(ip)   # creates bucket with last_seen = now

        removed = rate_limiter.cleanup_stale_buckets()
        assert removed == 0
        assert ip in rate_limiter._buckets
