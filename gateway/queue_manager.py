"""
queue_manager.py — Bounded concurrency control with backpressure signaling.

Core concept: asyncio.Semaphore
  A Semaphore holds an internal counter initialised to N.
  acquire() decrements the counter. If counter == 0, acquire() blocks
  (suspends the coroutine) until another coroutine calls release().
  release() increments the counter.

  In plain English:
    Semaphore(60) means "at most 60 coroutines may be inside the
    critical section simultaneously. The 61st must wait or be rejected."

Why Semaphore instead of asyncio.Queue?
  asyncio.Queue is designed for producer/consumer patterns where work
  items are passed between coroutines. Here we do not pass work items —
  we just want to limit how many requests are simultaneously in flight.
  Semaphore is the correct primitive for concurrency limiting.

  Queue would require a separate worker pool, adding complexity without
  benefit for this use case.

Why NOT a threading.Semaphore?
  threading.Semaphore uses OS-level locking. In an asyncio application,
  OS locks block the entire event loop thread. asyncio.Semaphore is
  implemented with asyncio primitives — acquire() is an await point,
  so it suspends only the current coroutine, not the whole loop.

Capacity design (from M7 benchmark data):
  Saturation point: 50-80 concurrent users.
  We set MAX_CONCURRENT = 60, giving a small buffer above the healthy
  range (50 users) and a hard ceiling below the crash point (80+ users).
  Requests beyond 60 get an immediate 503, protecting the 60 that are
  already being served.

  The queue depth (QUEUE_DEPTH) is how many requests we allow to wait
  for a Semaphore slot before rejecting. Set to 0 here — we reject
  immediately rather than making users wait in a hidden queue. This
  gives predictable latency: either you get in (fast) or you get a
  clear signal to retry (503).
"""

import asyncio
import time
from typing import Optional

from gateway import logger

# ---------------------------------------------------------------------------
# Configuration
#
# Based on M7 benchmark findings:
#   - System handles 50 users cleanly (TTFT p99 = 219ms)
#   - System saturates between 50-80 users
#   - Set limit at 60: comfortable buffer above 50, safe below 80
#
# Adjust MAX_CONCURRENT upward after M10 quantization reduces VRAM usage
# and frees KV cache headroom for more concurrent requests.
# ---------------------------------------------------------------------------

MAX_CONCURRENT: int = 60
RETRY_AFTER_SECONDS: int = 5


# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------

_semaphore: Optional[asyncio.Semaphore] = None
_active: int = 0          # current requests holding the semaphore
_rejected: int = 0        # total requests rejected since startup


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def init_queue_manager() -> None:
    """
    Initialise the semaphore.

    Must be called from inside a running event loop (FastAPI lifespan).
    asyncio.Semaphore() must be created in the same event loop that will
    use it — creating it at module import time (before the loop starts)
    causes a RuntimeError in Python 3.10+.
    """
    global _semaphore
    _semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    logger.info(
        "queue_manager_initialized",
        max_concurrent=MAX_CONCURRENT,
        retry_after_seconds=RETRY_AFTER_SECONDS,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_at_capacity() -> bool:
    """
    Return True if no semaphore slots are available.

    Semaphore._value is the current counter (available slots).
    Accessing _value is safe in single-threaded asyncio — there is no
    race condition because the event loop is not truly parallel.

    Why check before acquiring rather than just trying to acquire?
    We want to reject immediately (non-blocking) rather than suspending
    the coroutine. A non-blocking check + immediate 503 gives the client
    a clear, fast signal. A blocked acquire() would hold the client
    connection open silently, which is worse UX and wastes resources.
    """
    if _semaphore is None:
        return False
    return _semaphore._value == 0


class RequestSlot:
    """
    Context manager that acquires and releases the semaphore.

    Usage in gateway.py:
        slot = queue_manager.RequestSlot()
        if not await slot.acquire():
            raise HTTPException(503, ...)
        try:
            # handle request
        finally:
            slot.release()

    Why a class instead of a bare async with?
    We need to update metrics (_active counter, Prometheus Gauge) on
    both acquire and release. Wrapping in a class keeps that bookkeeping
    in one place and makes gateway.py clean.
    """

    def __init__(self):
        self._acquired = False

    async def acquire(self) -> bool:
        """
        Try to acquire a semaphore slot without blocking.

        Returns True if acquired (request may proceed).
        Returns False if at capacity (caller should return 503).
        """
        global _active, _rejected

        if _semaphore is None:
            # Queue manager not initialised — fail open (allow request)
            logger.warning("queue_manager_not_initialized")
            return True

        if is_at_capacity():
            _rejected += 1
            logger.warning(
                "request_rejected_capacity",
                active=_active,
                max_concurrent=MAX_CONCURRENT,
                total_rejected=_rejected,
            )
            return False

        # Semaphore has a free slot — acquire it.
        # acquire() would normally suspend if value == 0, but we already
        # checked above, so this returns immediately.
        await _semaphore.acquire()
        _active += 1

        logger.info(
            "request_slot_acquired",
            active=_active,
            available=_semaphore._value,
        )
        self._acquired = True
        return True

    def release(self) -> None:
        """
        Release the semaphore slot.
        Always call this in a finally block to prevent slot leaks.
        """
        global _active

        if self._acquired and _semaphore is not None:
            _semaphore.release()
            _active -= 1
            self._acquired = False

            logger.info(
                "request_slot_released",
                active=_active,
                available=_semaphore._value,
            )


# ---------------------------------------------------------------------------
# Status (for /health endpoint)
# ---------------------------------------------------------------------------

def get_status() -> dict:
    """Return current queue manager state for health checks."""
    return {
        "max_concurrent":    MAX_CONCURRENT,
        "active":            _active,
        "available":         _semaphore._value if _semaphore else 0,
        "total_rejected":    _rejected,
        "at_capacity":       is_at_capacity(),
    }
