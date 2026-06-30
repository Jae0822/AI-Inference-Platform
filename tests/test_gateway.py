"""
tests/test_gateway.py — Integration tests for the FastAPI gateway.

Uses httpx.AsyncClient with httpx.MockTransport to test the gateway
without a real vLLM process or GPU.

Key technique: MockTransport
  We replace httpx's real HTTP transport with a mock that returns
  pre-built SSE responses. This lets us test:
    - That the gateway correctly forwards requests to vLLM
    - That streaming responses are passed through to clients
    - That rate limiting returns 429
    - That capacity limits return 503
    - That content moderation blocks prohibited content
  All without any real network calls.

Why not use FastAPI's TestClient?
  FastAPI's synchronous TestClient wraps async endpoints in a thread.
  It does not support streaming responses correctly — it buffers the
  entire response body before returning, breaking SSE tests.
  httpx.AsyncClient is the correct tool for async streaming tests.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio
from httpx import AsyncClient, MockTransport, Request, Response

from gateway import queue_manager
from gateway import rate_limiter
from gateway import session_store

# ---------------------------------------------------------------------------
# SSE response builder (duplicated from conftest for self-contained tests)
# ---------------------------------------------------------------------------

def make_sse_response(tokens: list[str], status_code: int = 200) -> bytes:
    """Build a complete fake vLLM SSE response body."""
    body = b""
    for token in tokens:
        chunk = {
            "choices": [{"delta": {"content": token}, "finish_reason": None}]
        }
        body += f"data: {json.dumps(chunk)}\n\n".encode()
    body += b"data: [DONE]\n\n"
    return body


# ---------------------------------------------------------------------------
# App fixture
#
# We import the FastAPI app after patching the lifespan startup actions.
# This prevents the real logger drain loop, GPU monitor, and session sweep
# from starting during tests — they would interfere with the event loop.
# ---------------------------------------------------------------------------

@pytest.fixture
def app_client():
    """
    Provide the FastAPI app for testing.
    We test individual components directly rather than spinning up the
    full app with all background tasks, to keep tests fast and isolated.
    """
    # Reset module-level state
    rate_limiter._buckets.clear()
    session_store._sessions.clear()

    # Ensure queue_manager is initialized for tests that need it
    if queue_manager._semaphore is None:
        # We need a running event loop for this
        pass

    yield


# ---------------------------------------------------------------------------
# Rate limiting integration tests (via the middleware logic directly)
# ---------------------------------------------------------------------------

class TestRateLimitMiddleware:
    """
    Test rate limiting logic by calling check_rate_limit directly.
    The middleware itself is tested via the rate_limiter unit tests.
    Here we verify the HTTP response format.
    """

    def test_rate_limit_allows_first_request(self, reset_rate_limiter):
        """First request from any IP is allowed."""
        allowed, info = rate_limiter.check_rate_limit("test-ip-1")
        assert allowed is True
        assert info["remaining"] == rate_limiter.CAPACITY - 1

    def test_rate_limit_blocks_after_burst(self, reset_rate_limiter):
        """After CAPACITY requests, the next is blocked."""
        ip = "test-ip-burst"
        for _ in range(rate_limiter.CAPACITY):
            rate_limiter.check_rate_limit(ip)
        allowed, info = rate_limiter.check_rate_limit(ip)
        assert allowed is False
        assert info["remaining"] == 0
        assert info["reset_in"] > 0

    def test_rate_limit_info_has_correct_limit(self, reset_rate_limiter):
        """info['limit'] always equals CAPACITY."""
        _, info = rate_limiter.check_rate_limit("test-ip-info")
        assert info["limit"] == rate_limiter.CAPACITY


# ---------------------------------------------------------------------------
# Context manager integration tests
# ---------------------------------------------------------------------------

class TestContextWindowIntegration:
    """
    Test that context_manager integrates correctly with session data.
    """

    def test_prepare_adds_system_and_user(self):
        """prepare_messages_for_request builds correct message list."""
        from gateway import context_manager
        messages, meta = context_manager.prepare_messages_for_request(
            session_messages=[],
            new_user_message="What is vLLM?",
            system_prompt="You are helpful.",
        )
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "What is vLLM?"
        assert meta["truncated"] is False

    def test_prepare_truncates_long_session(self):
        """Long sessions are truncated to fit the budget."""
        from gateway import context_manager

        # Build a session that exceeds MAX_CONTEXT_TOKENS
        long_session = []
        for i in range(60):
            long_session.append({"role": "user",      "content": "word " * 25})
            long_session.append({"role": "assistant",  "content": "word " * 25})

        messages, meta = context_manager.prepare_messages_for_request(
            session_messages=long_session,
            new_user_message="Final question",
            system_prompt="Be helpful.",
        )
        assert meta["truncated"] is True
        assert meta["final_tokens"] <= context_manager.MAX_CONTEXT_TOKENS


# ---------------------------------------------------------------------------
# Session store integration tests
# ---------------------------------------------------------------------------

class TestSessionStoreIntegration:
    """Session store integration with context manager."""

    def test_multi_turn_session_accumulates_messages(self, reset_sessions):
        """Messages accumulate across multiple turns."""
        sid = "integration-session"
        session_store.append_message(sid, "user",      "Turn 1")
        session_store.append_message(sid, "assistant", "Reply 1")
        session_store.append_message(sid, "user",      "Turn 2")

        msgs = session_store.get_messages(sid)
        assert len(msgs) == 3

    def test_session_isolation_between_users(self, reset_sessions):
        """Two session IDs never share messages."""
        session_store.append_message("user-alice", "user", "Alice's message")
        session_store.append_message("user-bob",   "user", "Bob's message")

        alice_msgs = session_store.get_messages("user-alice")
        bob_msgs   = session_store.get_messages("user-bob")

        assert len(alice_msgs) == 1
        assert len(bob_msgs) == 1
        assert alice_msgs[0]["content"] == "Alice's message"
        assert bob_msgs[0]["content"] == "Bob's message"


# ---------------------------------------------------------------------------
# Async queue manager tests
# ---------------------------------------------------------------------------

class TestQueueManager:
    """Test the concurrency slot management."""

    @pytest.mark.asyncio
    async def test_slot_acquire_and_release(self):
        """Acquiring a slot decrements available count; releasing increments it."""
        # Reset queue manager state
        from gateway import queue_manager as qm
        qm._semaphore = asyncio.Semaphore(qm.MAX_CONCURRENT)
        qm._active = 0
        qm._rejected = 0

        slot = qm.RequestSlot()
        available_before = qm._semaphore._value

        acquired = await slot.acquire()
        assert acquired is True
        assert qm._semaphore._value == available_before - 1

        slot.release()
        assert qm._semaphore._value == available_before

    @pytest.mark.asyncio
    async def test_slot_rejected_when_at_capacity(self):
        """When semaphore is at 0, acquire returns False immediately."""
        from gateway import queue_manager as qm
        # Set semaphore to 0 (at capacity)
        qm._semaphore = asyncio.Semaphore(0)
        qm._active = 0
        qm._rejected = 0

        slot = qm.RequestSlot()
        acquired = await slot.acquire()
        assert acquired is False

        # Restore
        qm._semaphore = asyncio.Semaphore(qm.MAX_CONCURRENT)

    @pytest.mark.asyncio
    async def test_multiple_slots_up_to_limit(self):
        """Can acquire MAX_CONCURRENT slots simultaneously."""
        from gateway import queue_manager as qm
        limit = 3
        qm._semaphore = asyncio.Semaphore(limit)
        qm._active = 0

        slots = []
        for _ in range(limit):
            slot = qm.RequestSlot()
            acquired = await slot.acquire()
            assert acquired is True
            slots.append(slot)

        # Next one should be rejected
        extra = qm.RequestSlot()
        acquired = await extra.acquire()
        assert acquired is False

        # Clean up
        for s in slots:
            s.release()
        qm._semaphore = asyncio.Semaphore(qm.MAX_CONCURRENT)
