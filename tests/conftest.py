"""
tests/conftest.py — Shared pytest fixtures for the AI gateway test suite.

conftest.py is automatically loaded by pytest before any test file.
Fixtures defined here are available to all test files without explicit import.

Key fixtures:
  event_loop        : provides a fresh asyncio event loop per test session
  reset_rate_limiter: clears rate limiter state between tests
  reset_sessions    : clears session store state between tests
  mock_vllm_response: returns a factory for building fake SSE streams
"""

import asyncio

import pytest
import pytest_asyncio

from gateway import rate_limiter, session_store

# ---------------------------------------------------------------------------
# Event loop scope
#
# pytest-asyncio requires an event loop for async tests.
# We use "session" scope so one loop runs all async tests —
# this avoids the overhead of creating a new loop per test.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def event_loop():
    """Provide a single event loop for the entire test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ---------------------------------------------------------------------------
# State isolation fixtures
#
# rate_limiter and session_store hold module-level state (dicts).
# Without cleanup, state leaks between tests, causing order-dependent
# failures. These fixtures reset state before each test that needs it.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=False)
def reset_rate_limiter():
    """
    Clear rate limiter bucket state before and after each test.
    Use this fixture in tests that call check_rate_limit() to prevent
    bucket state from leaking between tests.
    """
    rate_limiter._buckets.clear()
    yield
    rate_limiter._buckets.clear()


@pytest.fixture(autouse=False)
def reset_sessions():
    """
    Clear session store state before and after each test.
    Use this fixture in tests that create sessions.
    """
    session_store._sessions.clear()
    yield
    session_store._sessions.clear()


# ---------------------------------------------------------------------------
# SSE stream factory
#
# vLLM returns Server-Sent Events (SSE) in this format:
#   data: {"choices":[{"delta":{"content":"Hello"},...}],...}\n\n
#   data: {"choices":[{"delta":{"content":" world"},...}],...}\n\n
#   data: [DONE]\n\n
#
# This factory builds fake SSE bytes that look exactly like real vLLM output.
# Tests use this to mock httpx streaming responses without a real vLLM.
# ---------------------------------------------------------------------------

def make_sse_chunk(content: str, finish_reason: str = None) -> bytes:
    """
    Build one SSE data line containing a chat completion delta.
    This is the exact format vLLM emits for streaming responses.
    """
    import json
    delta = {"content": content} if content else {}
    chunk = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 1718000000,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    }
    return f"data: {json.dumps(chunk)}\n\n".encode()


def make_sse_stream(tokens: list[str]) -> bytes:
    """
    Build a complete SSE stream from a list of token strings.
    Includes the [DONE] terminator at the end.

    Example:
        make_sse_stream(["Hello", " world", "!"])
        → b'data: {...delta: "Hello"...}\n\ndata: {...}\n\ndata: [DONE]\n\n'
    """
    chunks = b""
    for token in tokens:
        chunks += make_sse_chunk(token)
    chunks += b"data: [DONE]\n\n"
    return chunks


@pytest.fixture
def sse_stream_factory():
    """
    Fixture that provides the make_sse_stream helper to tests.
    Use in tests that need to build fake vLLM responses.
    """
    return make_sse_stream
