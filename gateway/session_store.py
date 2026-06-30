"""
session_store.py — Per-session conversation history with TTL-based expiry.

Responsibility: store and retrieve message history per session_id.
This file knows nothing about tokens, truncation, or vLLM.
That separation belongs to context_manager.py.

Design decisions:
  - Pure in-memory dict. No Redis, no SQLite. Simple and correct for M3.
  - TTL (time-to-live): sessions inactive for SESSION_TTL_SECONDS are
    evicted on the next access or on the background sweep.
  - A background asyncio.Task runs every SWEEP_INTERVAL_SECONDS and
    removes expired sessions so memory does not grow forever.
  - Thread-safety: not needed. asyncio is single-threaded. The event
    loop never runs two coroutines simultaneously, so dict access is safe
    without locks.
"""

import asyncio
import time
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Sessions with no activity for this long are expired and evicted.
# 30 minutes is appropriate for a conversational assistant.
SESSION_TTL_SECONDS: int = 1800

# How often the background task sweeps for expired sessions.
# Every 5 minutes is sufficient — expiry is enforced on access too.
SWEEP_INTERVAL_SECONDS: int = 300


# ---------------------------------------------------------------------------
# Internal storage
#
# Structure:
#   _sessions = {
#       "session-id-abc": {
#           "messages": [
#               {"role": "system",    "content": "You are ..."},
#               {"role": "user",      "content": "Hello"},
#               {"role": "assistant", "content": "Hi there"},
#               ...
#           ],
#           "last_active": 1718000000.0,   # Unix timestamp of last access
#           "created_at":  1718000000.0,   # Unix timestamp of creation
#       },
#       ...
#   }
# ---------------------------------------------------------------------------

_sessions: dict[str, dict[str, Any]] = {}
_sweep_task: asyncio.Task | None = None


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def start_session_store() -> None:
    """
    Start the background sweep task.
    Must be called from inside a running event loop (FastAPI lifespan startup).
    """
    global _sweep_task
    _sweep_task = asyncio.create_task(
        _sweep_loop(),
        name="session-sweep",
    )


async def stop_session_store() -> None:
    """Cancel the sweep task cleanly on shutdown."""
    global _sweep_task
    if _sweep_task and not _sweep_task.done():
        _sweep_task.cancel()
        try:
            await _sweep_task
        except asyncio.CancelledError:
            pass


async def _sweep_loop() -> None:
    """
    Background coroutine: evict expired sessions on a schedule.

    Why a sweep loop in addition to on-access expiry?
    Without a sweep, sessions from users who never return accumulate
    in memory forever. The sweep loop bounds maximum memory usage.
    """
    while True:
        await asyncio.sleep(SWEEP_INTERVAL_SECONDS)
        _evict_expired()


def _evict_expired() -> int:
    """
    Remove all sessions whose last_active is older than SESSION_TTL_SECONDS.
    Returns the number of sessions evicted.
    Called by the sweep loop and can also be called manually in tests.
    """
    now = time.time()
    expired = [
        sid for sid, data in _sessions.items()
        if now - data["last_active"] > SESSION_TTL_SECONDS
    ]
    for sid in expired:
        del _sessions[sid]
    return len(expired)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_or_create_session(session_id: str) -> dict[str, Any]:
    """
    Return the session dict for session_id, creating it if it does not exist.
    Updates last_active on every access.

    The returned dict contains:
      messages    : list of {"role": str, "content": str}
      last_active : float (Unix timestamp)
      created_at  : float (Unix timestamp)

    Callers should treat this as a read operation.
    To add messages, use append_message().
    """
    now = time.time()

    if session_id not in _sessions:
        _sessions[session_id] = {
            "messages":   [],
            "last_active": now,
            "created_at":  now,
        }
    else:
        # Refresh TTL on every access
        _sessions[session_id]["last_active"] = now

    return _sessions[session_id]


def get_messages(session_id: str) -> list[dict[str, str]]:
    """
    Return the message list for a session.
    Returns an empty list if the session does not exist.
    Does NOT create the session or update last_active.
    """
    session = _sessions.get(session_id)
    if session is None:
        return []
    return session["messages"]


def set_messages(session_id: str, messages: list[dict[str, str]]) -> None:
    """
    Overwrite the message list for a session.
    Used by context_manager after truncation.
    Creates the session if it does not exist.
    """
    session = get_or_create_session(session_id)
    session["messages"] = messages


def append_message(session_id: str, role: str, content: str) -> None:
    """
    Append one message to a session's history.
    Creates the session if it does not exist.
    """
    session = get_or_create_session(session_id)
    session["messages"].append({"role": role, "content": content})


def delete_session(session_id: str) -> None:
    """Explicitly delete a session. No-op if it does not exist."""
    _sessions.pop(session_id, None)


def session_count() -> int:
    """Return the number of active sessions. Useful for metrics and health checks."""
    return len(_sessions)
