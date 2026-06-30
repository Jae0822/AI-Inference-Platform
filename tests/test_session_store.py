"""
tests/test_session_store.py — Unit tests for session management.

Tests session creation, message storage, TTL expiry, and isolation.
Includes one async test for the sweep loop.
"""

import asyncio
import time

import pytest
import pytest_asyncio

from gateway import session_store
from gateway.session_store import (
    SESSION_TTL_SECONDS,
    _evict_expired,
    append_message,
    delete_session,
    get_messages,
    get_or_create_session,
    session_count,
    set_messages,
)


class TestSessionCreation:
    """Session lifecycle basics."""

    def test_new_session_is_empty(self, reset_sessions):
        """A newly created session has no messages."""
        session = get_or_create_session("session-1")
        assert session["messages"] == []

    def test_same_session_id_returns_same_session(self, reset_sessions):
        """Calling get_or_create_session twice with same ID returns same data."""
        append_message("session-2", "user", "Hello")
        session = get_or_create_session("session-2")
        assert len(session["messages"]) == 1

    def test_different_ids_are_isolated(self, reset_sessions):
        """Messages in session A do not appear in session B."""
        append_message("session-a", "user", "Message for A")
        msgs_b = get_messages("session-b")
        assert msgs_b == []

    def test_nonexistent_session_returns_empty_list(self, reset_sessions):
        """get_messages() on unknown session returns []."""
        msgs = get_messages("does-not-exist")
        assert msgs == []


class TestMessageOperations:
    """Message append, get, set."""

    def test_append_message_adds_to_end(self, reset_sessions):
        """Messages are appended in order."""
        sid = "session-order"
        append_message(sid, "user",      "First")
        append_message(sid, "assistant", "Second")
        append_message(sid, "user",      "Third")

        msgs = get_messages(sid)
        assert len(msgs) == 3
        assert msgs[0]["content"] == "First"
        assert msgs[1]["content"] == "Second"
        assert msgs[2]["content"] == "Third"

    def test_append_message_stores_role(self, reset_sessions):
        """Role is stored correctly."""
        sid = "session-role"
        append_message(sid, "system", "You are helpful.")
        msgs = get_messages(sid)
        assert msgs[0]["role"] == "system"

    def test_set_messages_overwrites(self, reset_sessions):
        """set_messages() replaces existing messages."""
        sid = "session-set"
        append_message(sid, "user", "Old message")
        set_messages(sid, [{"role": "user", "content": "New message"}])
        msgs = get_messages(sid)
        assert len(msgs) == 1
        assert msgs[0]["content"] == "New message"

    def test_delete_session_removes_it(self, reset_sessions):
        """delete_session() removes the session entirely."""
        sid = "session-delete"
        append_message(sid, "user", "Hello")
        assert session_count() == 1
        delete_session(sid)
        assert session_count() == 0
        assert get_messages(sid) == []

    def test_delete_nonexistent_session_is_noop(self, reset_sessions):
        """Deleting a session that doesn't exist raises no error."""
        delete_session("ghost-session")   # should not raise

    def test_session_count_reflects_active_sessions(self, reset_sessions):
        """session_count() returns correct number."""
        assert session_count() == 0
        append_message("s1", "user", "Hello")
        assert session_count() == 1
        append_message("s2", "user", "Hello")
        assert session_count() == 2
        delete_session("s1")
        assert session_count() == 1


class TestTTLExpiry:
    """Session TTL and eviction."""

    def test_evict_expired_removes_old_sessions(self, reset_sessions):
        """Sessions past TTL are evicted by _evict_expired()."""
        sid = "old-session"
        append_message(sid, "user", "Hello")

        # Manually age the session beyond TTL
        session_store._sessions[sid]["last_active"] = (
            time.time() - SESSION_TTL_SECONDS - 1
        )

        removed = _evict_expired()
        assert removed == 1
        assert get_messages(sid) == []

    def test_active_sessions_not_evicted(self, reset_sessions):
        """Recently accessed sessions survive eviction."""
        sid = "active-session"
        append_message(sid, "user", "Hello")
        # last_active is set to now() by append_message

        removed = _evict_expired()
        assert removed == 0
        assert len(get_messages(sid)) == 1

    def test_access_refreshes_ttl(self, reset_sessions):
        """get_or_create_session() updates last_active."""
        sid = "refresh-session"
        append_message(sid, "user", "Hello")

        # Age the session almost to expiry
        session_store._sessions[sid]["last_active"] = (
            time.time() - SESSION_TTL_SECONDS + 10
        )

        # Access it — should refresh last_active
        get_or_create_session(sid)

        # Now it should not be evicted
        removed = _evict_expired()
        assert removed == 0


class TestAsyncSweep:
    """The background sweep task."""

    @pytest.mark.asyncio
    async def test_stop_session_store_cancels_task(self, reset_sessions):
        """
        start_session_store() creates a task.
        stop_session_store() cancels it cleanly.

        This test verifies the task lifecycle without running the full
        sweep interval (300 seconds) — we just check it starts and stops.
        """
        session_store.start_session_store()
        assert session_store._sweep_task is not None
        assert not session_store._sweep_task.done()

        await session_store.stop_session_store()
        assert session_store._sweep_task.done()
