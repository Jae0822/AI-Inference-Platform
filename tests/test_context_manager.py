"""
tests/test_context_manager.py — Unit tests for context window management.

Tests the sliding window truncation algorithm, token counting,
and the prepare_messages_for_request() entry point.

All tests are synchronous — context_manager has no async code.
"""

import pytest

from gateway import context_manager
from gateway.context_manager import (
    MAX_CONTEXT_TOKENS,
    apply_sliding_window,
    count_tokens_in_message,
    count_tokens_in_messages,
    prepare_messages_for_request,
)


class TestTokenCounting:
    """Token counting correctness."""

    def test_empty_message_has_overhead_tokens(self):
        """
        Even an empty message has overhead tokens (role + framing).
        We add 4 overhead tokens per message in count_tokens_in_message.
        """
        msg = {"role": "user", "content": ""}
        count = count_tokens_in_message(msg)
        assert count >= 4   # at minimum the overhead

    def test_longer_content_has_more_tokens(self):
        """More content = more tokens."""
        short_msg = {"role": "user", "content": "Hi"}
        long_msg  = {"role": "user", "content": "Hi " * 100}
        assert count_tokens_in_message(long_msg) > count_tokens_in_message(short_msg)

    def test_message_list_count_is_sum_plus_primer(self):
        """
        Total count = sum of individual messages + 3 tokens (reply primer).
        """
        messages = [
            {"role": "user",      "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        individual_sum = sum(count_tokens_in_message(m) for m in messages)
        total = count_tokens_in_messages(messages)
        assert total == individual_sum + 3   # +3 for reply primer

    def test_single_message_list(self):
        """A list with one message is counted correctly."""
        messages = [{"role": "user", "content": "test"}]
        count = count_tokens_in_messages(messages)
        assert count > 0


class TestSlidingWindow:
    """Sliding window truncation algorithm."""

    def _make_messages(self, n_turns: int, content: str = "x " * 50) -> list:
        """Helper: build a conversation with n_turns of user+assistant pairs."""
        msgs = [{"role": "system", "content": "You are helpful."}]
        for i in range(n_turns):
            msgs.append({"role": "user",      "content": f"Turn {i}: {content}"})
            msgs.append({"role": "assistant",  "content": f"Reply {i}: {content}"})
        return msgs

    def test_no_truncation_when_within_budget(self):
        """Short conversations are returned unchanged."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user",   "content": "Hi"},
        ]
        truncated, original, final = apply_sliding_window(
            messages, max_tokens=MAX_CONTEXT_TOKENS
        )
        assert len(truncated) == len(messages)
        assert original == final   # no tokens dropped

    def test_truncation_removes_oldest_messages(self):
        """When over budget, oldest non-system messages are dropped."""
        # Build a conversation that exceeds the budget
        messages = self._make_messages(n_turns=30, content="word " * 30)
        original_count = count_tokens_in_messages(messages)

        truncated, original_tokens, final_tokens = apply_sliding_window(
            messages, max_tokens=MAX_CONTEXT_TOKENS
        )

        # If original exceeded budget, truncation should have occurred
        if original_count > MAX_CONTEXT_TOKENS:
            assert len(truncated) < len(messages)
            assert final_tokens <= MAX_CONTEXT_TOKENS

    def test_system_message_always_preserved(self):
        """System messages are never dropped, even under heavy truncation."""
        system_content = "You are a specialized assistant."
        messages = [{"role": "system", "content": system_content}]
        # Add many turns to exceed budget
        for i in range(50):
            messages.append({"role": "user",      "content": "word " * 40})
            messages.append({"role": "assistant",  "content": "word " * 40})

        truncated, _, _ = apply_sliding_window(
            messages, max_tokens=MAX_CONTEXT_TOKENS
        )

        system_msgs = [m for m in truncated if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0]["content"] == system_content

    def test_truncated_result_fits_within_budget(self):
        """After truncation, token count <= max_tokens."""
        messages = []
        for i in range(100):
            messages.append({"role": "user",      "content": "word " * 20})
            messages.append({"role": "assistant",  "content": "word " * 20})

        truncated, _, final_tokens = apply_sliding_window(
            messages, max_tokens=512   # very tight budget
        )
        assert final_tokens <= 512

    def test_returns_original_count_and_final_count(self):
        """apply_sliding_window returns (messages, original_count, final_count)."""
        messages = [{"role": "user", "content": "hello"}]
        result = apply_sliding_window(messages, max_tokens=MAX_CONTEXT_TOKENS)
        assert len(result) == 3   # tuple of 3 elements
        _, original, final = result
        assert isinstance(original, int)
        assert isinstance(final, int)
        assert final <= original


class TestPrepareMessages:
    """prepare_messages_for_request() integration."""

    def test_adds_system_prompt_to_empty_session(self):
        """First turn: system prompt is prepended if session is empty."""
        messages, meta = prepare_messages_for_request(
            session_messages=[],
            new_user_message="Hello",
            system_prompt="You are helpful.",
        )
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful."

    def test_does_not_duplicate_system_prompt(self):
        """If session already has a system message, no duplicate is added."""
        existing = [{"role": "system", "content": "Existing prompt."}]
        messages, _ = prepare_messages_for_request(
            session_messages=existing,
            new_user_message="Hello",
            system_prompt="New prompt.",
        )
        system_msgs = [m for m in messages if m["role"] == "system"]
        assert len(system_msgs) == 1
        # Should keep the existing one, not add the new one
        assert system_msgs[0]["content"] == "Existing prompt."

    def test_user_message_is_appended(self):
        """The new user message is the last message."""
        messages, _ = prepare_messages_for_request(
            session_messages=[],
            new_user_message="What is vLLM?",
            system_prompt="Be concise.",
        )
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "What is vLLM?"

    def test_metadata_truncated_false_when_within_budget(self):
        """Short conversations produce meta['truncated'] = False."""
        _, meta = prepare_messages_for_request(
            session_messages=[],
            new_user_message="Hi",
            system_prompt="Be helpful.",
        )
        assert meta["truncated"] is False
        assert meta["messages_dropped"] == 0

    def test_metadata_truncated_true_when_over_budget(self):
        """Long conversations produce meta['truncated'] = True."""
        # Build a session that will exceed the budget
        long_session = []
        for i in range(80):
            long_session.append({"role": "user",      "content": "word " * 30})
            long_session.append({"role": "assistant",  "content": "word " * 30})

        _, meta = prepare_messages_for_request(
            session_messages=long_session,
            new_user_message="One more question",
            system_prompt="Be helpful.",
        )
        assert meta["truncated"] is True
        assert meta["messages_dropped"] > 0
        assert meta["final_tokens"] <= MAX_CONTEXT_TOKENS

    def test_metadata_contains_budget(self):
        """meta['budget'] equals MAX_CONTEXT_TOKENS."""
        _, meta = prepare_messages_for_request(
            session_messages=[],
            new_user_message="Hi",
        )
        assert meta["budget"] == MAX_CONTEXT_TOKENS
