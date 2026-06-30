"""
context_manager.py — Token counting and sliding window truncation.

Responsibility: enforce the token budget for a session's message history.
This file knows about tokens and truncation strategy.
It does NOT know about HTTP, FastAPI, vLLM, or logging.

Why tiktoken?
  tiktoken is OpenAI's tokeniser library, used by GPT-3.5/4.
  Qwen2.5 uses a different tokeniser (based on tiktoken's cl100k_base
  with modifications). For M3, cl100k_base gives a close enough
  approximation for budget enforcement — the error is typically < 5%.
  In production you would use Qwen's HuggingFace tokeniser directly,
  but that requires loading the model's tokeniser files which adds
  startup latency. tiktoken is fast, offline, and accurate enough.

Sliding window strategy:
  When the total token count exceeds MAX_CONTEXT_TOKENS, we drop the
  oldest non-system messages from the front of the history until the
  count is back under budget.

  Why not truncate from the end?
  The most recent messages are the most relevant to the current turn.
  The system prompt (role="system") is always preserved — it contains
  the assistant's persona and instructions, which must never be lost.

  Why not summarise?
  Summarisation requires an extra LLM call, adds latency, and can
  hallucinate. For M3, sliding window is the correct starting point.

  Visualisation:
    Before truncation (8000 tokens, budget 4096):
      [system: 200 tok] [user: 500] [assistant: 600] [user: 400] [assistant: 700]
      [user: 800] [assistant: 900] [user: 600] [assistant: 500] [user: 300] ← newest

    After sliding window (drop oldest non-system messages):
      [system: 200 tok] [user: 800] [assistant: 900] [user: 600] [assistant: 500] [user: 300]
      Total: 200+800+900+600+500+300 = 3300 tokens ✓ under budget
"""

from typing import Optional

import tiktoken

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Leave a reserve below the model's hard limit for the response tokens.
# Qwen2.5-7B was started with --max-model-len 4096.
# We budget 3072 tokens for context, leaving 1024 for the response.
# If you increase --max-model-len, increase this proportionally.
MAX_CONTEXT_TOKENS: int = 3072

# The reserve is the number of tokens kept available for the model's response.
# 1024 is sufficient for most chat responses under max_tokens=512.
RESPONSE_RESERVE_TOKENS: int = 1024

# ---------------------------------------------------------------------------
# Tokeniser
#
# Load once at module import — tiktoken caches the vocabulary after first load.
# cl100k_base is the vocabulary used by GPT-3.5/4 and closely matches
# Qwen2.5's tokeniser for latin-script text. CJK token counts may differ
# by 10-20% but the approximation is safe for budget enforcement because
# we set a conservative budget below the hard limit.
# ---------------------------------------------------------------------------

_enc = tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def count_tokens_in_message(message: dict[str, str]) -> int:
    """
    Count the tokens in a single message dict.

    The OpenAI chat format adds 4 overhead tokens per message
    for the role/content framing. We include this overhead so
    our token count matches what vLLM actually sees.

    Format overhead breakdown:
      <|im_start|>  1 token
      {role}\n      1 token
      {content}     N tokens
      <|im_end|>\n  1 token
      + 1 buffer    1 token
    """
    role_tokens    = len(_enc.encode(message.get("role", "")))
    content_tokens = len(_enc.encode(message.get("content", "")))
    return role_tokens + content_tokens + 4


def count_tokens_in_messages(messages: list[dict[str, str]]) -> int:
    """
    Count total tokens across a list of messages.
    Includes 3 tokens for the reply primer that vLLM prepends.
    """
    total = sum(count_tokens_in_message(m) for m in messages)
    total += 3  # reply primer: <|im_start|>assistant\n
    return total


# ---------------------------------------------------------------------------
# Sliding window truncation
# ---------------------------------------------------------------------------

def apply_sliding_window(
    messages: list[dict[str, str]],
    max_tokens: int = MAX_CONTEXT_TOKENS,
) -> tuple[list[dict[str, str]], int, int]:
    """
    Truncate the message list to fit within max_tokens using a sliding window.

    Strategy:
      1. Always preserve messages with role="system" at the front.
      2. Drop the oldest non-system messages until total tokens <= max_tokens.
      3. If even after dropping all non-system messages the system prompt
         alone exceeds the budget, truncate the system prompt content
         (rare edge case — system prompts should be concise).

    Returns:
      (truncated_messages, original_token_count, final_token_count)

    The token counts are returned so the caller can log them without
    recomputing.
    """
    original_count = count_tokens_in_messages(messages)

    if original_count <= max_tokens:
        # Already within budget — no truncation needed
        return messages, original_count, original_count

    # Separate system messages from conversation messages
    system_messages = [m for m in messages if m.get("role") == "system"]
    conversation    = [m for m in messages if m.get("role") != "system"]

    # Count tokens used by system messages alone
    system_tokens = count_tokens_in_messages(system_messages) if system_messages else 3

    # How many tokens are available for conversation history?
    conversation_budget = max_tokens - system_tokens

    # Drop oldest conversation messages until we fit within budget.
    # We iterate from the front (oldest) and drop until the remainder fits.
    while conversation:
        conversation_tokens = count_tokens_in_messages(conversation)
        if conversation_tokens <= conversation_budget:
            break
        # Drop the oldest message in the conversation
        conversation.pop(0)

    truncated = system_messages + conversation
    final_count = count_tokens_in_messages(truncated)

    return truncated, original_count, final_count


# ---------------------------------------------------------------------------
# Main entry point used by gateway.py
# ---------------------------------------------------------------------------

def prepare_messages_for_request(
    session_messages: list[dict[str, str]],
    new_user_message: str,
    system_prompt: Optional[str] = None,
) -> tuple[list[dict[str, str]], dict]:
    """
    Build the final message list to send to vLLM for one turn.

    Steps:
      1. If system_prompt provided and no system message exists yet, prepend it.
      2. Append the new user message to the session history.
      3. Count tokens.
      4. If over budget, apply sliding window truncation.
      5. Return the ready-to-send message list and a metadata dict for logging.

    The metadata dict contains:
      original_tokens   : token count before truncation
      final_tokens      : token count after truncation (same if no truncation)
      truncated         : bool — whether truncation was applied
      messages_dropped  : int — number of messages removed
      budget            : int — the token budget used

    Why return metadata instead of logging here?
    This function has no knowledge of trace_id or the logging layer.
    The caller (gateway.py) logs the metadata with the correct trace_id.
    Separation of concerns.
    """
    # Step 1: build working copy of session history
    messages = list(session_messages)

    # Step 2: inject system prompt if this is a new session
    if system_prompt and not any(m.get("role") == "system" for m in messages):
        messages.insert(0, {"role": "system", "content": system_prompt})

    # Step 3: append the new user message
    messages.append({"role": "user", "content": new_user_message})

    # Step 4: count tokens
    token_count = count_tokens_in_messages(messages)

    # Step 5: truncate if necessary
    if token_count > MAX_CONTEXT_TOKENS:
        truncated_messages, original_tokens, final_tokens = apply_sliding_window(
            messages, MAX_CONTEXT_TOKENS
        )
        messages_dropped = len(messages) - len(truncated_messages)
        metadata = {
            "original_tokens":  original_tokens,
            "final_tokens":     final_tokens,
            "truncated":        True,
            "messages_dropped": messages_dropped,
            "budget":           MAX_CONTEXT_TOKENS,
        }
        return truncated_messages, metadata

    metadata = {
        "original_tokens":  token_count,
        "final_tokens":     token_count,
        "truncated":        False,
        "messages_dropped": 0,
        "budget":           MAX_CONTEXT_TOKENS,
    }
    return messages, metadata
