"""
test_stream.py — Multi-turn conversation test client for M3.

Tests:
  1. Stateless mode (backward compat) — same as M1/M2
  2. Stateful multi-turn — 4 turns in the same session
  3. Session inspection — show what the gateway remembers
  4. New session — prove sessions are isolated
"""

import json

import requests

GATEWAY_URL = "http://localhost:8000/v1/chat/completions"
SESSION_URL = "http://localhost:8000/session"


# ---------------------------------------------------------------------------
# Shared streaming helper
# ---------------------------------------------------------------------------

def stream_request(payload: dict, label: str) -> tuple[str, str | None]:
    """
    Send a request and stream the response token by token.
    Returns (full_response_text, trace_id).
    Works for both stateless (messages) and stateful (session_id+user_message).
    """
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    if "user_message" in payload:
        print(f"  Session  : {payload.get('session_id')}")
        print(f"  User     : {payload['user_message'][:80]}")
    else:
        last_user = next(
            (m["content"] for m in reversed(payload.get("messages", []))
             if m["role"] == "user"), ""
        )
        print(f"  Prompt   : {last_user[:80]}")

    print(f"{'─'*60}")
    print("  Response : ", end="", flush=True)

    response = requests.post(
        GATEWAY_URL,
        json=payload,
        stream=True,
        timeout=60,
    )

    trace_id   = response.headers.get("X-Trace-Id", "none")
    full_text  = []

    for line in response.iter_lines():
        if not line:
            continue
        line_str = line.decode("utf-8") if isinstance(line, bytes) else line
        if not line_str.startswith("data:"):
            continue
        data_str = line_str[5:].strip()
        if data_str in ("[DONE]", ""):
            continue
        try:
            chunk = json.loads(data_str)
            delta = chunk["choices"][0]["delta"].get("content", "")
            if delta:
                print(delta, end="", flush=True)
                full_text.append(delta)
        except (json.JSONDecodeError, KeyError, IndexError):
            pass

    print(f"\n{'─'*60}")
    print(f"  Trace ID : {trace_id}")
    print(f"{'='*60}")
    return "".join(full_text), trace_id


# ---------------------------------------------------------------------------
# Test 1: stateless mode — backward compatibility check
# ---------------------------------------------------------------------------

def test_stateless():
    print("\n" + "█"*60)
    print("  TEST 1 — Stateless mode (backward compatible)")
    print("█"*60)

    stream_request(
        payload={
            "messages": [
                {"role": "system",  "content": "You are a concise assistant."},
                {"role": "user",    "content": "What is vLLM in one sentence?"},
            ],
            "temperature": 0.7,
            "max_tokens":  100,
        },
        label="Stateless request (messages field)",
    )


# ---------------------------------------------------------------------------
# Test 2: multi-turn stateful conversation
# ---------------------------------------------------------------------------

def test_multi_turn():
    print("\n" + "█"*60)
    print("  TEST 2 — Multi-turn stateful conversation")
    print("█"*60)

    session_id = "test-session-m3"

    turns = [
        "My name is Alex. Please remember that.",
        "What is PagedAttention and why does it improve throughput?",
        "Can you summarise what we have discussed so far?",
        "What was my name again?",   # ← tests that the model remembers turn 1
    ]

    for i, turn in enumerate(turns, 1):
        stream_request(
            payload={
                "session_id":   session_id,
                "user_message": turn,
                "temperature":  0.7,
                "max_tokens":   150,
            },
            label=f"Turn {i} of {len(turns)}",
        )

    # After all turns, inspect what the gateway remembered
    print(f"\n{'─'*60}")
    print("  Session inspection after 4 turns:")
    resp = requests.get(f"{SESSION_URL}/{session_id}", timeout=10)
    if resp.status_code == 200:
        info = resp.json()
        print(f"  session_id     : {info['session_id']}")
        print(f"  message_count  : {info['message_count']}")
        print(f"  token_count    : {info['token_count']}")
        print(f"  budget         : {info['budget']}")
        print(f"  budget_used    : {info['budget_used_pct']}%")
    else:
        print(f"  ERROR: {resp.status_code} {resp.text}")


# ---------------------------------------------------------------------------
# Test 3: session isolation — new session has no memory of test 2
# ---------------------------------------------------------------------------

def test_session_isolation():
    print("\n" + "█"*60)
    print("  TEST 3 — Session isolation")
    print("█"*60)

    stream_request(
        payload={
            "session_id":   "completely-new-session",
            "user_message": "What is my name?",
            "temperature":  0.7,
            "max_tokens":   80,
        },
        label="New session — should not know Alex's name",
    )
    print("  Expected: model says it does not know / has no context.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n AI Gateway — M3 Multi-turn Test Client\n")
    test_stateless()
    test_multi_turn()
    test_session_isolation()
    print("\nAll tests complete.")
    print("Check logs: cat logs/gateway.jsonl | jq 'select(.event == \"context_window_status\")'")
