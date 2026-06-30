"""
logger.py — Async structured logger with trace ID propagation via contextvars.

Engineering goals:
  1. Never block the asyncio event loop on a disk write.
  2. Every log line is valid JSON (structured logging).
  3. trace_id is attached to the log record automatically via contextvars —
     no caller needs to pass it explicitly.
  4. A single background asyncio.Task drains the queue and writes to disk,
     so all 100 concurrent request-coroutines never compete on file I/O.

Key components:
  - contextvars.ContextVar  : holds the trace_id for the current async context
  - asyncio.Queue           : decouples log production from disk I/O
  - asyncio.Task            : background drain loop (started once at app startup)
  - json.dumps              : formats every record as a single-line JSON string
"""

import asyncio
import contextvars
import json
import os
import sys
import time
import uuid
from typing import Any

# ---------------------------------------------------------------------------
# 1. The ContextVar — this is the heart of trace ID propagation.
#
# A ContextVar is like a thread-local variable, but for async coroutines.
# Each coroutine that calls `ctx_trace_id.set(value)` gets its own copy
# of the variable. Child coroutines spawned with asyncio.create_task()
# automatically inherit the parent's ContextVar values at the moment of
# task creation (Python copies the entire context snapshot).
#
# This means: set trace_id once at request entry → every coroutine that
# runs within that request's async call chain sees the same trace_id,
# with zero explicit passing.
# ---------------------------------------------------------------------------
ctx_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "trace_id", default="no-trace"
)


# ---------------------------------------------------------------------------
# 2. The async log queue.
#
# asyncio.Queue is the standard async-safe data structure for producer/consumer
# patterns. It has no locks — it works via awaitable get/put operations that
# yield control to the event loop instead of spinning or blocking.
#
# maxsize=0 means unbounded. In production you would set a limit and drop
# or sample log records under extreme memory pressure, but for M1 unbounded
# is correct — simplicity over premature optimisation.
# ---------------------------------------------------------------------------
_log_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()


# ---------------------------------------------------------------------------
# 3. The background drain task.
#
# This asyncio.Task runs forever alongside your FastAPI application.
# It blocks on `await _log_queue.get()` — meaning it yields the event loop
# while waiting, so it costs zero CPU while idle.
#
# When a log record arrives, it formats it as JSON and writes to:
#   - the log file (appended, one JSON object per line = JSONL format)
#   - stderr (so you see logs in the terminal during development)
#
# The file write is the only synchronous I/O in this entire module.
# Because it runs in a single dedicated Task, only one coroutine is ever
# doing file I/O at a time — no contention, no need for locks.
# ---------------------------------------------------------------------------
_drain_task: asyncio.Task | None = None
_log_file_handle = None


async def _drain_loop(log_path: str) -> None:
    """
    Background coroutine: reads from _log_queue and writes to disk.

    Runs for the entire lifetime of the application. Terminates cleanly
    when it receives None as a sentinel value (sent by shutdown()).

    Why a dedicated drain loop rather than writing inline?
    Because `open(...).write(...)` is a synchronous blocking call.
    If you call it directly inside a streaming response coroutine, you
    block the event loop for every single token emission — destroying
    the concurrency you worked hard to build in M0.
    """
    global _log_file_handle

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Open once, keep the handle open for the application lifetime.
    # Appending mode: safe to restart without losing previous logs.
    _log_file_handle = open(log_path, "a", encoding="utf-8")

    try:
        while True:
            record = await _log_queue.get()

            # None is the shutdown sentinel — exit the loop cleanly.
            if record is None:
                break

            line = json.dumps(record, ensure_ascii=False)

            # Write to file
            _log_file_handle.write(line + "\n")
            _log_file_handle.flush()   # ensure line is on disk immediately

            # Mirror to stderr so you see it in the terminal
            print(line, file=sys.stderr)

    finally:
        if _log_file_handle:
            _log_file_handle.close()


# ---------------------------------------------------------------------------
# 4. Public API: startup / shutdown
# ---------------------------------------------------------------------------

def start_logger(log_path: str = "logs/gateway.jsonl") -> None:
    """
    Start the background drain task.

    Must be called from inside a running asyncio event loop — so call it
    from FastAPI's @app.on_event("startup") handler, not at module import.

    Why not start the task at module import time?
    asyncio.create_task() requires a running event loop. At module import
    time, no event loop is running yet. FastAPI's startup event fires after
    Uvicorn starts the loop, so that is the correct place.
    """
    global _drain_task
    _drain_task = asyncio.create_task(
        _drain_loop(log_path),
        name="log-drain",
    )


async def stop_logger() -> None:
    """
    Gracefully drain the queue and shut down the drain task.

    Sends None as sentinel, then awaits the task so all queued records
    are written before the process exits.
    """
    await _log_queue.put(None)
    if _drain_task:
        await _drain_task


# ---------------------------------------------------------------------------
# 5. Trace ID management
# ---------------------------------------------------------------------------

def generate_trace_id() -> str:
    """
    Generate a short, URL-safe unique trace ID.

    uuid4() gives 128 bits of randomness. We take the first 8 hex chars
    (32 bits) — enough to be unique across thousands of concurrent requests,
    short enough to read at a glance in logs.

    Example output: "a3f9c21b"
    """
    return uuid.uuid4().hex[:8]


def set_trace_id(trace_id: str) -> contextvars.Token:
    """
    Set the trace_id for the current async context.

    Returns a Token — you can pass this to ctx_trace_id.reset(token) to
    restore the previous value. In practice, FastAPI middleware sets the
    trace_id once per request and never needs to reset it, so the token
    is discarded.
    """
    return ctx_trace_id.set(trace_id)


def get_trace_id() -> str:
    """
    Read the trace_id for the current async context.

    Returns the ContextVar's default ("no-trace") if no trace_id has been
    set — which should only happen in background tasks not tied to a request.
    """
    return ctx_trace_id.get()


# ---------------------------------------------------------------------------
# 6. The main log() function — what callers use
# ---------------------------------------------------------------------------

def log(
    level: str,
    event: str,
    **kwargs: Any,
) -> None:
    """
    Enqueue a structured log record.

    This function is intentionally synchronous (not async).

    Why synchronous?
    Because _log_queue.put_nowait() is a non-blocking call — it places the
    record in the in-memory queue instantly and returns. The actual disk write
    happens later in the drain task. So log() can be called from anywhere:
    sync functions, async functions, inside generators — without awaiting.

    The record automatically includes:
      - timestamp  : Unix epoch float, precise to microseconds
      - level      : "INFO", "WARNING", "ERROR", etc.
      - trace_id   : pulled from the current async context via ContextVar
      - event      : human-readable description of what happened
      - **kwargs   : any structured fields the caller wants to attach

    Example call:
        log("INFO", "request_received", method="POST", path="/v1/chat/completions")

    Example output (one line of JSONL):
        {"ts": 1718000000.123, "level": "INFO", "trace_id": "a3f9c21b",
         "event": "request_received", "method": "POST", "path": "/v1/chat/completions"}
    """
    record: dict[str, Any] = {
        "ts": time.time(),
        "level": level.upper(),
        "trace_id": get_trace_id(),   # ← reads from ContextVar, no argument needed
        "event": event,
        **kwargs,
    }
    # put_nowait() never blocks. If the queue is full (it's unbounded here,
    # so that won't happen), it raises QueueFull — but we're not handling that
    # in M1. Production systems would sample or drop here.
    _log_queue.put_nowait(record)


# ---------------------------------------------------------------------------
# 7. Convenience wrappers — matching standard log level names
# ---------------------------------------------------------------------------

def info(event: str, **kwargs: Any) -> None:
    log("INFO", event, **kwargs)

def warning(event: str, **kwargs: Any) -> None:
    log("WARNING", event, **kwargs)

def error(event: str, **kwargs: Any) -> None:
    log("ERROR", event, **kwargs)

def debug(event: str, **kwargs: Any) -> None:
    log("DEBUG", event, **kwargs)
