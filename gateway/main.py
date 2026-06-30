"""
gateway.py — AI inference gateway.
  M1: structured async logging + trace ID
  M2: Prometheus metrics
  M3: multi-turn session management + context window enforcement
  M4: GPU hardware monitoring
  M8: bounded concurrency queue + backpressure
  M9: per-IP Token Bucket rate limiting

M9 middleware fix:
  BaseHTTPMiddleware has a known issue with StreamingResponse: it buffers
  the response body, breaking true streaming, and cannot reliably modify
  response headers after call_next(). The fix is to use Starlette's native
  @app.middleware("http") decorator which operates at the ASGI level and
  works correctly with streaming responses.

  Rate limiting is implemented as a pure @app.middleware("http") function.
  TraceMiddleware is also converted to the same pattern for consistency.

  Middleware execution order with @app.middleware("http"):
    First registered = outermost layer.
    We register rate_limit_middleware first (outermost) so rejected
    requests never reach the trace middleware.
"""

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from gateway import context_manager
from gateway import gpu_monitor
from gateway import logger
from gateway import metrics
from gateway import queue_manager
from gateway import rate_limiter
from gateway import session_store

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VLLM_URL      = "http://localhost:6006/v1/chat/completions"
MODEL_PATH    = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct-AWQ/qwen/Qwen2___5-7B-Instruct-AWQ"
LOG_PATH      = "logs/gateway.jsonl"
ENDPOINT_PATH = "/v1/chat/completions"

DEFAULT_SYSTEM_PROMPT = "You are a concise and helpful technical assistant."

_SILENT_PATHS = {"/metrics", "/health"}
_EXEMPT_PATHS = {"/metrics", "/health"}


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.start_logger(LOG_PATH)
    session_store.start_session_store()
    gpu_monitor.start_gpu_monitor()
    queue_manager.init_queue_manager()
    logger.info(
        "gateway_started",
        vllm_url=VLLM_URL,
        log_path=LOG_PATH,
        port=8000,
        max_concurrent=queue_manager.MAX_CONCURRENT,
        rate_limit_capacity=rate_limiter.CAPACITY,
        rate_limit_refill=rate_limiter.REFILL_RATE,
    )
    yield
    logger.info("gateway_stopping")
    await gpu_monitor.stop_gpu_monitor()
    await session_store.stop_session_store()
    await logger.stop_logger()


app = FastAPI(title="AI Inference Gateway — M9", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Middleware — native @app.middleware("http") pattern
#
# Why this instead of BaseHTTPMiddleware?
#
# BaseHTTPMiddleware wraps the ASGI app in a way that buffers streaming
# responses and loses header modifications after call_next() returns.
# This is a long-standing Starlette issue documented at:
# https://www.starlette.io/middleware/#basehttpmiddleware
#
# @app.middleware("http") operates at the raw ASGI level. It receives
# the request BEFORE routing, can short-circuit with any Response, and
# does NOT buffer streaming responses. Headers set on the Response object
# returned by call_next() ARE preserved for non-streaming responses, but
# for StreamingResponse we still use request.state to pass data to the
# endpoint (same approach as before, but now it actually works because
# the middleware correctly intercepts all requests).
#
# Execution order: FIRST registered = OUTERMOST layer.
# We register rate_limit_middleware before trace_middleware so that
# rejected requests never reach the trace layer.
# ---------------------------------------------------------------------------

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """
    Outermost middleware: per-IP Token Bucket rate limiting.
    Exempt paths bypass rate limiting entirely.
    """
    if request.url.path in _EXEMPT_PATHS:
        return await call_next(request)

    # Resolve client IP (respect X-Forwarded-For from reverse proxies)
    forwarded_for = request.headers.get("X-Forwarded-For")
    client_ip = (
        forwarded_for.split(",")[0].strip()
        if forwarded_for
        else (request.client.host if request.client else "unknown")
    )

    allowed, info = rate_limiter.check_rate_limit(client_ip)

    if not allowed:
        logger.warning(
            "rate_limit_exceeded",
            client_ip=client_ip,
            limit=info["limit"],
            reset_in=info["reset_in"],
        )
        return JSONResponse(
            status_code=429,
            content={
                "error":      "Too Many Requests",
                "limit":      info["limit"],
                "reset_in_s": info["reset_in"],
                "message":    f"Rate limit exceeded. Retry after {info['reset_in']}s.",
            },
            headers={
                "Retry-After":           str(int(info["reset_in"]) + 1),
                "X-RateLimit-Limit":     str(info["limit"]),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset":     str(int(info["reset_in"]) + 1),
            },
        )

    # Store info in request.state for the endpoint to read
    request.state.rate_limit_info = info
    return await call_next(request)


@app.middleware("http")
async def trace_middleware(request: Request, call_next):
    """
    Inner middleware: generate trace ID and log every real request.
    """
    trace_id = logger.generate_trace_id()
    logger.set_trace_id(trace_id)

    is_silent = request.url.path in _SILENT_PATHS

    if not is_silent:
        logger.info(
            "http_request_received",
            method=request.method,
            path=request.url.path,
            client=request.client.host if request.client else "unknown",
        )

    t_start  = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - t_start) * 1000

    # X-Trace-Id on the response works correctly with @app.middleware("http")
    response.headers["X-Trace-Id"] = trace_id

    if not is_silent:
        logger.info(
            "http_response_sent",
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
        )

    return response


# ---------------------------------------------------------------------------
# Helper: build X-RateLimit-* headers from request.state
# ---------------------------------------------------------------------------

def _rl_headers(request: Request) -> dict:
    info = getattr(request.state, "rate_limit_info", {})
    return {
        "X-RateLimit-Limit":     str(info.get("limit",     rate_limiter.CAPACITY)),
        "X-RateLimit-Remaining": str(info.get("remaining", rate_limiter.CAPACITY)),
        "X-RateLimit-Reset":     str(int(info.get("reset_in", 0)) + 1),
    }


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages:     Optional[List[Message]] = None
    session_id:   Optional[str]           = None
    user_message: Optional[str]           = None
    temperature:  float                   = 0.7
    max_tokens:   int                     = 512


# ---------------------------------------------------------------------------
# Streaming generator — unchanged from M4
# ---------------------------------------------------------------------------

async def stream_from_vllm(
    payload: dict,
    path: str = ENDPOINT_PATH,
) -> AsyncGenerator[bytes, None]:
    metrics.ACTIVE_REQUESTS.labels(path=path).inc()

    logger.info(
        "vllm_request_start",
        vllm_url=VLLM_URL,
        temperature=payload.get("temperature"),
        message_count=len(payload.get("messages", [])),
    )

    t_first_token: float | None = None
    token_count:   int          = 0
    status_code:   str          = "200"
    t_request_start             = time.perf_counter()

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", VLLM_URL, json=payload) as response:

                if response.status_code != 200:
                    status_code = str(response.status_code)
                    logger.error("vllm_upstream_error",
                                 status_code=response.status_code)
                    yield (
                        f"data: [ERROR] vLLM returned status "
                        f"{response.status_code}\n\n"
                    ).encode()
                    return

                async for chunk in response.aiter_bytes():
                    if not chunk:
                        continue

                    if t_first_token is None:
                        t_first_token = time.perf_counter()
                        ttft_seconds  = t_first_token - t_request_start
                        metrics.TTFT_HISTOGRAM.labels(path=path).observe(ttft_seconds)
                        logger.info("vllm_first_token",
                                    ttft_ms=round(ttft_seconds * 1000, 2))

                    token_count += 1
                    yield chunk

    except httpx.TimeoutException:
        status_code = "504"
        logger.error("vllm_timeout", timeout_seconds=120.0)
        yield b"data: [ERROR] vLLM request timed out\n\n"

    except httpx.ConnectError:
        status_code = "503"
        logger.error("vllm_connection_refused", vllm_url=VLLM_URL)
        yield b"data: [ERROR] Cannot connect to vLLM\n\n"

    except Exception as exc:
        status_code = "500"
        logger.error("vllm_unexpected_error",
                     error_type=type(exc).__name__, error=str(exc))
        yield b"data: [ERROR] Unexpected error during streaming\n\n"

    else:
        total_seconds = time.perf_counter() - t_request_start
        tps = (token_count / total_seconds) if total_seconds > 0 else 0.0
        metrics.TOKENS_PER_SECOND.labels(path=path).observe(tps)
        metrics.REQUEST_LATENCY.labels(path=path).observe(total_seconds)
        logger.info(
            "vllm_stream_complete",
            total_chunks=token_count,
            total_ms=round(total_seconds * 1000, 2),
            tokens_per_sec=round(tps, 2),
        )

    finally:
        metrics.ACTIVE_REQUESTS.labels(path=path).dec()
        metrics.REQUESTS_TOTAL.labels(
            path=path, status_code=status_code
        ).inc()


# ---------------------------------------------------------------------------
# Session-aware streaming — unchanged from M3
# ---------------------------------------------------------------------------

async def stream_and_save(
    payload: dict,
    session_id: str,
    path: str = ENDPOINT_PATH,
) -> AsyncGenerator[bytes, None]:
    import json
    accumulated_text = []

    async for chunk in stream_from_vllm(payload, path):
        yield chunk
        try:
            chunk_str = chunk.decode("utf-8")
            for line in chunk_str.split("\n"):
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str in ("[DONE]", ""):
                    continue
                data = json.loads(data_str)
                delta_content = (
                    data.get("choices", [{}])[0]
                    .get("delta", {})
                    .get("content", "")
                )
                if delta_content:
                    accumulated_text.append(delta_content)
        except (json.JSONDecodeError, IndexError, KeyError):
            pass

    full_reply = "".join(accumulated_text)
    if full_reply:
        session_store.append_message(session_id, "assistant", full_reply)
        logger.info(
            "session_reply_saved",
            session_id=session_id,
            reply_length=len(full_reply),
            session_message_count=len(
                session_store.get_messages(session_id)
            ),
        )


# ---------------------------------------------------------------------------
# Slot wrapper — unchanged from M8
# ---------------------------------------------------------------------------

async def _stream_with_slot(
    generator: AsyncGenerator[bytes, None],
    slot: queue_manager.RequestSlot,
) -> AsyncGenerator[bytes, None]:
    try:
        async for chunk in generator:
            yield chunk
    finally:
        slot.release()
        metrics.QUEUE_ACTIVE.dec()


# ---------------------------------------------------------------------------
# Main endpoint
# ---------------------------------------------------------------------------

@app.post(ENDPOINT_PATH)
async def chat_completions(request_body: ChatRequest, request: Request):

    # Acquire concurrency slot (M8)
    slot = queue_manager.RequestSlot()
    acquired = await slot.acquire()

    if not acquired:
        metrics.QUEUE_REJECTED_TOTAL.inc()
        raise HTTPException(
            status_code=503,
            detail={
                "error":       "Gateway at capacity",
                "active":      queue_manager._active,
                "max":         queue_manager.MAX_CONCURRENT,
                "retry_after": queue_manager.RETRY_AFTER_SECONDS,
            },
            headers={"Retry-After": str(queue_manager.RETRY_AFTER_SECONDS)},
        )

    metrics.QUEUE_ACTIVE.inc()

    # Rate limit headers to include in StreamingResponse
    rl = _rl_headers(request)

    # Mode A: stateless
    if request_body.messages is not None and request_body.session_id is None:
        for msg in request_body.messages:
            if "违规词" in msg.content:
                slot.release()
                metrics.QUEUE_ACTIVE.dec()
                logger.warning("content_blocked", reason="prohibited_keyword")
                raise HTTPException(status_code=400, detail="Content blocked")

        logger.info(
            "inference_request_validated",
            mode="stateless",
            message_count=len(request_body.messages),
            temperature=request_body.temperature,
            max_tokens=request_body.max_tokens,
        )

        payload = {
            "model":       MODEL_PATH,
            "messages":    [m.model_dump() for m in request_body.messages],
            "temperature": request_body.temperature,
            "max_tokens":  request_body.max_tokens,
            "stream":      True,
        }

        return StreamingResponse(
            _stream_with_slot(
                stream_from_vllm(payload, path=ENDPOINT_PATH), slot
            ),
            media_type="text/event-stream",
            headers={
                "X-Trace-Id":        logger.get_trace_id(),
                "Cache-Control":     "no-cache",
                "X-Accel-Buffering": "no",
                **rl,
            },
        )

    # Mode B: stateful
    if request_body.session_id is not None and request_body.user_message is not None:
        session_id   = request_body.session_id
        user_message = request_body.user_message

        if "违规词" in user_message:
            slot.release()
            metrics.QUEUE_ACTIVE.dec()
            logger.warning("content_blocked", reason="prohibited_keyword",
                           session_id=session_id)
            raise HTTPException(status_code=400, detail="Content blocked")

        existing_messages = session_store.get_messages(session_id)
        messages_for_request, ctx_meta = context_manager.prepare_messages_for_request(
            session_messages=existing_messages,
            new_user_message=user_message,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
        )

        logger.info(
            "context_window_status",
            session_id=session_id,
            original_tokens=ctx_meta["original_tokens"],
            final_tokens=ctx_meta["final_tokens"],
            truncated=ctx_meta["truncated"],
            messages_dropped=ctx_meta["messages_dropped"],
            budget=ctx_meta["budget"],
            history_length=len(existing_messages),
        )

        if ctx_meta["truncated"]:
            logger.warning(
                "context_truncated",
                session_id=session_id,
                dropped=ctx_meta["messages_dropped"],
                tokens_before=ctx_meta["original_tokens"],
                tokens_after=ctx_meta["final_tokens"],
            )

        session_store.append_message(session_id, "user", user_message)

        logger.info(
            "inference_request_validated",
            mode="stateful",
            session_id=session_id,
            message_count=len(messages_for_request),
            temperature=request_body.temperature,
            max_tokens=request_body.max_tokens,
        )

        payload = {
            "model":       MODEL_PATH,
            "messages":    messages_for_request,
            "temperature": request_body.temperature,
            "max_tokens":  request_body.max_tokens,
            "stream":      True,
        }

        return StreamingResponse(
            _stream_with_slot(
                stream_and_save(payload, session_id, path=ENDPOINT_PATH),
                slot,
            ),
            media_type="text/event-stream",
            headers={
                "X-Trace-Id":        logger.get_trace_id(),
                "X-Session-Id":      session_id,
                "Cache-Control":     "no-cache",
                "X-Accel-Buffering": "no",
                **rl,
            },
        )

    # Invalid request
    slot.release()
    metrics.QUEUE_ACTIVE.dec()
    raise HTTPException(
        status_code=422,
        detail=(
            "Provide either 'messages' (stateless mode) or "
            "both 'session_id' and 'user_message' (stateful mode)."
        ),
    )


# ---------------------------------------------------------------------------
# Session info
# ---------------------------------------------------------------------------

@app.get("/session/{session_id}")
async def get_session_info(session_id: str):
    messages = session_store.get_messages(session_id)
    if not messages:
        raise HTTPException(status_code=404, detail="Session not found")
    token_count = context_manager.count_tokens_in_messages(messages)
    return {
        "session_id":      session_id,
        "message_count":   len(messages),
        "token_count":     token_count,
        "budget":          context_manager.MAX_CONTEXT_TOKENS,
        "budget_used_pct": round(
            token_count / context_manager.MAX_CONTEXT_TOKENS * 100, 1
        ),
        "messages":        messages,
    }


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    idx = str(gpu_monitor.GPU_INDEX)
    try:
        vram_used  = gpu_monitor.GPU_VRAM_USED_BYTES.labels(gpu_index=idx)._value.get()
        vram_total = gpu_monitor.GPU_VRAM_TOTAL_BYTES.labels(gpu_index=idx)._value.get()
        util       = gpu_monitor.GPU_UTILIZATION_RATIO.labels(gpu_index=idx)._value.get()
        temp       = gpu_monitor.GPU_TEMPERATURE_CELSIUS.labels(gpu_index=idx)._value.get()
        power      = gpu_monitor.GPU_POWER_WATTS.labels(gpu_index=idx)._value.get()
        gpu_info = {
            "vram_used_gb":    round(vram_used / 1e9, 2),
            "vram_total_gb":   round(vram_total / 1e9, 2),
            "vram_free_pct":   round((1 - vram_used / vram_total) * 100, 1)
                               if vram_total > 0 else 0,
            "utilization_pct": round(util * 100, 1),
            "temperature_c":   temp,
            "power_w":         round(power, 1),
        }
    except Exception:
        gpu_info = {"error": "gpu metrics not yet available"}

    return {
        "status":          "ok",
        "active_sessions": session_store.session_count(),
        "queue":           queue_manager.get_status(),
        "rate_limiter":    rate_limiter.get_status(),
        "gpu":             gpu_info,
        "trace_id":        logger.get_trace_id(),
    }


# ---------------------------------------------------------------------------
# /metrics
# ---------------------------------------------------------------------------

@app.get("/metrics")
async def prometheus_metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
