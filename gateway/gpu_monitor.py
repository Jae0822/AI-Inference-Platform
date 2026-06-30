"""
gpu_monitor.py — Async GPU hardware monitor with Prometheus Gauges.

Responsibility:
  Poll nvidia-smi every POLL_INTERVAL_SECONDS.
  Parse the output into structured numbers.
  Write those numbers into Prometheus Gauge instruments.
  Never block the event loop.

Why nvidia-smi instead of pynvml or PyTorch's cuda API?
  pynvml: requires installing pynvml separately, and its C bindings
          occasionally deadlock when called from inside a process that
          already holds a CUDA context (e.g. vLLM). Not safe here.
  torch.cuda: only available if PyTorch is imported. We want gpu_monitor
              to be a standalone module with no ML dependencies.
  nvidia-smi: ships with every NVIDIA driver. Always available.
              Subprocess call is the safest isolation boundary —
              nvidia-smi runs in its own process, so any driver-level
              issue cannot crash the gateway.

Why asyncio.create_subprocess_exec instead of subprocess.run?
  subprocess.run() is a blocking call. Calling it directly in a coroutine
  would stall the entire event loop for the duration of the nvidia-smi
  process (typically 50-100ms). asyncio.create_subprocess_exec() launches
  the subprocess without blocking — the event loop remains free to serve
  requests while nvidia-smi is running.

Change from original M4:
  _poll_loop no longer calls logger.debug() on every poll cycle.
  Polling every 5 seconds produces 17,280 log lines per day —
  pure noise that makes logs/gateway.jsonl unsearchable.
  Instead, we log once on first successful reading (gpu_first_reading)
  and then run silently. Errors and warnings are still logged.
"""

import asyncio
import re
from typing import Optional

from prometheus_client import Gauge

from gateway import logger

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

POLL_INTERVAL_SECONDS: int = 5
GPU_INDEX: int = 0


# ---------------------------------------------------------------------------
# Prometheus Gauge definitions
# ---------------------------------------------------------------------------

GPU_VRAM_USED_BYTES = Gauge(
    name="gateway_gpu_vram_used_bytes",
    documentation=(
        "GPU VRAM currently in use, in bytes. "
        "Includes model weights + KV cache + framework overhead."
    ),
    labelnames=["gpu_index"],
)

GPU_VRAM_TOTAL_BYTES = Gauge(
    name="gateway_gpu_vram_total_bytes",
    documentation="Total GPU VRAM available, in bytes.",
    labelnames=["gpu_index"],
)

GPU_VRAM_FREE_BYTES = Gauge(
    name="gateway_gpu_vram_free_bytes",
    documentation="GPU VRAM currently free, in bytes.",
    labelnames=["gpu_index"],
)

GPU_UTILIZATION_RATIO = Gauge(
    name="gateway_gpu_utilization_ratio",
    documentation=(
        "GPU compute utilization as a ratio 0.0-1.0. "
        "1.0 means the GPU compute units are fully occupied."
    ),
    labelnames=["gpu_index"],
)

GPU_TEMPERATURE_CELSIUS = Gauge(
    name="gateway_gpu_temperature_celsius",
    documentation="GPU die temperature in degrees Celsius.",
    labelnames=["gpu_index"],
)

GPU_POWER_WATTS = Gauge(
    name="gateway_gpu_power_watts",
    documentation="GPU power draw in watts.",
    labelnames=["gpu_index"],
)


# ---------------------------------------------------------------------------
# nvidia-smi query and parsing
# ---------------------------------------------------------------------------

_NVIDIASMI_QUERY = (
    "memory.used,"
    "memory.total,"
    "memory.free,"
    "utilization.gpu,"
    "temperature.gpu,"
    "power.draw"
)

_NVIDIASMI_CMD = [
    "nvidia-smi",
    f"--query-gpu={_NVIDIASMI_QUERY}",
    "--format=csv,noheader,nounits",
    f"--id={GPU_INDEX}",
]

_NUMBER_RE = re.compile(r"[\d.]+")


async def _run_nvidiasmi() -> Optional[str]:
    """
    Run nvidia-smi as an async subprocess and return its stdout.
    Returns None if the process fails or is not found.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            *_NVIDIASMI_CMD,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            logger.warning(
                "nvidiasmi_nonzero_exit",
                returncode=proc.returncode,
                stderr=stderr.decode("utf-8", errors="replace").strip(),
            )
            return None

        return stdout.decode("utf-8", errors="replace").strip()

    except FileNotFoundError:
        logger.error("nvidiasmi_not_found", cmd=_NVIDIASMI_CMD[0])
        return None

    except Exception as exc:
        logger.error(
            "nvidiasmi_unexpected_error",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return None


def _parse_nvidiasmi_output(raw: str) -> Optional[dict]:
    """
    Parse the CSV output from nvidia-smi into a structured dict.

    Expected raw input (example):
      "14336, 32768, 18432, 87, 71, 245.30"

    Fields (in order, matching _NVIDIASMI_QUERY):
      0: memory.used   MiB
      1: memory.total  MiB
      2: memory.free   MiB
      3: utilization   %
      4: temperature   Celsius
      5: power.draw    W

    All memory values are converted from MiB to bytes.
    Utilization is converted from % to ratio (0.0-1.0).
    """
    MIB_TO_BYTES = 1_048_576

    try:
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) < 6:
            logger.warning(
                "nvidiasmi_parse_error",
                raw=raw,
                reason="too few fields",
            )
            return None

        def extract_float(s: str) -> float:
            m = _NUMBER_RE.search(s)
            if not m:
                raise ValueError(f"No number found in: {s!r}")
            return float(m.group())

        return {
            "vram_used_bytes":     int(extract_float(parts[0]) * MIB_TO_BYTES),
            "vram_total_bytes":    int(extract_float(parts[1]) * MIB_TO_BYTES),
            "vram_free_bytes":     int(extract_float(parts[2]) * MIB_TO_BYTES),
            "utilization_ratio":   extract_float(parts[3]) / 100.0,
            "temperature_celsius": extract_float(parts[4]),
            "power_watts":         extract_float(parts[5]),
        }

    except (ValueError, IndexError) as exc:
        logger.warning(
            "nvidiasmi_parse_error",
            raw=raw,
            error=str(exc),
        )
        return None


def _update_gauges(parsed: dict) -> None:
    """
    Write parsed GPU metrics into Prometheus Gauges.
    Separated from parsing so unit tests can call this directly
    without needing nvidia-smi to be present.
    """
    idx = str(GPU_INDEX)
    GPU_VRAM_USED_BYTES.labels(gpu_index=idx).set(parsed["vram_used_bytes"])
    GPU_VRAM_TOTAL_BYTES.labels(gpu_index=idx).set(parsed["vram_total_bytes"])
    GPU_VRAM_FREE_BYTES.labels(gpu_index=idx).set(parsed["vram_free_bytes"])
    GPU_UTILIZATION_RATIO.labels(gpu_index=idx).set(parsed["utilization_ratio"])
    GPU_TEMPERATURE_CELSIUS.labels(gpu_index=idx).set(parsed["temperature_celsius"])
    GPU_POWER_WATTS.labels(gpu_index=idx).set(parsed["power_watts"])


# ---------------------------------------------------------------------------
# Background poll loop
# ---------------------------------------------------------------------------

_poll_task: Optional[asyncio.Task] = None


async def _poll_loop() -> None:
    """
    Background coroutine: poll nvidia-smi every POLL_INTERVAL_SECONDS.

    Logging strategy:
      - gpu_monitor_started : logged once when the loop begins
      - gpu_first_reading   : logged once when the first successful
                              nvidia-smi response is parsed. Confirms
                              the monitor is working with real data.
      - After that          : silent. Gauges update every 5 seconds
                              but produce no log output.
      - Errors/warnings     : always logged via logger.error/warning.

    Why silent after first reading?
      17,280 debug lines per day (every 5s × 60 × 60 × 24) make
      logs/gateway.jsonl unsearchable and inflate disk usage.
      Prometheus already stores the time-series data — the log does
      not need to duplicate it. If you need to audit GPU state at a
      specific moment, query /metrics or Grafana instead.

    Loop structure (sleep at END, not start):
      Poll immediately → update Gauges → sleep 5s → repeat.
      Sleeping first would leave Gauges at 0 for the first 5 seconds,
      causing the first Prometheus scrape to return empty data.
    """
    logger.info(
        "gpu_monitor_started",
        poll_interval_seconds=POLL_INTERVAL_SECONDS,
        gpu_index=GPU_INDEX,
    )

    # Flag: have we logged the first successful reading yet?
    first_poll = True

    while True:
        raw = await _run_nvidiasmi()

        if raw is not None:
            parsed = _parse_nvidiasmi_output(raw)
            if parsed is not None:
                _update_gauges(parsed)

                # Log once on first successful poll, then stay silent.
                if first_poll:
                    logger.info(
                        "gpu_first_reading",
                        vram_used_gb=round(
                            parsed["vram_used_bytes"] / 1e9, 2
                        ),
                        vram_total_gb=round(
                            parsed["vram_total_bytes"] / 1e9, 2
                        ),
                        temperature_c=parsed["temperature_celsius"],
                        power_w=round(parsed["power_watts"], 1),
                    )
                    first_poll = False

        # Yield control to the event loop for POLL_INTERVAL_SECONDS.
        # During this sleep, the event loop freely serves HTTP requests,
        # drains the log queue, and runs the session sweep.
        await asyncio.sleep(POLL_INTERVAL_SECONDS)


# ---------------------------------------------------------------------------
# Public lifecycle API
# ---------------------------------------------------------------------------

def start_gpu_monitor() -> None:
    """
    Start the background GPU polling task.
    Must be called from inside a running event loop (FastAPI lifespan).
    """
    global _poll_task
    _poll_task = asyncio.create_task(
        _poll_loop(),
        name="gpu-monitor",
    )


async def stop_gpu_monitor() -> None:
    """Cancel the polling task cleanly on shutdown."""
    global _poll_task
    if _poll_task and not _poll_task.done():
        _poll_task.cancel()
        try:
            await _poll_task
        except asyncio.CancelledError:
            pass
        logger.info("gpu_monitor_stopped")
