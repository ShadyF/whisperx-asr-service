"""Prometheus metric definitions and helpers, shared by main.py and serve_app.py."""

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    CONTENT_TYPE_LATEST,
    generate_latest,
)
import torch

REQUESTS_TOTAL = Counter(
    "whisperx_requests_total",
    "Total HTTP requests by endpoint and status",
    ["endpoint", "status"],
)
REQUEST_DURATION = Histogram(
    "whisperx_request_duration_seconds",
    "End-to-end HTTP handling time",
    ["endpoint"],
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600, 1800),
)
ACTIVE_TRANSCRIPTIONS = Gauge(
    "whisperx_active_transcriptions",
    "In-flight /asr requests",
)
LOADED_MODELS = Gauge(
    "whisperx_loaded_models",
    "Whisper models currently held in the in-process cache",
)
MODEL_EVICTIONS_TOTAL = Counter(
    "whisperx_model_evictions_total",
    "Models unloaded by the idle-eviction sweep",
    ["model"],
)
AUDIO_DURATION = Histogram(
    "whisperx_audio_duration_seconds",
    "Submitted audio duration in seconds",
    buckets=(10, 30, 60, 300, 600, 1800, 3600, 7200, 14400),
)
AUDIO_SIZE_MB = Histogram(
    "whisperx_audio_size_megabytes",
    "Submitted audio file size in MB",
    buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000),
)
VRAM_ALLOCATED_BYTES = Gauge(
    "whisperx_vram_allocated_bytes",
    "Currently allocated VRAM (CUDA only; 0 on CPU)",
)
SERVICE_INFO = Info("whisperx_service", "Static service identity")
WHISPERX_DECODE_CHUNKS_TOTAL = Counter(
    "whisperx_decode_chunks_total",
    "Native Whisper decode chunk attempts",
    ["mode"],
)
WHISPERX_DECODE_FAILURES_TOTAL = Counter(
    "whisperx_decode_failures_total",
    "Native Whisper decode chunk failures",
    ["mode"],
)

# Create the fixed native series before its first decode attempt.
WHISPERX_DECODE_CHUNKS_TOTAL.labels(mode="native")
WHISPERX_DECODE_FAILURES_TOTAL.labels(mode="native")


def refresh_vram():
    try:
        if torch.cuda.is_available():
            VRAM_ALLOCATED_BYTES.set(torch.cuda.memory_allocated())
        else:
            VRAM_ALLOCATED_BYTES.set(0)
    except Exception:
        pass


def render():
    """Return (body, content_type) for the /metrics endpoint."""
    return generate_latest(), CONTENT_TYPE_LATEST
