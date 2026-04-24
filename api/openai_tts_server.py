"""Standalone OpenAI-compatible TTS API for VibeVoice-1.5B.

Intentionally self-contained. This module:
  * mounts only the `/v1/audio/*` router (OpenAI-shape)
  * never imports ASR / Realtime router code — the ASR-7B and Realtime-0.5B
    weights are not touched (neither on disk read nor VRAM alloc)
  * reuses `api.model_manager` but `model_manager` only loads a kind on first
    `get_<kind>()` call, and nothing in this process path calls anything other
    than `get_tts()`

Run::

    uvicorn api.openai_tts_server:app --host 0.0.0.0 --port 8000

See `start_openai_tts_api.bat` for a one-click launcher with the conda env
and offline-mode knobs pre-wired.

Env vars (subset of the unified api/server.py; only TTS-relevant ones apply):
    VIBEVOICE_MODEL_ROOT          parent of VibeVoice-1.5B/
    VIBEVOICE_DEVICE              "cuda" (default) or "cpu"
    VIBEVOICE_IDLE_EVICT_SECONDS  idle secs before the TTS model is released (default 60)
    VIBEVOICE_EVICT_AFTER_REQUEST 1 to release VRAM as soon as a response is sent
    VIBEVOICE_API_KEY             if set, /v1/audio/* requires Bearer auth
    VIBEVOICE_CUSTOM_VOICES_DIR   dir for uploaded custom voices (default <repo>/voices_custom)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

# Make the sibling `VibeVoice/` checkout importable without installing it.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_VIBEVOICE_DIR = _REPO_ROOT / "VibeVoice"
if _VIBEVOICE_DIR.exists() and str(_VIBEVOICE_DIR) not in sys.path:
    sys.path.insert(0, str(_VIBEVOICE_DIR))

from .config import DEVICE, TTS_MODEL_PATH  # noqa: E402
from .model_manager import get_manager  # noqa: E402
from .routers import openai_tts as openai_tts_router  # noqa: E402

app = FastAPI(
    title="VibeVoice TTS API (OpenAI-compatible)",
    version="1.0.0",
    description=(
        "Standalone text-to-speech server for VibeVoice-1.5B. "
        "Exposes `/v1/audio/speech` (OpenAI-shape) and voice management at "
        "`/v1/audio/voices`. ASR and Realtime models are never loaded."
    ),
)

app.include_router(openai_tts_router.router)


@app.exception_handler(HTTPException)
async def _openai_shape_http_exc_handler(request: Request, exc: HTTPException):
    """If a router already packaged `detail` as OpenAI's `{"error": {...}}`,
    emit it as-is; otherwise fall back to FastAPI's `{"detail": ...}`."""
    detail = exc.detail
    if isinstance(detail, dict) and "error" in detail:
        return JSONResponse(status_code=exc.status_code, content=detail)
    return JSONResponse(status_code=exc.status_code, content={"detail": detail})


@app.on_event("startup")
def _startup() -> None:
    # Only spin up the idle reaper thread. We do NOT preload TTS here — the
    # first request's model load is fast (~4s on a 4090) and lazy loading
    # keeps the process startable even if the model files are briefly
    # unavailable (e.g. mounted storage hiccup).
    get_manager()


@app.on_event("shutdown")
def _shutdown() -> None:
    get_manager().shutdown()


@app.get("/health")
def health() -> Dict[str, Any]:
    import torch

    loaded = get_manager().list_loaded()
    info: Dict[str, Any] = {
        "status": "ok",
        "service": "vibevoice-tts",
        "model": "vibevoice-1.5b",
        "model_path": TTS_MODEL_PATH,
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "tts_loaded": "tts" in loaded,
    }
    if torch.cuda.is_available():
        allocated_mib = torch.cuda.memory_allocated() / 1024 / 1024
        reserved_mib = torch.cuda.memory_reserved() / 1024 / 1024
        info["cuda_memory_allocated_mb"] = allocated_mib
        info["cuda_memory_reserved_mb"] = reserved_mib
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            total_mib = total_bytes / 1024 / 1024
            used_mib = (total_bytes - free_bytes) / 1024 / 1024
            info["gpu_memory_used_mib"] = used_mib
            info["gpu_memory_total_mib"] = total_mib
            info["sysmem_fallback_active"] = allocated_mib > total_mib + 64
        except Exception as exc:
            info["gpu_memory_query_error"] = str(exc)
    return info


@app.post("/v1/admin/evict")
def evict_tts() -> Dict[str, Any]:
    """Release VRAM by evicting the loaded TTS model. No-op if it isn't loaded."""
    return {"evicted": get_manager().evict("tts"), "kind": "tts"}
