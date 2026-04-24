"""FastAPI entrypoint for the VibeVoice unified API.

Run:
    uvicorn api.server:app --host 0.0.0.0 --port 8000

Env vars:
    VIBEVOICE_MODEL_ROOT         Root dir containing the three model folders
    VIBEVOICE_DEVICE             "cuda" (default) or "cpu"
    VIBEVOICE_IDLE_EVICT_SECONDS Idle seconds before a model is released (default 60)
    VIBEVOICE_API_KEY            If set, /v1/audio/* requires Bearer auth
    VIBEVOICE_CUSTOM_VOICES_DIR  Where uploaded custom voices live (default <repo>/voices_custom)
    ASR_LANGUAGE_MODEL_NAME      HF tokenizer for the ASR LM (default Qwen/Qwen2.5-7B)
    HF_ENDPOINT                  defaults to https://hf-mirror.com
"""
from __future__ import annotations

import os
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

from .config import (  # noqa: E402
    ASR_MODEL_PATH,
    DEVICE,
    REALTIME_MODEL_PATH,
    TTS_MODEL_PATH,
)
from .model_manager import get_manager  # noqa: E402
from .routers import asr as asr_router  # noqa: E402
from .routers import openai_tts as openai_tts_router  # noqa: E402
from .routers import realtime as realtime_router  # noqa: E402
from .routers import tts as tts_router  # noqa: E402
from .voices import list_voices  # noqa: E402

app = FastAPI(
    title="VibeVoice Unified API",
    version="1.0.0",
    description="Standard HTTP/WebSocket API around VibeVoice ASR-7B, TTS-1.5B and Realtime-0.5B.",
)

app.include_router(asr_router.router)
app.include_router(tts_router.router)
app.include_router(realtime_router.router)
# OpenAI-compatible /v1/audio/* endpoints for external integrations.
app.include_router(openai_tts_router.router)


@app.exception_handler(HTTPException)
async def _openai_shape_http_exc_handler(request: Request, exc: HTTPException):
    """Render HTTPException.detail as-is when routers already shaped it as
    OpenAI's `{"error": {...}}`; otherwise fall back to FastAPI's default
    `{"detail": ...}` so existing routers' error format stays untouched."""
    detail = exc.detail
    if isinstance(detail, dict) and "error" in detail:
        return JSONResponse(status_code=exc.status_code, content=detail)
    return JSONResponse(status_code=exc.status_code, content={"detail": detail})


@app.on_event("startup")
def _startup() -> None:
    get_manager()  # spin up the idle-reaper


@app.on_event("shutdown")
def _shutdown() -> None:
    get_manager().shutdown()


@app.get("/health")
def health() -> Dict[str, Any]:
    import torch

    loaded = get_manager().list_loaded()
    info: Dict[str, Any] = {
        "status": "ok",
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "loaded_models": list(loaded.keys()),
        "model_paths": {
            "asr": ASR_MODEL_PATH,
            "tts": TTS_MODEL_PATH,
            "realtime": REALTIME_MODEL_PATH,
        },
    }
    if torch.cuda.is_available():
        # PyTorch caching allocator view — what *our* process's tensors use.
        allocated_mib = torch.cuda.memory_allocated() / 1024 / 1024
        reserved_mib = torch.cuda.memory_reserved() / 1024 / 1024
        info["cuda_memory_allocated_mb"] = allocated_mib
        info["cuda_memory_reserved_mb"] = reserved_mib

        # Device-level view via NVML (torch.cuda.mem_get_info wraps
        # nvmlDeviceGetMemoryInfo). Unlike memory_allocated/reserved this
        # counts CUDA context + other processes on the same GPU, and is the
        # number you'd compare with `nvidia-smi`.
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            total_mib = total_bytes / 1024 / 1024
            used_mib = (total_bytes - free_bytes) / 1024 / 1024
            info["gpu_memory_used_mib"] = used_mib
            info["gpu_memory_total_mib"] = total_mib
            # On Windows with the default NVIDIA driver policy, PyTorch can
            # silently spill to host RAM via "CUDA Sysmem Fallback" when a
            # model doesn't fit — allocated then exceeds physical VRAM and
            # inference becomes PCIe-bound. Surface that as a clear flag so
            # callers don't have to infer it from the raw numbers.
            info["sysmem_fallback_active"] = allocated_mib > total_mib + 64
        except Exception as exc:
            info["gpu_memory_query_error"] = str(exc)
    return info


@app.get("/v1/voices")
def voices() -> Dict[str, Any]:
    presets = list_voices()
    return {"count": len(presets), "voices": sorted(presets.keys())}


@app.post("/v1/admin/evict")
def evict(kind: str) -> Dict[str, Any]:
    """Free VRAM by evicting a specific model kind: asr | tts | realtime."""
    if kind not in {"asr", "tts", "realtime"}:
        return {"evicted": False, "reason": f"unknown kind {kind!r}"}
    return {"evicted": get_manager().evict(kind), "kind": kind}
