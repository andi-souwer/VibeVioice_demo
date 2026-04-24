"""OpenAI-compatible TTS API for VibeVoice-1.5B.

Endpoints (prefix `/v1/audio`, mounted under the same FastAPI app as the
internal `/v1/tts/synthesize`):

    POST   /v1/audio/speech          OpenAI-shape text-to-speech
    GET    /v1/audio/voices          list built-in voices, aliases, custom
    POST   /v1/audio/voices          register a custom voice (multipart upload)
    DELETE /v1/audio/voices/{id}     remove a custom voice

Compatible with the official `openai` Python SDK::

    from openai import OpenAI
    c = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-local")
    r = c.audio.speech.create(model="vibevoice-1.5b", voice="alloy",
                              input="Hello, world!")
    r.write_to_file("speech.mp3")

VibeVoice-specific features that don't exist in OpenAI's TTS are exposed via
optional `vv_*` fields (OpenAI's API silently drops unknown fields, so existing
SDKs keep working):

    vv_speakers:        list[str]   overrides `voice`; one entry per Speaker N
    vv_cfg_scale:       float       CFG strength (default 1.3)
    vv_inference_steps: int         DDPM steps (default 10)
    vv_do_sample:       bool        greedy vs sample (default false)
    vv_temperature:     float       only if do_sample
    vv_top_p:           float       only if do_sample
    vv_seed:            int         >=0 fixes the seed, -1 random

Auth: if `VIBEVOICE_API_KEY` is set in the server env, every request must
carry `Authorization: Bearer <that-key>`; otherwise auth is skipped.
"""
from __future__ import annotations

import io
import os
import re
import time
import wave
from typing import List, Optional

import numpy as np
import torch
from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field

from ..config import SAMPLE_RATE
from ..model_manager import get_manager
from ..voice_store import get_voice_store
from ..voices import list_tts_voices, resolve_tts_voice

router = APIRouter(prefix="/v1/audio", tags=["openai-tts"])


# -- OpenAI voice alias -> VibeVoice preset -----------------------------------
# Picked to match OpenAI's published voice personas as closely as our 9 presets
# allow. Users can bypass the alias by passing a VibeVoice preset name directly
# (e.g. "en-Carter_man") or a custom voice id (e.g. "vv-abc...").
_VOICE_ALIAS = {
    "alloy":   "en-Carter_man",
    "echo":    "en-Frank_man",
    "fable":   "en-Carter_man",
    "onyx":    "en-Frank_man",
    "nova":    "en-Alice_woman",
    "shimmer": "en-Maya_woman",
    "ash":     "in-Samuel_man",
    "coral":   "en-Alice_woman",
    "sage":    "en-Maya_woman",
    "ballad":  "en-Maya_woman",
    "verse":   "en-Carter_man",
}

_SUPPORTED_FORMATS = {"wav", "mp3", "flac", "opus", "aac", "pcm"}
_CONTENT_TYPES = {
    "mp3":  "audio/mpeg",
    "opus": "audio/opus",
    "aac":  "audio/aac",
    "flac": "audio/flac",
    "wav":  "audio/wav",
    "pcm":  "application/octet-stream",
}

_SPEAKER_LINE = re.compile(r"^Speaker\s+\d+\s*:", re.IGNORECASE | re.MULTILINE)
_SPEAKER_INDEX = re.compile(r"Speaker\s+(\d+)\s*:", re.IGNORECASE)


# -- Auth dependency ----------------------------------------------------------
def _require_bearer(authorization: Optional[str] = Header(default=None)) -> None:
    """Enforce bearer auth iff VIBEVOICE_API_KEY is set. Uses OpenAI error shape."""
    required = os.environ.get("VIBEVOICE_API_KEY")
    if not required:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "Missing bearer token.",
                    "type": "authentication_error",
                    "param": None,
                    "code": "missing_authorization",
                }
            },
        )
    if authorization[len("Bearer "):] != required:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "Invalid bearer token.",
                    "type": "authentication_error",
                    "param": None,
                    "code": "invalid_authorization",
                }
            },
        )


def _error(status: int, message: str, *, type_: str = "invalid_request_error",
           param: Optional[str] = None, code: Optional[str] = None) -> HTTPException:
    """Raise-able HTTPException that the global handler will render as OpenAI shape."""
    return HTTPException(
        status_code=status,
        detail={"error": {"message": message, "type": type_, "param": param, "code": code}},
    )


# -- Helpers ------------------------------------------------------------------
def _normalise_script(text: str) -> str:
    t = (text or "").replace("’", "'").replace("“", '"').replace("”", '"').strip()
    if not t:
        raise ValueError("input must be a non-empty string")
    if not _SPEAKER_LINE.search(t):
        t = "Speaker 1: " + t
    return t


def _count_speakers_in_script(text: str) -> int:
    nums = {int(m.group(1)) for m in _SPEAKER_INDEX.finditer(text)}
    return max(len(nums), 1)


def _resolve_voice_to_path(name: str) -> str:
    """Resolve an OpenAI alias, VibeVoice preset name, or custom voice id to a file path."""
    if not name:
        raise ValueError("voice name is empty")
    if name in _VOICE_ALIAS:
        name = _VOICE_ALIAS[name]
    # Custom-registered voice?
    custom_path = get_voice_store().get_path(name)
    if custom_path is not None:
        return str(custom_path)
    # Built-in preset?
    try:
        return str(resolve_tts_voice(name))
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(str(exc))


def _apply_speed(samples: np.ndarray, speed: float) -> np.ndarray:
    """OpenAI `speed` param: time-stretch WITHOUT changing pitch (librosa)."""
    if abs(speed - 1.0) < 1e-3 or samples.size == 0:
        return samples
    try:
        import librosa  # noqa: WPS433
        return librosa.effects.time_stretch(samples.astype(np.float32), rate=float(speed))
    except Exception:
        # Fallback: linear resample (changes pitch, but better than failing).
        n_out = max(1, int(samples.size / float(speed)))
        idx = np.linspace(0, samples.size - 1, n_out)
        return np.interp(idx, np.arange(samples.size), samples).astype(np.float32)


def _pcm16(samples: np.ndarray) -> np.ndarray:
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)


def _encode_audio(samples: np.ndarray, fmt: str) -> bytes:
    """Encode float32 mono audio into the requested container."""
    fmt = fmt.lower()
    pcm = _pcm16(samples)
    if fmt == "wav":
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(pcm.tobytes())
        return buf.getvalue()
    if fmt == "pcm":
        return pcm.tobytes()
    if fmt == "flac":
        import soundfile as sf  # noqa: WPS433
        buf = io.BytesIO()
        sf.write(buf, pcm, SAMPLE_RATE, format="FLAC", subtype="PCM_16")
        return buf.getvalue()
    if fmt in {"mp3", "opus", "aac"}:
        try:
            from pydub import AudioSegment  # noqa: WPS433
        except ImportError as exc:
            raise RuntimeError(
                f"pydub is required for response_format={fmt!r}; install pydub + ffmpeg"
            ) from exc
        seg = AudioSegment(
            data=pcm.tobytes(),
            sample_width=2,
            frame_rate=SAMPLE_RATE,
            channels=1,
        )
        buf = io.BytesIO()
        kwargs: dict = {}
        if fmt == "mp3":
            kwargs = {"format": "mp3", "bitrate": "128k"}
        elif fmt == "opus":
            kwargs = {"format": "opus", "bitrate": "64k"}
        elif fmt == "aac":
            # pydub's `format='aac'` maps to ffmpeg `-f aac`, which ffmpeg
            # refuses; real .aac files on disk are ADTS-framed raw AAC.
            kwargs = {"format": "adts", "bitrate": "128k"}
        seg.export(buf, **kwargs)
        return buf.getvalue()
    raise ValueError(f"unsupported format: {fmt}")


# -- Request schema -----------------------------------------------------------
class SpeechRequest(BaseModel):
    # OpenAI recognises unknown fields by ignoring them; we do the same by
    # enabling `extra="allow"` so SDKs that add new fields won't break us.
    model_config = ConfigDict(extra="allow")

    # Standard OpenAI fields
    model: Optional[str] = Field(
        default="vibevoice-1.5b",
        description="Accepted for OpenAI compatibility; the serving model is always VibeVoice-1.5B.",
    )
    input: str = Field(..., description="Text to synthesise. Use 'Speaker N: ...' for multi-speaker.")
    voice: str = Field(..., description="OpenAI alias (alloy/echo/...), VibeVoice preset, or custom id.")
    response_format: str = Field(
        default="mp3",
        description="wav | mp3 | flac | opus | aac | pcm",
    )
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    stream_format: Optional[str] = Field(
        default="audio",
        description="'audio' (default) returns chunked audio bytes. 'sse' is not yet implemented.",
    )
    instructions: Optional[str] = Field(
        default=None,
        description="Accepted for OpenAI compatibility; ignored by VibeVoice-1.5B (no effect).",
    )

    # VibeVoice extensions
    vv_speakers: Optional[List[str]] = Field(
        default=None,
        description="Multi-speaker override; entry i maps to Speaker (i+1). Ignored if absent.",
    )
    vv_cfg_scale: float = 1.3
    vv_inference_steps: int = 10
    vv_do_sample: bool = False
    vv_temperature: float = 0.95
    vv_top_p: float = 0.95
    vv_seed: int = -1


# -- POST /v1/audio/speech ----------------------------------------------------
@router.post("/speech", dependencies=[Depends(_require_bearer)])
def speech(req: SpeechRequest):
    fmt = (req.response_format or "mp3").lower()
    if fmt not in _SUPPORTED_FORMATS:
        raise _error(
            400,
            f"Unsupported response_format {fmt!r}. Choose one of {sorted(_SUPPORTED_FORMATS)}.",
            param="response_format",
            code="invalid_value",
        )

    if req.stream_format and req.stream_format not in ("audio",):
        raise _error(
            400,
            f"stream_format={req.stream_format!r} is not supported yet; only 'audio'.",
            param="stream_format",
            code="not_implemented",
        )

    try:
        script = _normalise_script(req.input)
    except ValueError as exc:
        raise _error(400, str(exc), param="input")

    speakers_raw = req.vv_speakers or [req.voice]
    if not speakers_raw or not all(isinstance(s, str) and s for s in speakers_raw):
        raise _error(400, "voice is required", param="voice")

    # If user wrote "Speaker 2:" etc. they must also provide that many voices.
    n_in_script = _count_speakers_in_script(script)
    if n_in_script > len(speakers_raw):
        raise _error(
            400,
            (
                f"Script references {n_in_script} speakers but only {len(speakers_raw)} voice(s) "
                "supplied. Pass `vv_speakers` with one entry per speaker."
            ),
            param="vv_speakers",
        )

    try:
        voice_paths = [_resolve_voice_to_path(v) for v in speakers_raw]
    except ValueError as exc:
        raise _error(400, str(exc), param="voice", code="unknown_voice")

    try:
        loaded = get_manager().get_tts()
    except Exception as exc:
        raise _error(500, f"Model load failed: {exc}", type_="server_error")

    t0 = time.time()
    try:
        if req.vv_seed is not None and int(req.vv_seed) >= 0:
            from transformers import set_seed  # noqa: WPS433
            set_seed(int(req.vv_seed))

        processor = loaded.processor
        model = loaded.model

        with loaded.lock:
            if req.vv_inference_steps and req.vv_inference_steps > 0:
                model.set_ddpm_inference_steps(num_steps=int(req.vv_inference_steps))

            inputs = processor(
                text=[script],
                voice_samples=[voice_paths],
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            device = next(model.parameters()).device
            inputs = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in inputs.items()
            }

            outputs = model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=float(req.vv_cfg_scale),
                tokenizer=processor.tokenizer,
                generation_config={
                    "do_sample": bool(req.vv_do_sample),
                    "temperature": float(req.vv_temperature) if req.vv_do_sample else 1.0,
                    "top_p": float(req.vv_top_p) if req.vv_do_sample else 1.0,
                },
                verbose=False,
                is_prefill=True,
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise _error(500, f"Generation failed: {exc}", type_="server_error")
    finally:
        get_manager().mark_done("tts")

    speech_out = getattr(outputs, "speech_outputs", None)
    if not speech_out or speech_out[0] is None:
        raise _error(500, "Model produced no audio", type_="server_error")

    audio = speech_out[0]
    if torch.is_tensor(audio):
        audio = audio.detach().cpu().to(torch.float32).numpy()
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak

    audio = _apply_speed(audio, float(req.speed))

    try:
        data = _encode_audio(audio, fmt)
    except Exception as exc:
        raise _error(500, f"Audio encoding failed: {exc}", type_="server_error")

    elapsed = time.time() - t0
    duration = audio.size / SAMPLE_RATE
    headers = {
        "X-Audio-Duration-Sec": f"{duration:.3f}",
        "X-Generation-Time-Sec": f"{elapsed:.3f}",
        "X-Sample-Rate": str(SAMPLE_RATE),
        "X-Speakers": ",".join(speakers_raw),
        "Content-Disposition": f'attachment; filename="speech.{fmt if fmt != "pcm" else "raw"}"',
    }
    return Response(content=data, media_type=_CONTENT_TYPES[fmt], headers=headers)


# -- GET /v1/audio/voices -----------------------------------------------------
@router.get("/voices", dependencies=[Depends(_require_bearer)])
def list_voices():
    """List voice names callers can pass as `voice`:

    - `aliases`: OpenAI aliases mapped to a preset
    - `builtin`: VibeVoice-shipped presets
    - `custom`:  user-uploaded voices (`vv-*`)
    """
    return {
        "aliases": sorted(_VOICE_ALIAS.keys()),
        "builtin": sorted(list_tts_voices().keys()),
        "custom": [
            {
                "id": meta.get("id"),
                "name": meta.get("name"),
                "uploaded_filename": meta.get("uploaded_filename"),
                "created": meta.get("created"),
            }
            for meta in get_voice_store().list()
        ],
    }


# -- POST /v1/audio/voices ----------------------------------------------------
@router.post("/voices", dependencies=[Depends(_require_bearer)])
async def register_custom_voice(
    file: UploadFile = File(..., description="Voice sample file (wav/flac/mp3/m4a/ogg)"),
    name: Optional[str] = Form(default=None, description="Human-readable label"),
):
    if not file or not file.filename:
        raise _error(400, "file is required (multipart field 'file')", param="file")
    data = await file.read()
    if not data:
        raise _error(400, "empty file", param="file")
    vid = get_voice_store().register(
        data=data,
        uploaded_filename=file.filename,
        name=name or file.filename,
    )
    return {
        "id": vid,
        "name": name or file.filename,
        "uploaded_filename": file.filename,
        "created": int(time.time()),
    }


# -- DELETE /v1/audio/voices/{id} ---------------------------------------------
@router.delete("/voices/{voice_id}", dependencies=[Depends(_require_bearer)])
def delete_custom_voice(voice_id: str):
    if not voice_id.startswith("vv-"):
        raise _error(
            400,
            "Only custom voice ids (prefix 'vv-') can be deleted.",
            param="voice_id",
            code="invalid_value",
        )
    ok = get_voice_store().remove(voice_id)
    if not ok:
        raise _error(404, f"voice {voice_id!r} not found", type_="not_found_error",
                     param="voice_id", code="unknown_voice")
    return {"deleted": True, "id": voice_id}
