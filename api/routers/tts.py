"""POST /v1/tts/synthesize — multi-speaker long-form TTS (VibeVoice-1.5B).

The TTS `.generate()` implementation was removed from the official Microsoft
repository; this route uses the community-fork inference code vendored into
`VibeVoice/vibevoice/modular/modeling_vibevoice_inference.py`.

Script format expected by the processor (mirrors the community demo):

    Speaker 1: Hello there.
    Speaker 2: Hi, how are you?
    Speaker 1: I'm fine.
    ...

`speakers` lists the voice preset names (by index -> Speaker 1, 2, ...).
If `text` has no "Speaker N:" prefix, the server wraps it as `Speaker 1: <text>`.
"""
from __future__ import annotations

import io
import re
import time
import wave
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..config import SAMPLE_RATE
from ..model_manager import get_manager
from ..voices import list_tts_voices, resolve_tts_voice

router = APIRouter(prefix="/v1/tts", tags=["tts"])

_SPEAKER_LINE = re.compile(r"^Speaker\s+\d+\s*:", re.IGNORECASE | re.MULTILINE)


class TTSRequest(BaseModel):
    text: str = Field(..., description="Script. Either plain text, or 'Speaker N: ...' lines.")
    speakers: Optional[List[str]] = Field(
        None,
        description="Voice preset names; entry i maps to Speaker (i+1). Defaults to ['en-Carter_man'].",
    )
    cfg_scale: float = 1.3
    inference_steps: int = 10
    do_sample: bool = False
    temperature: float = 0.95
    top_p: float = 0.95


def _normalise_script(text: str, n_speakers: int) -> str:
    text = text.replace("\u2019", "'").strip()
    if not _SPEAKER_LINE.search(text):
        text = "Speaker 1: " + text
    return text


def _pcm16_bytes(samples: np.ndarray) -> bytes:
    samples = np.clip(samples, -1.0, 1.0)
    return (samples * 32767.0).astype(np.int16).tobytes()


def _wav_bytes(samples: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(_pcm16_bytes(samples))
    return buf.getvalue()


@router.post("/synthesize")
def synthesize(req: TTSRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    speakers = req.speakers or ["en-Carter_man"]
    try:
        voice_paths = [str(resolve_tts_voice(name)) for name in speakers]
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    script = _normalise_script(text, len(speakers))

    try:
        loaded = get_manager().get_tts()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TTS model load failed: {exc}")

    t0 = time.time()
    try:
        processor = loaded.processor
        model = loaded.model

        with loaded.lock:
            if req.inference_steps and req.inference_steps > 0:
                model.set_ddpm_inference_steps(num_steps=req.inference_steps)

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
                cfg_scale=req.cfg_scale,
                tokenizer=processor.tokenizer,
                generation_config={
                    "do_sample": req.do_sample,
                    "temperature": req.temperature if req.do_sample else 1.0,
                    "top_p": req.top_p if req.do_sample else 1.0,
                },
                verbose=False,
                is_prefill=True,
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {exc}")
    finally:
        get_manager().mark_done("tts")

    speech = outputs.speech_outputs
    if not speech or speech[0] is None:
        raise HTTPException(status_code=500, detail="No audio produced by TTS model")

    audio = speech[0]
    if torch.is_tensor(audio):
        audio = audio.detach().cpu().to(torch.float32).numpy()
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak

    wav = _wav_bytes(audio)
    headers = {
        "X-Audio-Duration-Sec": f"{audio.size / SAMPLE_RATE:.3f}",
        "X-Generation-Time-Sec": f"{time.time() - t0:.3f}",
        "X-Sample-Rate": str(SAMPLE_RATE),
        "X-Speakers": ",".join(speakers),
    }
    return StreamingResponse(iter([wav]), media_type="audio/wav", headers=headers)


@router.get("/synthesize/voices")
def list_tts() -> Dict[str, Any]:
    presets = list_tts_voices()
    return {"count": len(presets), "voices": sorted(presets.keys())}
