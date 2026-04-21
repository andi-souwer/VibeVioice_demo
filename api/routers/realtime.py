"""Realtime streaming TTS (VibeVoice-Realtime-0.5B).

Exposes two endpoints:

  - POST /v1/tts/realtime       — one-shot synthesis, returns a full WAV file
  - WEBSOCKET /v1/tts/realtime/ws — streaming synthesis, sends PCM16 chunks

The WebSocket protocol is intentionally simple:

  client -> server (first message, JSON):
      {"text": "...", "voice": "en-Carter_man", "cfg_scale": 1.5,
       "inference_steps": 5, "do_sample": false, "temperature": 0.9, "top_p": 0.9}

  server -> client:
      - text frames carry JSON events ({"event": "start"|"end"|"error", ...})
      - binary frames carry raw PCM16 mono @ 24kHz
"""
from __future__ import annotations

import asyncio
import copy
import io
import json
import threading
import time
import wave
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.websockets import WebSocket, WebSocketDisconnect, WebSocketState

from ..config import SAMPLE_RATE
from ..model_manager import get_manager
from ..voices import list_voices, resolve_voice

router = APIRouter(prefix="/v1/tts", tags=["tts-realtime"])


class RealtimeRequest(BaseModel):
    text: str = Field(..., description="Text to speak.")
    voice: Optional[str] = Field(None, description="Voice preset key, e.g. 'en-Carter_man'")
    cfg_scale: float = 1.5
    inference_steps: int = 5
    do_sample: bool = False
    temperature: float = 0.9
    top_p: float = 0.9
    refresh_negative: bool = True


def _pcm16(chunk: np.ndarray) -> bytes:
    chunk = np.clip(chunk, -1.0, 1.0)
    return (chunk * 32767.0).astype(np.int16).tobytes()


def _audio_chunk_to_float32(chunk) -> np.ndarray:
    if torch.is_tensor(chunk):
        arr = chunk.detach().cpu().to(torch.float32).numpy()
    else:
        arr = np.asarray(chunk, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak > 1.0:
        arr = arr / peak
    return arr.astype(np.float32, copy=False)


def _generate_stream(
    text: str,
    voice: Optional[str],
    cfg_scale: float,
    inference_steps: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    refresh_negative: bool,
):
    """Generator yielding float32 mono audio chunks (1-D numpy arrays)."""
    from vibevoice.modular.streamer import AudioStreamer

    loaded = get_manager().get_realtime()
    processor = loaded.processor
    model = loaded.model

    voice_path = resolve_voice(voice)
    device = next(model.parameters()).device

    with loaded.lock:
        if inference_steps and inference_steps > 0:
            model.set_ddpm_inference_steps(num_steps=inference_steps)

        prefilled = torch.load(
            str(voice_path), map_location=device, weights_only=False
        )
        inputs = processor.process_input_with_cached_prompt(
            text=text.strip(),
            cached_prompt=prefilled,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        streamer = AudioStreamer(batch_size=1, stop_signal=None, timeout=None)
        errors: List[BaseException] = []
        stop_event = threading.Event()

        def _run():
            try:
                model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=cfg_scale,
                    tokenizer=processor.tokenizer,
                    generation_config={
                        "do_sample": do_sample,
                        "temperature": temperature if do_sample else 1.0,
                        "top_p": top_p if do_sample else 1.0,
                    },
                    audio_streamer=streamer,
                    stop_check_fn=stop_event.is_set,
                    verbose=False,
                    refresh_negative=refresh_negative,
                    all_prefilled_outputs=copy.deepcopy(prefilled),
                )
            except Exception as exc:
                errors.append(exc)
                streamer.end()

        worker = threading.Thread(target=_run, daemon=True)
        worker.start()

        try:
            for raw_chunk in streamer.get_stream(0):
                yield _audio_chunk_to_float32(raw_chunk)
        finally:
            stop_event.set()
            streamer.end()
            worker.join(timeout=5.0)
            loaded.last_used = time.time()
            if errors:
                raise errors[0]


def _wav_bytes(samples: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(_pcm16(samples))
    return buf.getvalue()


@router.post("/realtime")
def synthesize_realtime(req: RealtimeRequest):
    """Synchronous call: return a full WAV."""
    try:
        chunks: List[np.ndarray] = list(
            _generate_stream(
                req.text,
                req.voice,
                req.cfg_scale,
                req.inference_steps,
                req.do_sample,
                req.temperature,
                req.top_p,
                req.refresh_negative,
            )
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Realtime TTS failed: {exc}")

    if not chunks:
        raise HTTPException(status_code=500, detail="No audio generated")

    full = np.concatenate(chunks)
    wav = _wav_bytes(full)
    headers = {
        "X-Audio-Duration-Sec": f"{full.size / SAMPLE_RATE:.3f}",
        "X-Sample-Rate": str(SAMPLE_RATE),
    }
    return StreamingResponse(iter([wav]), media_type="audio/wav", headers=headers)


@router.websocket("/realtime/ws")
async def realtime_ws(ws: WebSocket) -> None:
    await ws.accept()
    try:
        # Expect a single JSON init message, then stream.
        raw = await ws.receive_text()
        payload: Dict[str, Any] = json.loads(raw)
        req = RealtimeRequest(**payload)
    except (WebSocketDisconnect, json.JSONDecodeError, Exception) as exc:
        if ws.application_state == WebSocketState.CONNECTED:
            try:
                await ws.send_text(json.dumps({"event": "error", "message": str(exc)}))
            except Exception:
                pass
            await ws.close(code=1008)
        return

    await ws.send_text(json.dumps({
        "event": "start",
        "sample_rate": SAMPLE_RATE,
        "pcm": "s16le",
        "channels": 1,
    }))

    loop = asyncio.get_running_loop()

    async def pump() -> None:
        def _blocking_gen():
            return _generate_stream(
                req.text, req.voice, req.cfg_scale, req.inference_steps,
                req.do_sample, req.temperature, req.top_p, req.refresh_negative,
            )

        # Run the blocking generator in a thread and bridge chunks into asyncio.
        q: asyncio.Queue = asyncio.Queue()
        SENTINEL = object()

        def _drain():
            try:
                for ch in _blocking_gen():
                    asyncio.run_coroutine_threadsafe(q.put(ch), loop)
            except Exception as exc:
                asyncio.run_coroutine_threadsafe(q.put(exc), loop)
            finally:
                asyncio.run_coroutine_threadsafe(q.put(SENTINEL), loop)

        worker = threading.Thread(target=_drain, daemon=True)
        worker.start()

        total = 0
        while True:
            item = await q.get()
            if item is SENTINEL:
                break
            if isinstance(item, BaseException):
                await ws.send_text(json.dumps({"event": "error", "message": str(item)}))
                return
            await ws.send_bytes(_pcm16(item))
            total += int(item.size)
            await ws.send_text(json.dumps({
                "event": "chunk",
                "samples": int(item.size),
                "elapsed_sec": total / SAMPLE_RATE,
            }))

        await ws.send_text(json.dumps({
            "event": "end",
            "total_samples": total,
            "duration_sec": total / SAMPLE_RATE,
        }))

    try:
        await pump()
    except WebSocketDisconnect:
        return
    except Exception as exc:
        if ws.application_state == WebSocketState.CONNECTED:
            try:
                await ws.send_text(json.dumps({"event": "error", "message": str(exc)}))
            except Exception:
                pass
    finally:
        if ws.application_state == WebSocketState.CONNECTED:
            await ws.close()


@router.get("/voices")
def voices() -> Dict[str, Any]:
    presets = list_voices()
    return {"count": len(presets), "voices": sorted(presets.keys())}
