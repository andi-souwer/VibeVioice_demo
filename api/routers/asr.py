"""POST /v1/asr/transcribe — speech recognition with VibeVoice-ASR."""
from __future__ import annotations

import os
import tempfile
import time
from typing import Any, Dict, List, Optional

import torch
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..model_manager import get_manager

router = APIRouter(prefix="/v1/asr", tags=["asr"])


@router.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(..., description="Audio file (wav/mp3/m4a/flac, etc.)"),
    max_new_tokens: int = Form(4096),
    temperature: float = Form(0.0),
    top_p: float = Form(1.0),
    do_sample: bool = Form(False),
    num_beams: int = Form(1),
    hotwords: Optional[str] = Form(
        None,
        description="Optional context_info: free-form hotwords / domain terms",
    ),
) -> Dict[str, Any]:
    """Transcribe an uploaded audio file and return structured segments."""
    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty audio payload")

    suffix = os.path.splitext(audio.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name

    try:
        loaded = get_manager().get_asr()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"ASR model load failed: {exc}")

    try:
        processor = loaded.processor
        model = loaded.model

        # Single-example processing (the processor accepts a list).
        with loaded.lock:
            proc_kwargs: Dict[str, Any] = dict(
                audio=[tmp_path],
                sampling_rate=None,
                return_tensors="pt",
                padding=True,
                add_generation_prompt=True,
            )
            if hotwords and hotwords.strip():
                # The processor forwards `context_info` into the prompt template
                # (see VibeVoiceASRProcessor._process_single_audio).
                proc_kwargs["context_info"] = hotwords.strip()

            inputs = processor(**proc_kwargs)
            device = next(model.parameters()).device
            inputs = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in inputs.items()
            }

            gen_kwargs: Dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": processor.pad_id,
                "eos_token_id": processor.tokenizer.eos_token_id,
            }
            if num_beams > 1:
                gen_kwargs["num_beams"] = num_beams
                gen_kwargs["do_sample"] = False
            else:
                gen_kwargs["do_sample"] = do_sample
                if do_sample:
                    gen_kwargs["temperature"] = temperature
                    gen_kwargs["top_p"] = top_p

            t0 = time.time()
            with torch.no_grad():
                output_ids = model.generate(**inputs, **gen_kwargs)
            elapsed = time.time() - t0

            input_length = inputs["input_ids"].shape[1]
            generated = output_ids[0, input_length:]
            eos_pos = (generated == processor.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                generated = generated[: eos_pos[0] + 1]

            raw_text = processor.decode(generated, skip_special_tokens=True)
            segments: List[Dict[str, Any]]
            try:
                segments = processor.post_process_transcription(raw_text)
            except Exception:
                segments = []

        return {
            "file": audio.filename,
            "raw_text": raw_text,
            "segments": segments,
            "generation_time_sec": elapsed,
        }
    finally:
        get_manager().mark_done("asr")
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
