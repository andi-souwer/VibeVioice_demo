# VibeVoice Unified API — Reference

All endpoints are served by `api.server:app` (FastAPI). Base URL assumed
`http://localhost:8000`. OpenAPI is served at `/docs` and `/openapi.json`.

Three voice models are wrapped:

| Kind      | Model weights                               | Role                                                           |
|-----------|---------------------------------------------|----------------------------------------------------------------|
| ASR       | `microsoft/VibeVoice-ASR` (7B)              | Long-form speech recognition with speaker diarization + timestamps |
| TTS       | `microsoft/VibeVoice-1.5B`                  | Multi-speaker long-form TTS                                    |
| Realtime  | `microsoft/VibeVoice-Realtime-0.5B`         | Low-latency streaming single-speaker TTS                       |

Models are **loaded lazily** on first request and released after
`VIBEVOICE_IDLE_EVICT_SECONDS` seconds of inactivity (default `60`). Use
`POST /v1/admin/evict` to release VRAM on demand.

---

## 1. Service endpoints

### `GET /health`

Returns device info, currently-loaded model kinds, and CUDA memory.

**Response (200)**

```json
{
  "status": "ok",
  "device": "cuda",
  "cuda_available": true,
  "loaded_models": ["asr"],
  "model_paths": {
    "asr": "E:\\lenv\\llmmode\\localdown\\VibeVoice-ASR",
    "tts": "E:\\lenv\\llmmode\\localdown\\VibeVoice-1.5B",
    "realtime": "E:\\lenv\\llmmode\\localdown\\VibeVoice-Realtime-0.5B"
  },
  "cuda_memory_allocated_mb": 16601.08,
  "cuda_memory_reserved_mb": 21992.0
}
```

### `POST /v1/admin/evict?kind={asr|tts|realtime}`

Immediately releases the selected model to free VRAM.

**Response (200)**

```json
{"evicted": true, "kind": "asr"}
```

Returns `{"evicted": false, ...}` if the model was not loaded.

---

## 2. ASR — `POST /v1/asr/transcribe`

Transcribe an audio file with speaker diarization and word-level timestamps
in a single pass. Accepts audio up to ~60 minutes (ASR model's 64K-token
context).

**Request** — `multipart/form-data`

| Field            | Type   | Default | Description                                                           |
|------------------|--------|---------|-----------------------------------------------------------------------|
| `audio`          | file   | —       | Audio file (`wav`, `mp3`, `m4a`, `flac`, …). Auto-resampled by librosa. |
| `max_new_tokens` | int    | `4096`  | Upper bound on generated tokens.                                      |
| `do_sample`      | bool   | `false` | Enable stochastic decoding.                                           |
| `temperature`    | float  | `0.0`   | Used only when `do_sample=true`.                                      |
| `top_p`          | float  | `1.0`   | Used only when `do_sample=true`.                                      |
| `num_beams`      | int    | `1`     | `>1` forces `do_sample=false`.                                        |
| `hotwords`       | str    | —       | Free-form domain context / names / glossary (passed as `context_info`). |

**Response (200)** — `application/json`

```json
{
  "file": "meeting.wav",
  "raw_text": "assistant\n[{\"Start\":0.0,\"End\":9.16,\"Content\":\"[Silence]\"},{\"Start\":9.16,\"End\":48.52,\"Speaker\":0,\"Content\":\"Athens, two thousand four hundred years ago...\"},...]",
  "segments": [
    {"start_time": 0.0,   "end_time": 9.16,  "speaker_id": null, "text": "[Silence]"},
    {"start_time": 9.16,  "end_time": 48.52, "speaker_id": 0,    "text": "Athens, two thousand four hundred years ago..."},
    {"start_time": 48.52, "end_time": 66.06, "speaker_id": 0,    "text": "Thirty-six, all dialogues..."}
  ],
  "generation_time_sec": 140.44
}
```

**Structured output contract** — every entry in `segments` carries the
"who / when / what" triple:

- `speaker_id`: integer speaker index, or `null` for non-speech (`[Silence]`, `[Music]`, ...).
- `start_time` / `end_time`: seconds, monotonically non-decreasing.
- `text`: transcribed content for the segment.

`raw_text` is the exact model output (a JSON-like string); `segments` is the
parsed form produced by `VibeVoiceASRProcessor.post_process_transcription`.
If parsing fails, `segments` is an empty list and only `raw_text` is reliable.

**Errors**

| Status | Meaning                                                    |
|--------|------------------------------------------------------------|
| 400    | Empty audio payload.                                       |
| 500    | Model load or generation error (detail contains the cause).|

**Examples**

```bash
curl -F audio=@meeting.wav \
     -F hotwords="Aiden Host,Tea Brew" \
     http://localhost:8000/v1/asr/transcribe
```

```python
import requests
with open("meeting.wav", "rb") as f:
    r = requests.post(
        "http://localhost:8000/v1/asr/transcribe",
        files={"audio": f},
        data={"hotwords": "Aiden Host,Tea Brew"},
        timeout=600,
    )
for seg in r.json()["segments"]:
    print(f"[{seg['start_time']:.2f}-{seg['end_time']:.2f}] "
          f"spk={seg['speaker_id']}: {seg['text']}")
```

---

## 3. Long-form TTS — `POST /v1/tts/synthesize`

Multi-speaker (up to 4) long-form synthesis with the 1.5B model.

> **Implementation note**: The TTS `.generate()` code was removed from the
> official Microsoft/VibeVoice repository on 2025-09-05. We vendor it from
> the community fork `https://github.com/vibevoice-community/VibeVoice` at
> `VibeVoice/vibevoice/modular/modeling_vibevoice_inference.py`. The 1.5B
> model weights themselves are unchanged.

**Request** — `application/json`

```json
{
  "text": "Speaker 1: Hello.\nSpeaker 2: Hi, how are you?",
  "speakers": ["en-Carter_man", "en-Alice_woman"],
  "cfg_scale": 1.3,
  "inference_steps": 10,
  "do_sample": false,
  "temperature": 0.95,
  "top_p": 0.95
}
```

| Field             | Type        | Default             | Description |
|-------------------|-------------|---------------------|-------------|
| `text`            | str         | —                   | Either plain text (→ wrapped as `Speaker 1: …`) or lines prefixed with `Speaker N:`. |
| `speakers`        | List[str]   | `["en-Carter_man"]` | Voice preset names. Entry `i` becomes `Speaker (i+1)`. |
| `cfg_scale`       | float       | `1.3`               | Classifier-free guidance scale. |
| `inference_steps` | int         | `10`                | DDPM sampling steps. |
| `do_sample`       | bool        | `false`             | Enable stochastic LM decoding. |
| `temperature`     | float       | `0.95`              | Used when `do_sample=true`. |
| `top_p`           | float       | `0.95`              | Used when `do_sample=true`. |

List valid preset names via `GET /v1/tts/synthesize/voices`
(e.g. `en-Carter_man`, `en-Alice_woman`, `zh-Xinran_woman`, `zh-Bowen_man`, …).

**Response (200)** — `audio/wav`, mono PCM16 @ 24 kHz.

Response headers:

| Header                    | Meaning                                            |
|---------------------------|----------------------------------------------------|
| `X-Audio-Duration-Sec`    | Generated audio duration.                          |
| `X-Generation-Time-Sec`   | Wall-clock generation time (excludes model load).  |
| `X-Sample-Rate`           | Always `24000`.                                    |
| `X-Speakers`              | Comma-separated speaker preset names (echo).       |

**Errors**

| Status | Meaning                                                        |
|--------|----------------------------------------------------------------|
| 400    | Empty `text` or unknown/ambiguous speaker preset.              |
| 500    | Model load error or generation produced no audio.              |

**Example**

```bash
curl -X POST http://localhost:8000/v1/tts/synthesize \
     -H "Content-Type: application/json" \
     -d '{"text":"Speaker 1: Hello.\nSpeaker 2: Hi.","speakers":["en-Carter_man","en-Alice_woman"]}' \
     --output dialog.wav
```

### `GET /v1/tts/synthesize/voices`

```json
{"count": 9, "voices": ["en-Alice_woman", "en-Carter_man", "en-Frank_man", "en-Mary_woman_bgm", "en-Maya_woman", "in-Samuel_man", "zh-Anchen_man_bgm", "zh-Bowen_man", "zh-Xinran_woman"]}
```

---

## 4. Realtime TTS (synchronous) — `POST /v1/tts/realtime`

One-shot streaming-model synthesis. Server internally runs the streaming
generator and returns the concatenated WAV in a single response.

**Request** — `application/json`

```json
{
  "text": "Hello world.",
  "voice": "en-Carter_man",
  "cfg_scale": 1.5,
  "inference_steps": 5,
  "do_sample": false,
  "temperature": 0.9,
  "top_p": 0.9,
  "refresh_negative": true
}
```

| Field              | Type   | Default            | Description |
|--------------------|--------|--------------------|-------------|
| `text`             | str    | —                  | English script (one speaker only). |
| `voice`            | str    | `en-Carter_man`    | Realtime voice preset (see `GET /v1/voices`). Case-insensitive partial match is attempted. |
| `cfg_scale`        | float  | `1.5`              | Classifier-free guidance. |
| `inference_steps`  | int    | `5`                | DDPM steps. Lower = faster but less detail. |
| `do_sample`        | bool   | `false`            | |
| `temperature`      | float  | `0.9`              | |
| `top_p`            | float  | `0.9`              | |
| `refresh_negative` | bool   | `true`             | Re-compute negative condition each window. |

**Response (200)** — `audio/wav` (same headers as `/v1/tts/synthesize`).

**Errors**

| Status | Meaning                                                              |
|--------|----------------------------------------------------------------------|
| 400    | Ambiguous voice preset (when partial match hits multiple presets).   |
| 404    | Voice preset not found.                                              |
| 500    | Generation error (detail contains the cause).                        |

**Example**

```bash
curl -X POST http://localhost:8000/v1/tts/realtime \
     -H "Content-Type: application/json" \
     -d '{"text":"Hello world.","voice":"en-Carter_man"}' \
     --output hello.wav
```

---

## 5. Realtime TTS (streaming) — `WEBSOCKET /v1/tts/realtime/ws`

True streaming — audio chunks are emitted as the model produces them.

### Protocol

1. Client opens the socket, sends **one** text frame carrying a JSON object
   with the same schema as `POST /v1/tts/realtime`.
2. Server responds with a `start` event:

   ```json
   {"event": "start", "sample_rate": 24000, "pcm": "s16le", "channels": 1}
   ```

3. The server then alternates:
   - **binary frames** — raw PCM16 (little-endian, mono, 24 kHz).
   - **text frames** — per-chunk progress:

     ```json
     {"event": "chunk", "samples": 3840, "elapsed_sec": 0.16}
     ```

4. Final text frame on success:

   ```json
   {"event": "end", "total_samples": 118400, "duration_sec": 4.93}
   ```

5. On failure the server sends one text frame and closes:

   ```json
   {"event": "error", "message": "…"}
   ```

### Python client

```python
import asyncio, json, wave
import websockets

async def run():
    uri = "ws://localhost:8000/v1/tts/realtime/ws"
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({
            "text": "Hello world. This is a quick test.",
            "voice": "en-Carter_man",
        }))
        pcm = bytearray()
        async for msg in ws:
            if isinstance(msg, bytes):
                pcm.extend(msg)
            else:
                evt = json.loads(msg)
                print(evt)
                if evt.get("event") == "end":
                    break
                if evt.get("event") == "error":
                    raise RuntimeError(evt["message"])
    with wave.open("stream.wav", "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
        w.writeframes(bytes(pcm))

asyncio.run(run())
```

---

## 6. Voice preset enumeration

| Endpoint                       | Model       | Format | Location                                  |
|--------------------------------|-------------|--------|-------------------------------------------|
| `GET /v1/voices`               | Realtime    | `.pt`  | `VibeVoice/demo/voices/streaming_model/`  |
| `GET /v1/tts/synthesize/voices`| TTS-1.5B    | `.wav` | `VibeVoice/demo/voices/tts_model/`        |

Both return `{"count": N, "voices": [...]}` sorted alphabetically.

Realtime presets today (25): `de-Spk0_man`, `de-Spk1_woman`, `en-Carter_man`,
`en-Davis_man`, `en-Emma_woman`, `en-Frank_man`, `en-Grace_woman`,
`en-Mike_man`, `fr-Spk0_man`, `fr-Spk1_woman`, `in-Samuel_man`,
`it-Spk0_woman`, `it-Spk1_man`, `jp-Spk0_man`, `jp-Spk1_woman`,
`kr-Spk0_woman`, `kr-Spk1_man`, `nl-Spk0_man`, `nl-Spk1_woman`,
`pl-Spk0_man`, `pl-Spk1_woman`, `pt-Spk0_woman`, `pt-Spk1_man`,
`sp-Spk0_woman`, `sp-Spk1_man`.

TTS presets today (9): `en-Alice_woman`, `en-Carter_man`, `en-Frank_man`,
`en-Mary_woman_bgm`, `en-Maya_woman`, `in-Samuel_man`, `zh-Anchen_man_bgm`,
`zh-Bowen_man`, `zh-Xinran_woman`.

---

## 7. Expected performance (reference — bf16, SDPA, single RTX 4090 / A100-ish)

| Operation                 | Cold start (incl. load) | Steady-state  | VRAM (model only) |
|---------------------------|-------------------------|---------------|-------------------|
| Realtime TTS (5 steps)    | ~50 s                   | ~7× realtime  | ~6 GB             |
| TTS-1.5B (10 steps, 8 s)  | ~60 s                   | RTF ≈ 2.6×    | ~6.4 GB           |
| ASR (6.5 min audio)       | ~60 s                   | RTF ≈ 0.36    | ~22 GB            |

With `flash_attn` installed, generation latency drops further; without it,
the server silently falls back to PyTorch SDPA.

---

## 8. Error-envelope shape

All non-2xx HTTP responses return:

```json
{"detail": "<human-readable message>"}
```

WebSocket errors are delivered as `{"event": "error", "message": "..."}` on
the same socket before it closes with code `1008` (for init-time failures)
or `1000` (for normal completion).
