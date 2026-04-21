# VibeVoice Unified API

Standard HTTP + WebSocket API wrapping the three locally-downloaded models:

| Kind      | Model                        | Local path                                     |
|-----------|------------------------------|------------------------------------------------|
| ASR       | VibeVoice-ASR-7B             | `E:\lenv\llmmode\localdown\VibeVoice-ASR`      |
| TTS       | VibeVoice-TTS-1.5B           | `E:\lenv\llmmode\localdown\VibeVoice-1.5B`     |
| Realtime  | VibeVoice-Realtime-0.5B      | `E:\lenv\llmmode\localdown\VibeVoice-Realtime-0.5B` |

Models are **loaded lazily** on first request and released after
`VIBEVOICE_IDLE_EVICT_SECONDS` (default 600) seconds of inactivity.

## Launch

```cmd
:: from the repository root (E:\work\yifu\code\huggingface_demo\VibeVioice_demo)
start_api.bat
```

The script activates the `pytorchst1` conda env and runs
`uvicorn api.server:app` on port `8000`. Override via env vars before launching:

```cmd
set VIBEVOICE_MODEL_ROOT=D:\other\path
set VIBEVOICE_DEVICE=cpu
set PORT=9000   :: edit start_api.bat if you want a non-default port
start_api.bat
```

First-launch requirements (do once inside `pytorchst1`):

```cmd
conda activate pytorchst1
pip install -e VibeVoice
pip install fastapi "uvicorn[standard]" python-multipart
```

> If `flash-attn` is not installed, the server automatically falls back to
> PyTorch SDPA (slower but works).

## Endpoints

### `GET /health`

Returns device info and currently-loaded models.

### `GET /v1/voices`

Lists available voice presets for the Realtime model (scanned from
`VibeVoice/demo/voices/streaming_model/*.pt`).

### `POST /v1/asr/transcribe` — multipart/form-data

| Field             | Type   | Default | Notes                                       |
|-------------------|--------|---------|---------------------------------------------|
| `audio`           | file   | —       | wav/mp3/m4a/flac etc.                       |
| `max_new_tokens`  | int    | 4096    |                                             |
| `do_sample`       | bool   | false   |                                             |
| `temperature`     | float  | 0.0     | only used when `do_sample=true`             |
| `top_p`           | float  | 1.0     | only used when `do_sample=true`             |
| `num_beams`       | int    | 1       | >1 disables sampling                        |
| `hotwords`        | str    | null    | comma-separated hotwords / customized ctx   |

Response:

```json
{
  "file": "meeting.wav",
  "raw_text": "...",
  "segments": [
    {"speaker_id": 0, "start_time": "00:00:00", "end_time": "00:00:12", "text": "..."}
  ],
  "generation_time_sec": 4.21
}
```

Example:

```bash
curl -F audio=@meeting.wav -F hotwords="Aiden Host,Tea Brew" \
     http://localhost:8000/v1/asr/transcribe
```

### `POST /v1/tts/realtime` — JSON, returns `audio/wav`

```json
{"text": "Hello world.", "voice": "en-Carter_man", "cfg_scale": 1.5}
```

```bash
curl -X POST http://localhost:8000/v1/tts/realtime \
     -H "Content-Type: application/json" \
     -d '{"text":"Hello world.","voice":"en-Carter_man"}' \
     --output out.wav
```

Full request schema:

| Field              | Type   | Default |
|--------------------|--------|---------|
| `text`             | str    | —       |
| `voice`            | str    | en-Carter_man |
| `cfg_scale`        | float  | 1.5     |
| `inference_steps`  | int    | 5       |
| `do_sample`        | bool   | false   |
| `temperature`      | float  | 0.9     |
| `top_p`            | float  | 0.9     |
| `refresh_negative` | bool   | true    |

### `WEBSOCKET /v1/tts/realtime/ws` — streaming PCM16

Send one JSON frame with the same schema as above, then receive:

- binary frames: raw PCM16 mono @ 24000 Hz
- text frames: JSON events (`start` / `chunk` / `end` / `error`)

Minimal Python client:

```python
import asyncio, json, websockets, wave

async def run():
    uri = "ws://localhost:8000/v1/tts/realtime/ws"
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({"text": "Hello world.", "voice": "en-Carter_man"}))
        pcm = bytearray()
        async for msg in ws:
            if isinstance(msg, bytes):
                pcm.extend(msg)
            else:
                ev = json.loads(msg)
                print(ev)
                if ev.get("event") == "end":
                    break
    with wave.open("out.wav", "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
        w.writeframes(bytes(pcm))

asyncio.run(run())
```

### `POST /v1/tts/synthesize` — multi-speaker long-form TTS (VibeVoice-1.5B)

Request (JSON):

```json
{
  "text": "Speaker 1: Hello.\nSpeaker 2: Hi, how are you?",
  "speakers": ["en-Carter_man", "en-Alice_woman"],
  "cfg_scale": 1.3,
  "inference_steps": 10
}
```

- If `text` has no `Speaker N:` prefixes, the server wraps it as
  `Speaker 1: <text>`.
- `speakers[i]` is the voice preset for Speaker `i+1`. Get available presets
  from `GET /v1/tts/synthesize/voices` (distinct from the Realtime voices at
  `/v1/voices`).
- Returns `audio/wav` at 24 kHz mono; headers `X-Audio-Duration-Sec` and
  `X-Generation-Time-Sec` report metrics.

The TTS `generate()` code was removed from the official repo on 2025-09-05;
we vendor it from `https://github.com/vibevoice-community/VibeVoice` at
`VibeVoice/vibevoice/modular/modeling_vibevoice_inference.py`. The 1.5B
weights themselves are unchanged.

```bash
curl -X POST http://localhost:8000/v1/tts/synthesize \
     -H "Content-Type: application/json" \
     -d '{"text":"Speaker 1: Hello.\nSpeaker 2: Hi.","speakers":["en-Carter_man","en-Alice_woman"]}' \
     --output dialog.wav
```

### `POST /v1/admin/evict?kind=asr|tts|realtime`

Immediately releases the selected model to free VRAM.
