# VibeVoice Unified API

A FastAPI service that wraps Microsoft's three VibeVoice voice models —
**ASR-7B**, **TTS-1.5B**, and **Realtime-0.5B** — behind a single, standard
HTTP + WebSocket interface.

- **ASR** — long-form speech recognition with speaker diarization and
  timestamps, returned as structured segments (`speaker_id`, `start_time`,
  `end_time`, `text`).
- **TTS** — multi-speaker long-form synthesis (up to 4 speakers).
- **Realtime** — low-latency streaming single-speaker TTS (~300 ms first-audio
  latency), available both synchronously (`POST`) and streamed over
  WebSocket (PCM16 @ 24 kHz).

Models are loaded **lazily** on first request and released after an idle
timeout, so you can run all three from a single GPU by rotating them.

## Repository layout

```
VibeVioice_demo/
├── api/                       FastAPI service (the thing you run)
│   ├── server.py              app entry (uvicorn api.server:app)
│   ├── model_manager.py       lazy-load + LRU idle eviction
│   ├── config.py              env-var driven config
│   ├── voices.py              voice preset discovery
│   └── routers/
│       ├── asr.py             POST /v1/asr/transcribe
│       ├── tts.py             POST /v1/tts/synthesize (+ /voices)
│       └── realtime.py        POST /v1/tts/realtime + WS /v1/tts/realtime/ws
│
├── VibeVoice/                 Official Microsoft/VibeVoice checkout (editable install source)
│   ├── vibevoice/             Python package with model/processor code
│   │   └── modular/
│   │       └── modeling_vibevoice_inference.py  ← vendored from community fork
│   └── demo/voices/
│       ├── streaming_model/   .pt voice presets used by Realtime-0.5B (25 files)
│       └── tts_model/         .wav voice samples used by TTS-1.5B      (9 files)
│
├── third_party/
│   └── VibeVoiceCommunity/    Clone of vibevoice-community/VibeVoice (source of the vendored inference file)
│
├── docs/
│   └── API.md                 Endpoint reference (schemas, examples, headers)
├── test/                      Local test fixtures (audio + PDF)
├── logs/                      Runtime output (gitignorable)
├── download_mode.ipynb        Notebook that downloads the three model checkpoints
├── start_api.bat              One-click launcher (activates `pytorchst1`)
└── README.md                  (this file)
```

## Models

The server expects all three model folders to already be on disk. Default
root is `E:\lenv\llmmode\localdown\` (overridable via
`VIBEVOICE_MODEL_ROOT`). Sub-folders:

| Kind     | Path                      | HuggingFace ID                        |
|----------|---------------------------|---------------------------------------|
| ASR      | `VibeVoice-ASR/`          | `microsoft/VibeVoice-ASR`             |
| TTS      | `VibeVoice-1.5B/`         | `microsoft/VibeVoice-1.5B`            |
| Realtime | `VibeVoice-Realtime-0.5B/`| `microsoft/VibeVoice-Realtime-0.5B`   |

`download_mode.ipynb` shows how to fetch them via `huggingface_hub` with
the `hf-mirror.com` endpoint.

## Install (one time)

Requires a working CUDA-capable GPU and the conda env `pytorchst1`.

```cmd
conda activate pytorchst1

:: 1. Core Python deps
pip install -e VibeVoice              :: imports `vibevoice.*` from the sibling checkout
pip install fastapi "uvicorn[standard]" python-multipart

:: 2. Pin transformers to a version compatible with the vendored models
pip install "transformers==4.51.3"
```

> **Why `transformers==4.51.3`?** Newer releases changed the `Cache`
> layout and model `__init__` kwargs (`dtype=` became `torch_dtype=`).
> Upstream already pins this under `pyproject.toml[streamingtts]`.

If `flash-attn` isn't installed, the server automatically falls back to
PyTorch SDPA (slower but correct).

## Run

```cmd
start_api.bat
```

The script activates `pytorchst1`, sets sensible env-var defaults, and
runs `uvicorn api.server:app` on port 8000. Override any of:

| Variable                        | Default                              | Purpose                                |
|---------------------------------|--------------------------------------|----------------------------------------|
| `VIBEVOICE_MODEL_ROOT`          | `E:\lenv\llmmode\localdown`          | Parent folder of the three model dirs. |
| `VIBEVOICE_DEVICE`              | `cuda`                               | `cuda` or `cpu`.                       |
| `VIBEVOICE_IDLE_EVICT_SECONDS`  | `600`                                | Idle seconds before a loaded model is released. |
| `ASR_LANGUAGE_MODEL_NAME`       | `Qwen/Qwen2.5-7B`                    | HF tokenizer base for the ASR LM.      |
| `HF_ENDPOINT`                   | `https://hf-mirror.com`              | Used when the ASR processor pulls the Qwen tokenizer. |

Health check:

```bash
curl http://localhost:8000/health
```

Interactive OpenAPI UI: <http://localhost:8000/docs>

## Endpoints at a glance

| Method | Path                              | Purpose                                     |
|-------:|-----------------------------------|---------------------------------------------|
| GET    | `/health`                         | Device + loaded-model + VRAM info.          |
| GET    | `/v1/voices`                      | Realtime voice presets (`.pt`).             |
| GET    | `/v1/tts/synthesize/voices`       | TTS-1.5B voice presets (`.wav`).            |
| POST   | `/v1/asr/transcribe`              | Speech-to-text with who / when / what.      |
| POST   | `/v1/tts/synthesize`              | Multi-speaker long-form TTS → WAV.          |
| POST   | `/v1/tts/realtime`                | One-shot Realtime TTS → WAV.                |
| WS     | `/v1/tts/realtime/ws`             | Streaming Realtime TTS → PCM16 + events.    |
| POST   | `/v1/admin/evict?kind=…`          | Force-release ASR / TTS / Realtime model.   |

Full request/response schemas and examples: **[docs/API.md](docs/API.md)**.

## Quick examples

**ASR (hotwords, speaker diarization, timestamps in one call)**

```bash
curl -F audio=@test/@listening_with_alisher.mp3 \
     -F hotwords="Plato,Athens,Socrates" \
     http://localhost:8000/v1/asr/transcribe
```

Each returned `segments[i]` carries `speaker_id`, `start_time`, `end_time`,
`text`. Non-speech gaps come back with `speaker_id=null` and `text="[Silence]"`.

**Multi-speaker TTS**

```bash
curl -X POST http://localhost:8000/v1/tts/synthesize \
     -H "Content-Type: application/json" \
     -d '{"text":"Speaker 1: Hello.\nSpeaker 2: Hi.","speakers":["en-Carter_man","en-Alice_woman"]}' \
     --output dialog.wav
```

**Realtime streaming**

```python
import asyncio, json, wave, websockets

async def run():
    async with websockets.connect("ws://localhost:8000/v1/tts/realtime/ws") as ws:
        await ws.send(json.dumps({"text": "Hello world.", "voice": "en-Carter_man"}))
        pcm = bytearray()
        async for m in ws:
            if isinstance(m, bytes):
                pcm.extend(m)
            elif json.loads(m).get("event") == "end":
                break
    with wave.open("stream.wav", "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(bytes(pcm))

asyncio.run(run())
```

## Notes on TTS-1.5B

Microsoft removed the TTS `.generate()` implementation from the upstream
repository on 2025-09-05. We vendor it from the community fork
`https://github.com/vibevoice-community/VibeVoice` by copying the single
file
`vibevoice/modular/modeling_vibevoice_inference.py` into our
`VibeVoice/vibevoice/modular/`. The weights are unaltered.

The ASR and Realtime models continue to use the official upstream code in
`VibeVoice/vibevoice/`.

## License

- `VibeVoice/` follows the original upstream MIT license.
- `third_party/VibeVoiceCommunity/` follows that fork's license.
- Everything in `api/`, `docs/`, `start_api.bat` is provided as-is for
  internal use.
