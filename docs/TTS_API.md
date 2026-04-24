# VibeVoice TTS API

Standalone, OpenAI-compatible text-to-speech service backed by
**VibeVoice-1.5B**. This is the contract external callers should integrate
against — it is independent of the ASR and Realtime services and does not
depend on their weights being installed.

- **Server module**: `api.openai_tts_server:app` (FastAPI + uvicorn)
- **One-click start**: `start_openai_tts_api.bat`
- **Default port**: `8000`
- **Implementation**: `api/routers/openai_tts.py`, `api/voice_store.py`
- **Test harness**: `test/test_openai_tts_api.py` (54 checks, ~100s end-to-end)

---

## Quick start

```cmd
REM Launch (defaults: host 0.0.0.0, port 8000, no auth)
start_openai_tts_api.bat

REM Stop
stop_demos.bat
```

Environment knobs honored by the launcher (all optional):

| Env var | Default | Purpose |
|---|---|---|
| `VIBEVOICE_MODEL_ROOT` | `E:\lenv\llmmode\localdown` | Parent folder holding `VibeVoice-1.5B/` |
| `VIBEVOICE_DEVICE` | `cuda` | `cuda` or `cpu` |
| `VIBEVOICE_IDLE_EVICT_SECONDS` | `60` | Idle seconds before the model is released from VRAM |
| `VIBEVOICE_EVICT_AFTER_REQUEST` | `0` | `1` = release VRAM after every request (tight-VRAM hosts) |
| `VIBEVOICE_API_KEY` | *(unset)* | If set, every `/v1/audio/*` call must carry `Authorization: Bearer <value>` |
| `VIBEVOICE_CUSTOM_VOICES_DIR` | `<repo>/voices_custom` | Where user-uploaded voice samples are persisted |
| `HF_HUB_OFFLINE` | `1` (set by launcher) | Prevent accidental HuggingFace fetches |

---

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/audio/speech` | Synthesize audio (OpenAI-shape request) |
| `GET`  | `/v1/audio/voices` | List available voices (aliases / built-in / custom) |
| `POST` | `/v1/audio/voices` | Upload a voice sample for cloning (multipart/form-data) |
| `DELETE` | `/v1/audio/voices/{id}` | Remove a custom voice by id |
| `GET`  | `/health` | Liveness + VRAM state + `sysmem_fallback_active` flag |
| `POST` | `/v1/admin/evict` | Force TTS model out of VRAM (no-op if not loaded) |

The OpenAI base URL for this server is **`http://<host>:8000/v1`** — pass it as
`base_url` to the official `openai` SDK.

---

## `POST /v1/audio/speech`

### Request (JSON)

OpenAI-standard fields (quoted from OpenAI's Audio API):

| Field | Type | Default | Notes |
|---|---|---|---|
| `model` | string | `"vibevoice-1.5b"` | Accepted for OpenAI compatibility. The server always uses VibeVoice-1.5B regardless of value. |
| `input` | string | **required** | Text to synthesize. Multi-speaker: write `Speaker 1: ...` / `Speaker 2: ...` on their own lines. Plain text without a `Speaker N:` prefix is auto-wrapped as `Speaker 1:`. |
| `voice` | string | **required** | OpenAI alias (`alloy`/`echo`/...), VibeVoice preset (`en-Carter_man`), or custom voice id (`vv-<hex>`). |
| `response_format` | string | `"mp3"` | `wav` / `mp3` / `flac` / `opus` / `aac` / `pcm` |
| `speed` | float | `1.0` | `0.25` – `4.0`. Uses `librosa.effects.time_stretch` (keeps pitch). |
| `stream_format` | string | `"audio"` | Only `"audio"` is supported (chunked binary). `"sse"` is not implemented. |
| `instructions` | string | — | Accepted and **ignored** — VibeVoice-1.5B has no instruction-style prompting. Kept for OpenAI client compatibility. |

VibeVoice-specific extensions (OpenAI SDKs pass them through `extra_body`):

| Field | Type | Default | Notes |
|---|---|---|---|
| `vv_speakers` | string[] | — | One voice per speaker; entry `i` maps to `Speaker (i+1)`. Overrides `voice` when present. |
| `vv_cfg_scale` | float | `1.3` | Classifier-free guidance strength |
| `vv_inference_steps` | int | `10` | DDPM denoising steps |
| `vv_do_sample` | bool | `false` | `false` = greedy; `true` enables sampling |
| `vv_temperature` | float | `0.95` | Only applied when `vv_do_sample=true` |
| `vv_top_p` | float | `0.95` | Only applied when `vv_do_sample=true` |
| `vv_seed` | int | `-1` | `>= 0` fixes the seed (`transformers.set_seed`), `-1` = random |

Unknown fields are tolerated (`extra="allow"` on the pydantic model), so newer
OpenAI SDK versions that add fields won't break the server.

### Response

Binary audio in the requested format. The server sets a format-matched
`Content-Type` and adds a few informational headers.

| Header | Value |
|---|---|
| `Content-Type` | `audio/wav` · `audio/mpeg` · `audio/flac` · `audio/opus` · `audio/aac` · `application/octet-stream` (for `pcm`) |
| `Content-Disposition` | `attachment; filename="speech.<ext>"` |
| `X-Audio-Duration-Sec` | Generated audio duration (after `speed` applied) |
| `X-Generation-Time-Sec` | Wall-clock generation time |
| `X-Sample-Rate` | Always `24000` |
| `X-Speakers` | Comma-separated list of voices that were used (echoes what callers passed) |

`pcm` is raw signed 16-bit little-endian mono at 24 kHz — no container, no
headers.

### Voice resolution order

`voice` (and each entry of `vv_speakers`) is resolved in this order:

1. **OpenAI alias** — mapped to a VibeVoice preset (see table below).
2. **Custom voice id** (`vv-<hex>`) — read from the uploaded voice store.
3. **VibeVoice preset name** (e.g. `en-Carter_man`, `zh-Bowen_man`).

If none match, the server returns `400 invalid_request_error` with
`code: "unknown_voice"`.

### OpenAI alias → VibeVoice preset

| OpenAI alias | VibeVoice preset |
|---|---|
| `alloy`, `fable`, `verse` | `en-Carter_man` |
| `echo`, `onyx` | `en-Frank_man` |
| `nova`, `coral` | `en-Alice_woman` |
| `shimmer`, `sage`, `ballad` | `en-Maya_woman` |
| `ash` | `in-Samuel_man` |

To bypass the alias layer, pass a preset name or custom id directly.

---

## `GET /v1/audio/voices`

Lists every voice callers can pass as `voice` or inside `vv_speakers`.

```json
{
  "aliases": ["alloy", "ash", "ballad", "coral", "echo", "fable",
              "nova", "onyx", "sage", "shimmer", "verse"],
  "builtin": ["en-Alice_woman", "en-Carter_man", "en-Frank_man",
              "en-Mary_woman_bgm", "en-Maya_woman", "in-Samuel_man",
              "zh-Anchen_man_bgm", "zh-Bowen_man", "zh-Xinran_woman"],
  "custom": [
    {
      "id": "vv-4d00893a4b1f",
      "name": "boss-voice",
      "uploaded_filename": "boss_original.wav",
      "created": 1777004200
    }
  ]
}
```

---

## `POST /v1/audio/voices` — register a custom voice

`multipart/form-data`:

| Field | Required | Notes |
|---|---|---|
| `file` | yes | A voice sample. Supported file types: `.wav`, `.flac`, `.mp3`, `.m4a`, `.ogg` |
| `name` | no | Human-readable label. Defaults to the uploaded filename. |

Response:

```json
{
  "id": "vv-4d00893a4b1f",
  "name": "boss-voice",
  "uploaded_filename": "boss_original.wav",
  "created": 1777004200
}
```

Persist the returned `id` — pass it back as `voice` (or inside `vv_speakers`)
to synthesize with the uploaded voice.

The raw file is written under `VIBEVOICE_CUSTOM_VOICES_DIR` alongside a
`<id>.meta.json` manifest; deleting either the file or the directory removes
the voice.

---

## `DELETE /v1/audio/voices/{id}`

Removes the file and its manifest. Returns `404` when the id is unknown.
Only ids with the `vv-` prefix are accepted here — built-in presets can't be
deleted through the API (and won't be: they're on disk under
`VibeVoice/demo/voices/tts_model/`).

```json
{"deleted": true, "id": "vv-4d00893a4b1f"}
```

---

## `GET /health`

Purely informational, useful for readiness probes and OOM debugging:

```json
{
  "status": "ok",
  "service": "vibevoice-tts",
  "model": "vibevoice-1.5b",
  "model_path": "E:\\lenv\\llmmode\\localdown\\VibeVoice-1.5B",
  "device": "cuda",
  "cuda_available": true,
  "tts_loaded": false,
  "cuda_memory_allocated_mb": 0.0,
  "cuda_memory_reserved_mb": 0.0,
  "gpu_memory_used_mib": 1571.5,
  "gpu_memory_total_mib": 24563.5,
  "sysmem_fallback_active": false
}
```

Use `service == "vibevoice-tts"` to distinguish a standalone TTS deployment
from the unified `api.server` deployment (which reports no `service` field).

`sysmem_fallback_active` flips to `true` if PyTorch's allocator claims more
than the card's physical VRAM — a sign the NVIDIA driver's sysmem fallback
kicked in and inference will be PCIe-bound. Treat it as an alert.

---

## Authentication

Disabled by default (local dev). Enable by setting `VIBEVOICE_API_KEY` **in
the server's environment**, then clients must send:

```
Authorization: Bearer <VIBEVOICE_API_KEY>
```

on every `/v1/audio/*` request.

| Scenario | Status | Body |
|---|---|---|
| `VIBEVOICE_API_KEY` unset | any call → 200 | normal |
| Set, header missing / wrong scheme | **401** | `error.code = "missing_authorization"` |
| Set, header has wrong bearer | **401** | `error.code = "invalid_authorization"` |
| Set, header matches | 200 | normal |

---

## Error format

Every `4xx` / `5xx` response uses OpenAI's error envelope:

```json
{
  "error": {
    "message": "Script references 3 speakers but only 2 voice(s) supplied. Pass `vv_speakers` with one entry per speaker.",
    "type": "invalid_request_error",
    "param": "vv_speakers",
    "code": null
  }
}
```

Common `error.code` values (the `type` is always `invalid_request_error` for
4xx and `server_error` for 5xx; `authentication_error` and `not_found_error`
are also used where they make sense):

| `code` | When |
|---|---|
| `missing_authorization` | `VIBEVOICE_API_KEY` set but header missing |
| `invalid_authorization` | Bearer token does not match `VIBEVOICE_API_KEY` |
| `invalid_value` | `response_format` or other enum out of range |
| `not_implemented` | `stream_format="sse"` (we only do `"audio"`) |
| `unknown_voice` | `voice` or any `vv_speakers` entry did not resolve |

Clients should key off `response.error.code` and treat everything else
(`message`, `param`) as human-readable debug help.

---

## Examples

### curl

```bash
# Basic synthesis → MP3 file on disk
curl -X POST http://YOUR_HOST:8000/v1/audio/speech \
  -H 'Authorization: Bearer sk-your-key' \
  -H 'Content-Type: application/json' \
  -d '{"model":"vibevoice-1.5b","input":"Hello from VibeVoice.","voice":"alloy"}' \
  -o hello.mp3

# Multi-speaker podcast, WAV output, fixed seed
curl -X POST http://YOUR_HOST:8000/v1/audio/speech \
  -H 'Authorization: Bearer sk-your-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "Speaker 1: Welcome to the show.\nSpeaker 2: Glad to be here.",
    "voice": "alloy",
    "response_format": "wav",
    "vv_speakers": ["alloy", "nova"],
    "vv_seed": 42
  }' \
  -o podcast.wav

# List voices
curl http://YOUR_HOST:8000/v1/audio/voices

# Upload a custom voice sample, then synthesize with it
curl -X POST http://YOUR_HOST:8000/v1/audio/voices \
  -H 'Authorization: Bearer sk-your-key' \
  -F 'file=@/path/to/voice_sample.wav' \
  -F 'name=boss-voice'
# → {"id": "vv-abc123...", "name": "boss-voice", ...}

curl -X POST http://YOUR_HOST:8000/v1/audio/speech \
  -H 'Authorization: Bearer sk-your-key' \
  -H 'Content-Type: application/json' \
  -d '{"input":"Cloned voice test","voice":"vv-abc123..."}' \
  -o cloned.mp3
```

### Python — `openai` SDK (canonical client)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://YOUR_HOST:8000/v1",
    api_key="sk-your-key",         # any string if auth is disabled
)

# Plain call — looks like standard OpenAI TTS
r = client.audio.speech.create(
    model="vibevoice-1.5b",
    voice="nova",
    input="Speech generated via the OpenAI Python SDK.",
)
r.write_to_file("out.mp3")

# With VibeVoice extensions via extra_body
r = client.audio.speech.create(
    model="vibevoice-1.5b",
    voice="alloy",
    input="Speaker 1: Morning!\nSpeaker 2: Morning. Coffee?",
    response_format="wav",
    speed=1.1,
    extra_body={
        "vv_speakers": ["alloy", "nova"],
        "vv_cfg_scale": 1.4,
        "vv_inference_steps": 12,
        "vv_seed": 42,
    },
)
open("duet.wav", "wb").write(r.content)
```

### Python — plain `requests`

```python
import requests

r = requests.post(
    "http://YOUR_HOST:8000/v1/audio/speech",
    headers={"Authorization": "Bearer sk-your-key"},
    json={
        "model": "vibevoice-1.5b",
        "input": "你好，这是中文合成测试。",
        "voice": "zh-Bowen_man",
        "response_format": "wav",
        "vv_seed": 7,
    },
    timeout=120,
)
r.raise_for_status()
print(
    "duration:", r.headers["X-Audio-Duration-Sec"], "s",
    "| generated in:", r.headers["X-Generation-Time-Sec"], "s",
)
open("out.wav", "wb").write(r.content)
```

### Node.js — `openai` SDK

```js
import OpenAI from "openai";
import fs from "node:fs";

const client = new OpenAI({
  baseURL: "http://YOUR_HOST:8000/v1",
  apiKey: "sk-your-key",
});

const r = await client.audio.speech.create({
  model: "vibevoice-1.5b",
  voice: "shimmer",
  input: "Hello from Node.js.",
  response_format: "mp3",
  // @ts-expect-error — vv_* are VibeVoice extensions beyond the OpenAI type
  vv_seed: 42,
  vv_cfg_scale: 1.3,
});
fs.writeFileSync("out.mp3", Buffer.from(await r.arrayBuffer()));
```

---

## Operational notes

- **Single-GPU, single-model, serialized generation.** The model is loaded
  lazily on the first request (~4 s for the 1.5B model on a 4090). All
  `generate()` calls are serialized through a per-model `threading.Lock`, so
  concurrent requests queue but never corrupt the model's state.
- **Memory footprint.** Post-load, PyTorch holds ~7 GB on the GPU (bf16
  weights). Expect nvidia-smi to report ~8.5 GB total (weights + CUDA
  context). The ASR-7B and Realtime-0.5B models are never loaded by this
  server.
- **Idle eviction** (default 60 s). If no request arrives for
  `VIBEVOICE_IDLE_EVICT_SECONDS`, the model is moved to CPU and the GPU
  cache is released. The next request reloads it — cold path is ~4 s + your
  usual synthesis time. Override with the env var if you want it warm for
  longer.
- **Aggressive release** — set `VIBEVOICE_EVICT_AFTER_REQUEST=1` to evict
  after every response. Use on tight-VRAM hosts that share the GPU with
  other workloads.
- **`sysmem_fallback_active` in `/health`** — if this ever reads `true` in
  production, something else on the card is crowding VibeVoice. Investigate
  with `nvidia-smi` before trusting generation latency.
- **Concurrency model.** FastAPI + uvicorn workers are threads; the
  generation lock is in-process. Scale horizontally with multiple uvicorn
  processes (different ports, different GPUs) rather than more threads — a
  second thread just waits on the lock.

---

## What is *not* supported (and why)

| Feature | Status | Reason |
|---|---|---|
| `stream_format="sse"` — Server-Sent Events with base64 audio deltas | Returns `400 not_implemented` | Deliberately out of scope. The `AudioStreamer` primitive exists in the SDK; ask and it can be wired up. |
| `instructions` field from `gpt-4o-mini-tts` | Accepted, ignored | VibeVoice-1.5B has no instruction-style prompting. We surface neither 4xx nor log noise so OpenAI clients keep working. |
| `negative_prompt_ids` (advanced CFG) | Not exposed | The upstream model default (`speech_start_id` as negative prompt) matches what the official demos ship. Overriding usually degrades output. |
| `vv_max_new_tokens` cap | Not exposed | VibeVoice emits `<|speech_end|>` on its own when the script ends. Capping early truncates audio; the config's `max_position_embeddings=131072` is already the hard ceiling. |

---

## Verifying a deployment

Use the bundled test harness — 54 checks across every feature, ~100 s wall.

```cmd
start_openai_tts_api.bat                :: server on :8000
python test\test_openai_tts_api.py --host http://127.0.0.1:8000
```

Exit code `0` = all green. On failure, the script lists each failing case
with the raw server response so you can grep it against this doc.
