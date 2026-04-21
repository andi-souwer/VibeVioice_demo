# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A FastAPI server that wraps Microsoft's three VibeVoice models behind one
standard HTTP/WebSocket API:

- **ASR-7B** — long-form speech recognition with speaker diarization + timestamps
- **TTS-1.5B** — multi-speaker long-form synthesis (up to 4 speakers)
- **Realtime-0.5B** — streaming single-speaker TTS

User-facing entry point is `api.server:app`. Everything under `api/` is
our code; `VibeVoice/` is an editable-installed checkout of
microsoft/VibeVoice that we import from; `third_party/VibeVoiceCommunity/`
is a clone of the community fork kept as a reference.

## Core commands

Conda env: **`pytorchst1`**.

```cmd
:: install (once)
conda activate pytorchst1
pip install -e VibeVoice
pip install fastapi "uvicorn[standard]" python-multipart
pip install "transformers==4.51.3"          :: pinned — newer breaks the models

:: run
start_api.bat                                :: port 8000, uvicorn api.server:app

:: smoke test
curl http://localhost:8000/health
curl http://localhost:8000/v1/voices
curl http://localhost:8000/v1/tts/synthesize/voices
```

There are no automated tests in this repo. Manual verification is done by
calling the endpoints; `test/@listening_with_alisher.mp3` (+ matching
`PHILOSOPHY - PLATO.pdf`) is the reference ASR fixture.

## Non-obvious facts worth knowing before editing

### 1. `transformers` must be pinned to `4.51.3`

Both the streaming model and the vendored TTS inference code use the
pre-4.52 `Cache` API and the `torch_dtype=` kwarg (not `dtype=`). Upgrading
`transformers` breaks all three models with `IndexError: list index out of
range` inside `cache_utils.py::get_mask_sizes`, or
`TypeError: Object of type dtype is not JSON serializable` during load.

`VibeVoice/pyproject.toml` already pins this under the `streamingtts`
extra; keep it pinned for all three kinds.

### 2. TTS-1.5B code is vendored from a community fork

Microsoft removed the TTS `.generate()` implementation from the upstream
repo on 2025-09-05. We keep it alive by copying one file from the fork:

```
VibeVoice/vibevoice/modular/modeling_vibevoice_inference.py
    ← from third_party/VibeVoiceCommunity/vibevoice/modular/modeling_vibevoice_inference.py
```

The shared base (`modeling_vibevoice.py`, `modular_vibevoice_*`) in the
official repo is identical byte-for-byte to the fork (modulo a trailing
newline), so only this one inference file is the delta. If you touch any
`VibeVoice/vibevoice/modular/*.py`, re-diff both trees before assuming
they're still compatible.

### 3. ASR and TTS use different voice preset formats

| Kind      | Location                                    | Format | Used by                       |
|-----------|---------------------------------------------|--------|-------------------------------|
| Realtime  | `VibeVoice/demo/voices/streaming_model/`    | `.pt`  | `api.voices.list_voices()`    |
| TTS       | `VibeVoice/demo/voices/tts_model/`          | `.wav` | `api.voices.list_tts_voices()`|

`.pt` presets are pre-computed KV caches loaded via `torch.load`;
`.wav` presets are short audio clips consumed by the TTS processor's
voice-cloning path. Don't swap them.

### 4. ModelManager is a singleton with idle eviction

`api/model_manager.py::get_manager()` returns the process-wide
`ModelManager`. Loads each model kind at most once, protected by
`_registry_lock`. A background thread runs every 30 s and evicts any kind
untouched for more than `VIBEVOICE_IDLE_EVICT_SECONDS` (default 600) s.

When you add a new kind, also update `/v1/admin/evict`'s allow-list in
`api/server.py`.

### 5. Per-model locking rule

Every route that touches `model.generate()` must hold `loaded.lock`
(a `threading.Lock` on the `LoadedModel`). The three models themselves
are not thread-safe — concurrent `generate()` on the same model corrupts
its cache state. Different kinds can run concurrently.

### 6. Flash-attention fallback is automatic

`_load_with_fallback` in `model_manager.py` tries `flash_attention_2`
first and silently retries with `sdpa` on `ImportError`. Don't add
hard dependencies on `flash_attn`; CI/dev boxes usually don't have it.
Generation quality is slightly worse with SDPA but correct.

### 7. `sys.path` hack, not a package install

`api/server.py` prepends `<repo>/VibeVoice` to `sys.path` so `import
vibevoice.*` works without `pip install -e VibeVoice` strictly being
required. Production should still install the package; the hack exists
to keep dev iteration fast.

### 8. `HF_ENDPOINT` is defaulted to the mirror

`api/config.py` sets `os.environ.setdefault("HF_ENDPOINT",
"https://hf-mirror.com")` because the ASR processor fetches the base
Qwen tokenizer on first load. In an environment without mainland China
network restrictions, override with `set HF_ENDPOINT=https://huggingface.co`.

### 9. Model paths are hard-coded to Windows

Default is `E:\lenv\llmmode\localdown\`. When adapting to Linux or another
machine, set `VIBEVOICE_MODEL_ROOT` rather than editing `api/config.py`.

## Architecture in one diagram

```
                     ┌────────────────┐
                     │   start_api.bat│
                     └────────┬───────┘
                              │  uvicorn api.server:app
                              ▼
                     ┌────────────────┐
HTTP/WebSocket ────▶ │ FastAPI (app)  │
                     └─┬────┬───┬─────┘
                       │    │   │
       ┌───────────────┘    │   └────────────────────┐
       ▼                    ▼                        ▼
┌──────────────┐   ┌──────────────┐          ┌───────────────┐
│ routers/asr  │   │ routers/tts  │          │routers/realtime│
└───────┬──────┘   └───────┬──────┘          └───────┬────────┘
        │                  │                         │
        └──────────────────┼─────────────────────────┘
                           ▼
                 ┌─────────────────┐
                 │ ModelManager    │  singleton, threading locks,
                 │ (api/model_...) │  LRU idle eviction (30 s reaper)
                 └─┬───────┬─────┬─┘
                   │       │     │
                   ▼       ▼     ▼
         ┌─────────┐ ┌─────────┐ ┌──────────┐
         │ ASR 7B  │ │ TTS 1.5B│ │ RT 0.5B  │      loaded on demand
         │ vibevoice│ │ vibevoice│ │ vibevoice│      from VIBEVOICE_MODEL_ROOT
         │ .modular.│ │ .modular.│ │ .modular.│
         │ modeling_│ │ modeling_│ │ modeling_│
         │ vibevoice│ │ vibevoice│ │ vibevoice│
         │ _asr     │ │ _inference│ │ _streaming_inference
         └─────────┘ └─────────┘ └──────────┘
                         ↑
                  vendored from
                  third_party/VibeVoiceCommunity/
```

## Adding a new endpoint — checklist

1. Add a router under `api/routers/<name>.py` with an `APIRouter(prefix=…)`.
2. Include it in `api/server.py` with `app.include_router(...)`.
3. Acquire the model via `get_manager().get_<kind>()`; do `with loaded.lock:`
   around any `model.generate()` call.
4. Update `loaded.last_used = time.time()` when you're done so the reaper
   doesn't evict mid-request (the existing routes do this).
5. Write a curl example into `docs/API.md`.

## Adding a new model kind

1. Add `FOO_MODEL_PATH` in `api/config.py`.
2. Write `_load_foo(self)` in `api/model_manager.py`, wrap with
   `_load_with_fallback` so flash-attn missing is not fatal.
3. Expose `get_foo()` on `ModelManager`.
4. Extend the `kind in {"asr", "tts", "realtime"}` set in
   `api/server.py::evict`.
5. Update `docs/API.md` and `README.md` tables.

## Pitfalls observed in the wild

- Passing `dtype=` to the TTS/Realtime model constructor → use `torch_dtype=`.
- Loading the `.pt` streaming preset with `weights_only=True` → must be
  `weights_only=False` (it's a dict, not pure tensors).
- Wrapping text with curly quotes (`’`, `”`) — the processor chokes. The
  existing routes normalise these before calling `generate`.
- The ASR processor silently re-downloads `Qwen/Qwen2.5-7B` tokenizer on
  every fresh host. The first ASR call on a clean box takes minutes; it's
  network, not model load.
- Don't hold the per-model lock while writing uploaded audio to disk or
  doing FFmpeg work — do that before `get_manager().get_asr()`.

## Where docs live

- User-facing service docs → `README.md`
- Endpoint reference → `docs/API.md`
- Per-module internal docs → `api/README.md` (shorter, older; prefer `docs/API.md` for new work)
