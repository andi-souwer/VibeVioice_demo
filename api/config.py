"""Configuration for the VibeVoice API."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VIBEVOICE_DIR = REPO_ROOT / "VibeVoice"

MODEL_ROOT = Path(os.environ.get("VIBEVOICE_MODEL_ROOT", r"E:\lenv\llmmode\localdown"))

ASR_MODEL_PATH = str(MODEL_ROOT / "VibeVoice-ASR")
TTS_MODEL_PATH = str(MODEL_ROOT / "VibeVoice-1.5B")
REALTIME_MODEL_PATH = str(MODEL_ROOT / "VibeVoice-Realtime-0.5B")

ASR_LANGUAGE_MODEL_NAME = os.environ.get("ASR_LANGUAGE_MODEL_NAME", "Qwen/Qwen2.5-7B")

VOICES_DIR = VIBEVOICE_DIR / "demo" / "voices" / "streaming_model"
TTS_VOICES_DIR = VIBEVOICE_DIR / "demo" / "voices" / "tts_model"

SAMPLE_RATE = 24_000

IDLE_EVICT_SECONDS = int(os.environ.get("VIBEVOICE_IDLE_EVICT_SECONDS", "60"))

# When set, each API call evicts its model (and frees VRAM) as soon as the
# response is produced. Default off: keep the model warm for IDLE_EVICT_SECONDS.
EVICT_AFTER_REQUEST = os.environ.get(
    "VIBEVOICE_EVICT_AFTER_REQUEST", "0"
).strip().lower() in {"1", "true", "yes", "on"}

DEVICE = os.environ.get("VIBEVOICE_DEVICE", "cuda")

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
