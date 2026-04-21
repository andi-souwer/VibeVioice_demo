"""Voice preset discovery for the streaming (Realtime) model."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from .config import TTS_VOICES_DIR, VOICES_DIR


def list_voices() -> Dict[str, str]:
    """Return {voice_key: absolute_path} for all `.pt` voice presets (Realtime)."""
    if not VOICES_DIR.exists():
        return {}
    presets: Dict[str, str] = {}
    for pt in sorted(VOICES_DIR.rglob("*.pt")):
        presets[pt.stem] = str(pt.resolve())
    return presets


def list_tts_voices() -> Dict[str, str]:
    """Return {voice_key: absolute_path} for all `.wav` voice samples (TTS-1.5B)."""
    if not TTS_VOICES_DIR.exists():
        return {}
    presets: Dict[str, str] = {}
    for wav in sorted(TTS_VOICES_DIR.rglob("*.wav")):
        presets[wav.stem] = str(wav.resolve())
    return presets


def resolve_tts_voice(name: str) -> Path:
    presets = list_tts_voices()
    if not presets:
        raise FileNotFoundError(f"No TTS voice samples found in {TTS_VOICES_DIR}")
    if name in presets:
        return Path(presets[name])
    lname = name.lower()
    matches: List[str] = [
        k for k in presets
        if lname == k.lower()
        or lname in k.lower()
        or any(part.lower() == lname for part in k.split("-") + k.split("_"))
    ]
    if len(matches) == 1:
        return Path(presets[matches[0]])
    if len(matches) > 1:
        raise ValueError(f"Ambiguous TTS voice {name!r}; candidates: {matches}")
    raise FileNotFoundError(f"TTS voice {name!r} not found. Available: {sorted(presets)}")


def resolve_voice(name: Optional[str]) -> Path:
    presets = list_voices()
    if not presets:
        raise FileNotFoundError(f"No voice presets found in {VOICES_DIR}")

    if name:
        if name in presets:
            return Path(presets[name])
        lname = name.lower()
        matches: List[str] = [k for k in presets if lname in k.lower() or k.lower() in lname]
        if len(matches) == 1:
            return Path(presets[matches[0]])
        if len(matches) > 1:
            raise ValueError(f"Ambiguous voice name {name!r}; candidates: {matches}")

    for default in ("en-Carter_man", "en-Davis_man"):
        if default in presets:
            return Path(presets[default])
    return Path(next(iter(presets.values())))
