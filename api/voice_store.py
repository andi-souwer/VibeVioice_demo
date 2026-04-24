"""Custom voice registry for the OpenAI-compatible TTS API.

Lets callers upload a voice sample once (multipart POST /v1/audio/voices)
and then reference it by id (`vv-<hex>`) in later synthesis requests.

Persisted under `VIBEVOICE_CUSTOM_VOICES_DIR` (default `<repo>/voices_custom/`).
Per voice:
    <id>.<ext>      raw uploaded audio (the file the processor reads)
    <id>.meta.json  {"id", "name", "uploaded_filename", "audio_path", "created"}
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from .config import REPO_ROOT

_AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


class VoiceStore:
    """Thread-safe, file-backed store for user-uploaded voice samples."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def register(self, data: bytes, uploaded_filename: str, name: str) -> str:
        vid = "vv-" + uuid.uuid4().hex[:12]
        ext = os.path.splitext(uploaded_filename or "")[1].lower()
        if ext not in _AUDIO_EXTS:
            ext = ".wav"
        audio_path = self.root / f"{vid}{ext}"
        meta_path = self.root / f"{vid}.meta.json"
        with self._lock:
            audio_path.write_bytes(data)
            meta_path.write_text(
                json.dumps(
                    {
                        "id": vid,
                        "name": name,
                        "uploaded_filename": uploaded_filename,
                        "audio_path": audio_path.name,
                        "created": int(time.time()),
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        return vid

    def get_path(self, voice_id: str) -> Optional[Path]:
        """Return the on-disk audio path for a `vv-*` id, or None if unknown."""
        if not voice_id or not voice_id.startswith("vv-"):
            return None
        meta_path = self.root / f"{voice_id}.meta.json"
        if not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        audio_name = meta.get("audio_path")
        if not audio_name:
            return None
        audio_path = self.root / audio_name
        return audio_path if audio_path.exists() else None

    def list(self) -> List[Dict]:
        out: List[Dict] = []
        for meta_path in sorted(self.root.glob("*.meta.json")):
            try:
                out.append(json.loads(meta_path.read_text(encoding="utf-8")))
            except Exception:
                continue
        return out

    def remove(self, voice_id: str) -> bool:
        meta_path = self.root / f"{voice_id}.meta.json"
        if not meta_path.exists():
            return False
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        with self._lock:
            audio_name = meta.get("audio_path", f"{voice_id}.wav")
            audio_path = self.root / audio_name
            try:
                audio_path.unlink()
            except FileNotFoundError:
                pass
            meta_path.unlink()
        return True


_store: Optional[VoiceStore] = None
_store_lock = threading.Lock()


def get_voice_store() -> VoiceStore:
    """Process-wide singleton. Root dir comes from VIBEVOICE_CUSTOM_VOICES_DIR."""
    global _store
    with _store_lock:
        if _store is None:
            root = Path(
                os.environ.get(
                    "VIBEVOICE_CUSTOM_VOICES_DIR", str(REPO_ROOT / "voices_custom")
                )
            )
            _store = VoiceStore(root)
        return _store
