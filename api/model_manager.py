"""Lazy model loading + LRU idle eviction.

Holds at most one of each of the three model kinds. Loads on first request;
releases VRAM after `IDLE_EVICT_SECONDS` of no use.
"""
from __future__ import annotations

import gc
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import torch

from .config import (
    ASR_LANGUAGE_MODEL_NAME,
    ASR_MODEL_PATH,
    DEVICE,
    IDLE_EVICT_SECONDS,
    REALTIME_MODEL_PATH,
    TTS_MODEL_PATH,
)


@dataclass
class LoadedModel:
    kind: str
    model: Any
    processor: Any
    last_used: float
    lock: threading.Lock


def _pick_dtype_attn(device: str):
    if device == "cuda":
        return torch.bfloat16, "flash_attention_2"
    return torch.float32, "sdpa"


def _load_with_fallback(loader: Callable[[str], Any], preferred_attn: str):
    try:
        return loader(preferred_attn)
    except Exception as exc:
        if preferred_attn != "sdpa":
            print(f"[model_manager] {preferred_attn} load failed ({type(exc).__name__}: {exc}); "
                  f"retrying with sdpa")
            traceback.print_exc()
            return loader("sdpa")
        raise


class ModelManager:
    """Thread-safe lazy loader with idle-based eviction."""

    def __init__(self, device: str = DEVICE, idle_seconds: int = IDLE_EVICT_SECONDS) -> None:
        self.device = device
        self.idle_seconds = idle_seconds
        self._models: Dict[str, LoadedModel] = {}
        self._registry_lock = threading.Lock()
        self._reaper_stop = threading.Event()
        self._reaper_thread: Optional[threading.Thread] = None

    # -- public API ---------------------------------------------------------

    def start_reaper(self) -> None:
        if self._reaper_thread is not None:
            return
        t = threading.Thread(target=self._reaper_loop, name="model-reaper", daemon=True)
        self._reaper_thread = t
        t.start()

    def shutdown(self) -> None:
        self._reaper_stop.set()
        if self._reaper_thread is not None:
            self._reaper_thread.join(timeout=2.0)
        with self._registry_lock:
            kinds = list(self._models.keys())
        for kind in kinds:
            self.evict(kind)

    def get_asr(self) -> LoadedModel:
        return self._get_or_load("asr", self._load_asr)

    def get_realtime(self) -> LoadedModel:
        return self._get_or_load("realtime", self._load_realtime)

    def get_tts(self) -> LoadedModel:
        return self._get_or_load("tts", self._load_tts)

    def list_loaded(self) -> Dict[str, float]:
        with self._registry_lock:
            return {k: m.last_used for k, m in self._models.items()}

    def evict(self, kind: str) -> bool:
        with self._registry_lock:
            loaded = self._models.pop(kind, None)
        if loaded is None:
            return False
        with loaded.lock:
            try:
                del loaded.model
                del loaded.processor
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[model_manager] Evicted {kind}")
        return True

    # -- internal -----------------------------------------------------------

    def _get_or_load(self, kind: str, loader: Callable[[], LoadedModel]) -> LoadedModel:
        with self._registry_lock:
            existing = self._models.get(kind)
            if existing is not None:
                existing.last_used = time.time()
                return existing
        # load outside registry lock to avoid blocking other kinds
        loaded = loader()
        with self._registry_lock:
            # Double-check after load (another thread may have loaded concurrently)
            existing = self._models.get(kind)
            if existing is not None:
                # Discard the one we just built
                try:
                    del loaded.model
                    del loaded.processor
                except Exception:
                    pass
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                existing.last_used = time.time()
                return existing
            self._models[kind] = loaded
            return loaded

    def _reaper_loop(self) -> None:
        while not self._reaper_stop.wait(30.0):
            now = time.time()
            to_evict = []
            with self._registry_lock:
                for kind, loaded in self._models.items():
                    if now - loaded.last_used > self.idle_seconds:
                        to_evict.append(kind)
            for kind in to_evict:
                self.evict(kind)

    # -- loaders ------------------------------------------------------------

    def _load_asr(self) -> LoadedModel:
        from vibevoice.modular.modeling_vibevoice_asr import (
            VibeVoiceASRForConditionalGeneration,
        )
        from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

        print(f"[model_manager] Loading ASR from {ASR_MODEL_PATH}")
        dtype, preferred_attn = _pick_dtype_attn(self.device)

        processor = VibeVoiceASRProcessor.from_pretrained(
            ASR_MODEL_PATH,
            language_model_pretrained_name=ASR_LANGUAGE_MODEL_NAME,
        )

        def _loader(attn: str):
            return VibeVoiceASRForConditionalGeneration.from_pretrained(
                ASR_MODEL_PATH,
                torch_dtype=dtype,
                attn_implementation=attn,
                trust_remote_code=True,
            )

        model = _load_with_fallback(_loader, preferred_attn)
        if self.device != "auto":
            model = model.to(self.device)
        model.eval()
        print(f"[model_manager] ASR loaded on {self.device}")
        return LoadedModel("asr", model, processor, time.time(), threading.Lock())

    def _load_realtime(self) -> LoadedModel:
        from vibevoice.modular.modeling_vibevoice_streaming_inference import (
            VibeVoiceStreamingForConditionalGenerationInference,
        )
        from vibevoice.processor.vibevoice_streaming_processor import (
            VibeVoiceStreamingProcessor,
        )

        print(f"[model_manager] Loading Realtime from {REALTIME_MODEL_PATH}")
        dtype, preferred_attn = _pick_dtype_attn(self.device)

        processor = VibeVoiceStreamingProcessor.from_pretrained(REALTIME_MODEL_PATH)

        def _loader(attn: str):
            return VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                REALTIME_MODEL_PATH,
                torch_dtype=dtype,
                device_map=self.device if self.device != "cpu" else "cpu",
                attn_implementation=attn,
            )

        model = _load_with_fallback(_loader, preferred_attn)
        model.eval()
        # Switch scheduler to SDE-DPM++ (matches official demo)
        model.model.noise_scheduler = model.model.noise_scheduler.from_config(
            model.model.noise_scheduler.config,
            algorithm_type="sde-dpmsolver++",
            beta_schedule="squaredcos_cap_v2",
        )
        model.set_ddpm_inference_steps(num_steps=5)
        print(f"[model_manager] Realtime loaded on {self.device}")
        return LoadedModel("realtime", model, processor, time.time(), threading.Lock())

    def _load_tts(self) -> LoadedModel:
        """Load VibeVoice-TTS-1.5B using the community-fork inference class.

        `modeling_vibevoice_inference.py` was vendored from
        https://github.com/vibevoice-community/VibeVoice because the official
        upstream removed TTS generation code. The weights are unchanged.
        """
        from vibevoice.modular.modeling_vibevoice_inference import (
            VibeVoiceForConditionalGenerationInference,
        )
        from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
        from .config import TTS_MODEL_PATH

        print(f"[model_manager] Loading TTS from {TTS_MODEL_PATH}")
        dtype, preferred_attn = _pick_dtype_attn(self.device)

        processor = VibeVoiceProcessor.from_pretrained(TTS_MODEL_PATH)

        def _loader(attn: str):
            return VibeVoiceForConditionalGenerationInference.from_pretrained(
                TTS_MODEL_PATH,
                torch_dtype=dtype,
                device_map=self.device if self.device != "cpu" else "cpu",
                attn_implementation=attn,
            )

        model = _load_with_fallback(_loader, preferred_attn)
        model.eval()
        model.set_ddpm_inference_steps(num_steps=10)
        print(f"[model_manager] TTS loaded on {self.device}")
        return LoadedModel("tts", model, processor, time.time(), threading.Lock())


_manager: Optional[ModelManager] = None
_manager_lock = threading.Lock()


def get_manager() -> ModelManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = ModelManager()
            _manager.start_reaper()
        return _manager
