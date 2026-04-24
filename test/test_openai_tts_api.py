"""End-to-end test matrix for the standalone VibeVoice TTS API.

Exercises every advertised VibeVoice capability via HTTP and prints a
pass/fail summary. Run against a live server:

    start_openai_tts_api.bat
    python test/test_openai_tts_api.py --host http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import hashlib
import io
import sys
import time
import wave
from pathlib import Path
from typing import Optional

import requests

TEST_DIR = Path(__file__).resolve().parent
SAMPLE_VOICE_WAV = (
    TEST_DIR.parent / "VibeVoice" / "demo" / "voices" / "tts_model" / "en-Alice_woman.wav"
)

_PASS = "✓"   # ✓
_FAIL = "✗"   # ✗
_SKIP = "-"

results: list[tuple[str, str, str]] = []  # (name, status, detail)


def record(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, _PASS if ok else _FAIL, detail))
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}  {detail}", flush=True)


def section(label: str) -> None:
    print(f"\n== {label} ==", flush=True)


# -- Helpers ------------------------------------------------------------------
def speech(host: str, body: dict, expect_status: int = 200, headers: Optional[dict] = None):
    r = requests.post(
        f"{host}/v1/audio/speech",
        json=body,
        headers=headers or {},
        timeout=180,
    )
    if r.status_code != expect_status:
        raise AssertionError(
            f"expected {expect_status}, got {r.status_code}: {r.text[:200]}"
        )
    return r


def wav_duration_sec(data: bytes) -> float:
    with wave.open(io.BytesIO(data), "rb") as w:
        return w.getnframes() / w.getframerate()


# -- Tests --------------------------------------------------------------------
def test_health(host: str) -> None:
    section("Service health")
    r = requests.get(f"{host}/health", timeout=5)
    j = r.json()
    record("GET /health 200", r.status_code == 200)
    record("service == vibevoice-tts", j.get("service") == "vibevoice-tts",
           f"got={j.get('service')!r}")
    record("model path configured", bool(j.get("model_path")),
           f"path={j.get('model_path')}")
    record("cuda available", j.get("cuda_available", False))


def test_voice_catalog(host: str) -> None:
    section("Voice catalog")
    r = requests.get(f"{host}/v1/audio/voices", timeout=5)
    j = r.json()
    record("GET /v1/audio/voices 200", r.status_code == 200)
    record("has 11 OpenAI aliases", len(j["aliases"]) == 11, f"got={len(j['aliases'])}")
    record("has builtin presets", len(j["builtin"]) >= 9, f"got={len(j['builtin'])}")
    # spot-check specific entries
    record("'alloy' in aliases", "alloy" in j["aliases"])
    record("'en-Carter_man' in builtin", "en-Carter_man" in j["builtin"])
    record("'zh-Bowen_man' in builtin", "zh-Bowen_man" in j["builtin"])


def test_formats(host: str) -> None:
    section("All response formats")
    fmt_to_magic = {
        "wav":  (b"RIFF",          "audio/wav"),
        "mp3":  (b"ID3",           "audio/mpeg"),
        "flac": (b"fLaC",          "audio/flac"),
        "opus": (b"OggS",          "audio/opus"),
        "aac":  (None,             "audio/aac"),      # ADTS header varies
        "pcm":  (None,             "application/octet-stream"),
    }
    body_base = {"model": "vibevoice-1.5b", "voice": "alloy",
                 "input": "Format coverage check.", "vv_seed": 7}
    for fmt, (magic, ct) in fmt_to_magic.items():
        r = speech(host, {**body_base, "response_format": fmt})
        ok_ct = r.headers.get("content-type", "").startswith(ct.split(";")[0])
        ok_magic = True if magic is None else r.content[:len(magic)] == magic
        ok_size = len(r.content) > 1000
        detail = (
            f"ct={r.headers.get('content-type')} size={len(r.content)}"
            f" magic={r.content[:4].hex() if r.content else 'empty'}"
        )
        record(f"format={fmt}", ok_ct and ok_magic and ok_size, detail)


def test_headers(host: str) -> None:
    section("Response headers")
    r = speech(host, {"voice": "alloy", "input": "Headers test.",
                      "response_format": "wav", "vv_seed": 11})
    for h in ("X-Audio-Duration-Sec", "X-Generation-Time-Sec",
              "X-Sample-Rate", "X-Speakers", "Content-Disposition"):
        record(f"header {h} present", h in r.headers, f"{h}={r.headers.get(h)}")
    record("Sample-Rate == 24000", r.headers.get("X-Sample-Rate") == "24000")
    record("Speakers echoed", r.headers.get("X-Speakers") == "alloy")


def test_aliases_and_presets(host: str) -> None:
    section("Alias + preset voices")
    for name in ["alloy", "nova", "shimmer", "ash"]:
        r = speech(host, {"voice": name, "input": f"Alias {name}.",
                          "response_format": "wav", "vv_seed": 1})
        dur = wav_duration_sec(r.content)
        record(f"alias '{name}'", dur > 0.2, f"duration={dur:.2f}s")
    for name in ["en-Carter_man", "zh-Bowen_man"]:
        txt = "Testing preset." if "en-" in name else "测试原生 preset。"
        r = speech(host, {"voice": name, "input": txt,
                          "response_format": "wav", "vv_seed": 1})
        record(f"preset '{name}'", wav_duration_sec(r.content) > 0.2,
               f"size={len(r.content)}")


def test_multi_speaker(host: str) -> None:
    section("Multi-speaker via vv_speakers + 'Speaker N:' in input")
    r = speech(host, {
        "voice": "alloy",
        "input": "Speaker 1: Hey.\nSpeaker 2: Hi.\nSpeaker 1: How are you?",
        "vv_speakers": ["alloy", "nova"],
        "response_format": "wav", "vv_seed": 3,
    })
    record("2-speaker synthesis", wav_duration_sec(r.content) > 1.0,
           f"dur={wav_duration_sec(r.content):.2f}s speakers={r.headers.get('X-Speakers')}")


def test_speed(host: str) -> None:
    section("speed parameter scales duration")
    durations = {}
    for sp in (0.5, 1.0, 2.0):
        r = speech(host, {"voice": "alloy", "response_format": "wav",
                          "input": "Speed coverage test one two three.",
                          "speed": sp, "vv_seed": 5})
        durations[sp] = wav_duration_sec(r.content)
    record("speed=0.5 longer than 1.0", durations[0.5] > durations[1.0],
           f"0.5={durations[0.5]:.2f}s vs 1.0={durations[1.0]:.2f}s")
    record("speed=2.0 shorter than 1.0", durations[2.0] < durations[1.0],
           f"2.0={durations[2.0]:.2f}s vs 1.0={durations[1.0]:.2f}s")
    # Duration ratio should be roughly 1/speed (librosa time_stretch)
    ratio = durations[1.0] / durations[2.0]
    record("speed=2.0 ~= 2x faster", 1.6 <= ratio <= 2.3,
           f"ratio={ratio:.2f}")


def test_seed_reproducibility(host: str) -> None:
    section("Seed reproducibility")
    body = {"voice": "alloy", "input": "Reproducible output please.",
            "response_format": "wav", "vv_seed": 999}
    h1 = hashlib.sha256(speech(host, body).content).hexdigest()
    h2 = hashlib.sha256(speech(host, body).content).hexdigest()
    record("same seed -> same bytes", h1 == h2, f"sha256={h1[:12]}")
    body_diff = {**body, "vv_seed": 1000}
    h3 = hashlib.sha256(speech(host, body_diff).content).hexdigest()
    record("different seed -> different bytes", h1 != h3,
           f"alt sha256={h3[:12]}")


def test_sampling_path(host: str) -> None:
    section("Sampling params exercised (do_sample=True)")
    r = speech(host, {"voice": "alloy", "input": "Sampling path smoke.",
                      "response_format": "wav",
                      "vv_do_sample": True, "vv_temperature": 0.9, "vv_top_p": 0.92,
                      "vv_seed": 7})
    record("do_sample=True works", wav_duration_sec(r.content) > 0.2,
           f"dur={wav_duration_sec(r.content):.2f}s")


def test_cfg_and_steps(host: str) -> None:
    section("CFG scale + inference steps")
    for cfg in (1.0, 1.3, 2.0):
        r = speech(host, {"voice": "alloy", "input": f"CFG {cfg}.",
                          "response_format": "wav",
                          "vv_cfg_scale": cfg, "vv_seed": 2})
        record(f"cfg_scale={cfg}", r.status_code == 200,
               f"size={len(r.content)}")
    for steps in (5, 10, 20):
        r = speech(host, {"voice": "alloy", "input": "Step count test.",
                          "response_format": "wav",
                          "vv_inference_steps": steps, "vv_seed": 2})
        record(f"inference_steps={steps}", r.status_code == 200,
               f"size={len(r.content)}")


def test_custom_voice(host: str) -> None:
    section("Custom voice upload + synthesize + delete")
    if not SAMPLE_VOICE_WAV.exists():
        record("upload", False, f"no sample at {SAMPLE_VOICE_WAV}")
        return
    with open(SAMPLE_VOICE_WAV, "rb") as f:
        r = requests.post(
            f"{host}/v1/audio/voices",
            files={"file": ("sample.wav", f, "audio/wav")},
            data={"name": "api-test-voice"},
            timeout=30,
        )
    record("POST /v1/audio/voices 200", r.status_code == 200, f"body={r.json()}")
    vid = r.json()["id"]
    record("id has vv- prefix", vid.startswith("vv-"), f"id={vid}")

    # Synthesize using the custom voice
    r = speech(host, {"voice": vid, "input": "Custom voice is cloned.",
                      "response_format": "wav", "vv_seed": 1})
    record("synthesize with custom voice", wav_duration_sec(r.content) > 0.2,
           f"dur={wav_duration_sec(r.content):.2f}s speakers={r.headers.get('X-Speakers')}")

    # List must include it
    r = requests.get(f"{host}/v1/audio/voices", timeout=5)
    custom_ids = {v["id"] for v in r.json().get("custom", [])}
    record("listed in /voices custom", vid in custom_ids, f"custom={sorted(custom_ids)}")

    # Delete
    r = requests.delete(f"{host}/v1/audio/voices/{vid}", timeout=10)
    record("DELETE /v1/audio/voices/{id} 200", r.status_code == 200,
           f"body={r.json()}")

    # Post-delete ref should 400 with unknown_voice code
    r = requests.post(
        f"{host}/v1/audio/speech", timeout=30,
        json={"voice": vid, "input": "should fail", "response_format": "wav"},
    )
    ok = (
        r.status_code == 400
        and r.json().get("error", {}).get("code") == "unknown_voice"
    )
    record("deleted voice -> 400 unknown_voice", ok, f"body={r.text[:120]}")


def test_error_shapes(host: str) -> None:
    section("Error handling (OpenAI-shape)")
    def _assert_openai_error(r, expected_code: Optional[str] = None,
                              expected_param: Optional[str] = None):
        if r.status_code < 400:
            return False, f"status={r.status_code}"
        try:
            err = r.json().get("error", {})
        except Exception:
            return False, "non-json body"
        ok_shape = all(k in err for k in ("message", "type", "param", "code"))
        ok_code = expected_code is None or err.get("code") == expected_code
        ok_param = expected_param is None or err.get("param") == expected_param
        return ok_shape and ok_code and ok_param, (
            f"status={r.status_code} err={err}"
        )

    r = requests.post(f"{host}/v1/audio/speech", timeout=15,
                      json={"voice": "alloy", "input": "   "})
    ok, detail = _assert_openai_error(r, expected_param="input")
    record("empty input -> 400 param=input", ok, detail)

    r = requests.post(f"{host}/v1/audio/speech", timeout=15,
                      json={"voice": "no-such-voice-xyz", "input": "hi"})
    ok, detail = _assert_openai_error(r, expected_code="unknown_voice",
                                       expected_param="voice")
    record("unknown voice -> 400 unknown_voice", ok, detail)

    r = requests.post(f"{host}/v1/audio/speech", timeout=15,
                      json={"voice": "alloy", "input": "hi",
                            "response_format": "wma"})
    ok, detail = _assert_openai_error(r, expected_param="response_format",
                                       expected_code="invalid_value")
    record("bad format -> 400 invalid_value", ok, detail)

    r = requests.post(f"{host}/v1/audio/speech", timeout=15,
                      json={"voice": "alloy", "input": "hi",
                            "stream_format": "sse"})
    ok, detail = _assert_openai_error(r, expected_param="stream_format",
                                       expected_code="not_implemented")
    record("stream_format=sse -> 400 not_implemented", ok, detail)

    # Speaker count mismatch
    r = requests.post(f"{host}/v1/audio/speech", timeout=15,
                      json={"voice": "alloy",
                            "input": "Speaker 1: hi.\nSpeaker 2: yo.\nSpeaker 3: sup?",
                            "vv_speakers": ["alloy", "nova"]})
    ok, detail = _assert_openai_error(r, expected_param="vv_speakers")
    record("speaker count mismatch -> 400", ok, detail)


def test_openai_sdk(host: str) -> None:
    section("OpenAI Python SDK compatibility")
    try:
        from openai import OpenAI
    except ImportError:
        record("openai SDK installed", False, "`pip install openai` not available")
        return
    c = OpenAI(base_url=f"{host}/v1", api_key="sk-local")
    r = c.audio.speech.create(model="vibevoice-1.5b", voice="nova",
                               input="SDK compat check.",
                               response_format="wav",
                               extra_body={"vv_seed": 13})
    record("SDK audio.speech.create -> RIFF", r.content[:4] == b"RIFF",
           f"size={len(r.content)} header={r.content[:4]!r}")


# -- Runner -------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://127.0.0.1:8000",
                    help="Base URL of the standalone TTS API")
    args = ap.parse_args()
    host = args.host.rstrip("/")

    print(f"Target: {host}", flush=True)
    t0 = time.time()

    test_health(host)
    test_voice_catalog(host)
    test_formats(host)
    test_headers(host)
    test_aliases_and_presets(host)
    test_multi_speaker(host)
    test_speed(host)
    test_seed_reproducibility(host)
    test_sampling_path(host)
    test_cfg_and_steps(host)
    test_custom_voice(host)
    test_error_shapes(host)
    test_openai_sdk(host)

    wall = time.time() - t0
    passes = sum(1 for _, s, _ in results if s == _PASS)
    fails = sum(1 for _, s, _ in results if s == _FAIL)

    print("\n" + "=" * 74)
    print(f"{'passed':>10}: {passes}")
    print(f"{'failed':>10}: {fails}")
    print(f"{'wall':>10}: {wall:.1f}s")
    print("=" * 74)
    if fails:
        print("\nFAILURES:")
        for n, s, d in results:
            if s == _FAIL:
                print(f"  {n}  {d}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
