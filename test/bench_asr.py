"""Benchmark /v1/asr/transcribe against every audio file under test/.

Reports wall time, audio duration, and real-time factor (RTF = wall/duration).
Run after starting the API server:

    python test/bench_asr.py [--max-new-tokens 2048] [--host http://localhost:8000]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import requests

AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".flac", ".ogg"}


def audio_duration_seconds(path: Path) -> float:
    out = subprocess.run(
        [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(out.stdout)["format"]["duration"])


def transcribe(host: str, path: Path, max_new_tokens: int, timeout: int) -> tuple[float, dict]:
    url = f"{host.rstrip('/')}/v1/asr/transcribe"
    with open(path, "rb") as f:
        files = {"audio": (path.name, f, "audio/mpeg")}
        data = {"max_new_tokens": str(max_new_tokens), "do_sample": "false"}
        t0 = time.time()
        r = requests.post(url, files=files, data=data, timeout=timeout)
        wall = time.time() - t0
    r.raise_for_status()
    return wall, r.json()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:8000")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--timeout", type=int, default=1800, help="Per-file HTTP timeout (sec)")
    ap.add_argument("--dir", default=str(Path(__file__).parent))
    ap.add_argument("--file", help="Only benchmark this single file (relative to --dir)")
    args = ap.parse_args()

    root = Path(args.dir)
    if args.file:
        files = [root / args.file]
    else:
        files = sorted(
            p for p in root.iterdir()
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS
        )
    if not files:
        print(f"No audio files found under {root}", file=sys.stderr)
        return 2

    # Warm up: make sure the model is already loaded so the first file's timing
    # isn't dominated by load time.
    try:
        h = requests.get(f"{args.host.rstrip('/')}/health", timeout=5).json()
        if "asr" not in h.get("loaded_models", []):
            print("[warmup] ASR not loaded yet; first file will include load time.")
    except Exception as exc:
        print(f"[warmup] /health unreachable: {exc}", file=sys.stderr)
        return 2

    rows: list[tuple[str, float, float, float, int]] = []
    for p in files:
        try:
            duration = audio_duration_seconds(p)
        except Exception as exc:
            print(f"[{p.name}] ffprobe failed: {exc}")
            continue

        size_mb = p.stat().st_size / 1024 / 1024
        print(f"\n=== {p.name} | {duration:.1f}s audio, {size_mb:.1f} MB ===")
        try:
            wall, resp = transcribe(args.host, p, args.max_new_tokens, args.timeout)
        except requests.Timeout:
            print(f"  TIMEOUT after {args.timeout}s")
            rows.append((p.name, duration, -1.0, -1.0, 0))
            continue
        except Exception as exc:
            print(f"  FAILED: {exc}")
            rows.append((p.name, duration, -1.0, -1.0, 0))
            continue

        gen_sec = resp.get("generation_time_sec", 0.0)
        rtf = wall / duration if duration else 0.0
        n_segs = len(resp.get("segments") or [])
        raw = (resp.get("raw_text") or "").strip().replace("\n", " ")
        preview = (raw[:160] + "…") if len(raw) > 160 else raw
        print(f"  wall={wall:.2f}s  generate={gen_sec:.2f}s  RTF={rtf:.3f}  segments={n_segs}")
        print(f"  preview: {preview}")
        rows.append((p.name, duration, wall, rtf, n_segs))

    print("\n" + "=" * 78)
    print(f"{'file':<48} {'audio_s':>8} {'wall_s':>8} {'RTF':>6} {'segs':>5}")
    print("-" * 78)
    for name, dur, wall, rtf, n in rows:
        wall_s = f"{wall:8.2f}" if wall >= 0 else f"{'TIMEOUT':>8}"
        rtf_s = f"{rtf:6.3f}" if rtf >= 0 else f"{'-':>6}"
        print(f"{name[:48]:<48} {dur:8.1f} {wall_s} {rtf_s} {n:5d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
