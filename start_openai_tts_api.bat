@echo off
REM One-click launcher for the STANDALONE VibeVoice-1.5B TTS API.
REM OpenAI-compatible endpoints, TTS model ONLY (no ASR, no Realtime).
REM
REM Served at:
REM   POST http://<host>:<port>/v1/audio/speech            OpenAI-shape TTS
REM   GET  http://<host>:<port>/v1/audio/voices            list voices
REM   POST http://<host>:<port>/v1/audio/voices            upload custom voice
REM   GET  http://<host>:<port>/health                     liveness + VRAM stats
REM
REM Stop via Ctrl+C, window close, or stop_demos.bat.

setlocal
title VibeVoice-TTS-API

set CONDA_ENV=pytorchst1
set HOST=0.0.0.0
set PORT=8000

cd /d "%~dp0"

REM --- Override points (env takes precedence if already set) ----------------
if "%VIBEVOICE_MODEL_ROOT%"=="" set VIBEVOICE_MODEL_ROOT=E:\lenv\llmmode\localdown
if "%HF_HOME%"==""             set HF_HOME=E:\lenv\huggingface
if "%VIBEVOICE_DEVICE%"==""    set VIBEVOICE_DEVICE=cuda
REM VIBEVOICE_API_KEY left unset by default -> no auth. Set it before running
REM to require `Authorization: Bearer <value>` on every /v1/audio/* call.
REM VIBEVOICE_EVICT_AFTER_REQUEST=1 releases VRAM after every request (tight
REM GPUs); default 0 keeps the model warm for 60s (VIBEVOICE_IDLE_EVICT_SECONDS).

REM --- Offline / Windows-safe knobs -----------------------------------------
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set TRANSFORMERS_VERBOSITY=error
set USE_TF=0
set TF_CPP_MIN_LOG_LEVEL=3
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set HF_ENDPOINT=

echo.
echo [TTS API] conda env        : %CONDA_ENV%
echo [TTS API] model path       : %VIBEVOICE_MODEL_ROOT%\VibeVoice-1.5B
echo [TTS API] HF cache (HOME)  : %HF_HOME%
echo [TTS API] device           : %VIBEVOICE_DEVICE%
if defined VIBEVOICE_API_KEY (
    echo [TTS API] bearer auth      : ENABLED
) else (
    echo [TTS API] bearer auth      : DISABLED ^(set VIBEVOICE_API_KEY to enable^)
)
echo [TTS API] listening on     : http://%HOST%:%PORT%
echo [TTS API] ASR / Realtime   : NOT loaded, NOT served
echo.

call conda activate %CONDA_ENV%
if errorlevel 1 (
    echo [ERROR] conda activate %CONDA_ENV% failed. Run `conda init cmd.exe` once, then retry.
    exit /b 1
)

python -m uvicorn api.openai_tts_server:app --host %HOST% --port %PORT%

REM Belt-and-suspenders cleanup in case python exited leaving children.
taskkill /FI "WINDOWTITLE eq VibeVoice-TTS-API" /T /F >nul 2>&1
echo.
echo [TTS API] stopped. VRAM released.
endlocal
