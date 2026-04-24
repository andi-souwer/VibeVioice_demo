@echo off
REM One-click launcher for the VibeVoice-ASR Gradio demo.
REM Model: VibeVoice-ASR (8.67B), served at http://127.0.0.1:7861
REM Conda env: pytorchst1. No network - HF_HUB_OFFLINE blocks all HF fetches.
REM
REM To STOP: either close this window, press Ctrl+C, or run stop_demos.bat.
REM VRAM is auto-released when the Python process exits (OS reclaims the CUDA
REM context). The tagged window title below lets stop_demos.bat find us.

setlocal
title VibeVoice-ASR-Gradio

set CONDA_ENV=pytorchst1
set HOST=127.0.0.1
set PORT=7861

REM Anchor to this script's directory so relative paths resolve.
cd /d "%~dp0"

REM --- Override points (env vars take precedence if already set) -------------
if "%VIBEVOICE_MODEL_ROOT%"=="" set VIBEVOICE_MODEL_ROOT=E:\lenv\llmmode\localdown
if "%HF_HOME%"==""             set HF_HOME=E:\lenv\huggingface
if "%ASR_ATTN_IMPL%"==""       set ASR_ATTN_IMPL=sdpa

REM --- Offline / Windows-safe knobs ------------------------------------------
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set TRANSFORMERS_VERBOSITY=error
REM USE_TF=0 stops transformers from auto-loading the TensorFlow backend on
REM AutoModel import. Without this, TF is pulled in transitively and greedily
REM reserves ~5-6 GB of VRAM at startup, which pushes the 16 GB ASR model past
REM the 24 GB cap on a 4090 and triggers Windows CUDA sysmem fallback. That is
REM what made transcription take hours instead of ~90s.
set USE_TF=0
set TF_CPP_MIN_LOG_LEVEL=3
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set PYTHONPATH=%~dp0VibeVoice
REM HF_ENDPOINT intentionally unset — offline mode makes it irrelevant and any
REM stale mirror value can stall on DNS retries.
set HF_ENDPOINT=

echo.
echo [ASR gradio] conda env       : %CONDA_ENV%
echo [ASR gradio] model path      : %VIBEVOICE_MODEL_ROOT%\VibeVoice-ASR
echo [ASR gradio] HF cache (HOME) : %HF_HOME%
echo [ASR gradio] attn impl       : %ASR_ATTN_IMPL%
echo [ASR gradio] listening on    : http://%HOST%:%PORT%
echo.

REM `call` is required so control returns to this .bat after conda activate.
call conda activate %CONDA_ENV%
if errorlevel 1 (
    echo [ERROR] conda activate %CONDA_ENV% failed. Run `conda init cmd.exe` once, then retry.
    exit /b 1
)

python third_party\VibeVoiceCommunity\demo\vibevoice_asr_gradio_demo.py ^
    --model_path "%VIBEVOICE_MODEL_ROOT%\VibeVoice-ASR" ^
    --attn_implementation %ASR_ATTN_IMPL% ^
    --host %HOST% ^
    --port %PORT%

REM Belt-and-suspenders cleanup: python already exited, but if gradio spawned
REM any straggler Python children in this console they get killed here too.
taskkill /FI "WINDOWTITLE eq VibeVoice-ASR-Gradio" /T /F >nul 2>&1
echo.
echo [ASR gradio] stopped. VRAM released.
endlocal
