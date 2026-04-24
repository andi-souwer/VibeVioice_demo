@echo off
REM One-click launcher for the VibeVoice-1.5B TTS Gradio demo.
REM Model: VibeVoice-1.5B (multi-speaker long-form TTS), served at http://127.0.0.1:7860
REM Conda env: pytorchst1. No network - HF_HUB_OFFLINE blocks all HF fetches.
REM
REM To STOP: either close this window, press Ctrl+C, or run stop_demos.bat.
REM VRAM is auto-released when the Python process exits (OS reclaims the CUDA
REM context). The tagged window title below lets stop_demos.bat find us.

setlocal
title VibeVoice-TTS-Gradio

set CONDA_ENV=pytorchst1
set HOST=127.0.0.1
set PORT=7860

REM Anchor to this script's directory so relative paths resolve.
cd /d "%~dp0"

REM --- Override points (env vars take precedence if already set) -------------
if "%VIBEVOICE_MODEL_ROOT%"=="" set VIBEVOICE_MODEL_ROOT=E:\lenv\llmmode\localdown
if "%HF_HOME%"==""             set HF_HOME=E:\lenv\huggingface
if "%TTS_INFERENCE_STEPS%"=="" set TTS_INFERENCE_STEPS=10

REM --- Offline / Windows-safe knobs ------------------------------------------
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set TRANSFORMERS_VERBOSITY=error
REM USE_TF=0 stops transformers from auto-loading the TF backend on AutoModel
REM import; TF otherwise grabs ~5-6 GB VRAM greedily and makes coexisting models
REM spill to host RAM via sysmem fallback.
set USE_TF=0
set TF_CPP_MIN_LOG_LEVEL=3
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
REM vibevoice_tts_gradio_demo.py already inserts VibeVoice/ into sys.path, so
REM PYTHONPATH is not strictly required; we still scrub HF_ENDPOINT to avoid
REM any DNS retries pointed at a stale mirror.
set HF_ENDPOINT=

echo.
echo [TTS gradio] conda env         : %CONDA_ENV%
echo [TTS gradio] model path        : %VIBEVOICE_MODEL_ROOT%\VibeVoice-1.5B
echo [TTS gradio] HF cache (HOME)   : %HF_HOME%
echo [TTS gradio] inference steps   : %TTS_INFERENCE_STEPS%
echo [TTS gradio] listening on      : http://%HOST%:%PORT%
echo.

REM `call` is required so control returns to this .bat after conda activate.
call conda activate %CONDA_ENV%
if errorlevel 1 (
    echo [ERROR] conda activate %CONDA_ENV% failed. Run `conda init cmd.exe` once, then retry.
    exit /b 1
)

python VibeVoice\demo\vibevoice_tts_gradio_demo.py ^
    --model_path "%VIBEVOICE_MODEL_ROOT%\VibeVoice-1.5B" ^
    --inference_steps %TTS_INFERENCE_STEPS% ^
    --host %HOST% ^
    --port %PORT%

REM Belt-and-suspenders cleanup: python already exited, but if gradio spawned
REM any straggler Python children in this console they get killed here too.
taskkill /FI "WINDOWTITLE eq VibeVoice-TTS-Gradio" /T /F >nul 2>&1
echo.
echo [TTS gradio] stopped. VRAM released.
endlocal
