@echo off
REM Launch the VibeVoice unified API server inside conda env pytorchst1.

setlocal

set CONDA_ENV=pytorchst1
set HOST=0.0.0.0
set PORT=8000

REM Anchor to the directory containing this script so relative paths work.
cd /d "%~dp0"

REM Default model root. Override with: set VIBEVOICE_MODEL_ROOT=...
if "%VIBEVOICE_MODEL_ROOT%"=="" set VIBEVOICE_MODEL_ROOT=E:\lenv\llmmode\localdown
if "%VIBEVOICE_DEVICE%"=="" set VIBEVOICE_DEVICE=cuda
if "%HF_ENDPOINT%"=="" set HF_ENDPOINT=https://hf-mirror.com

echo.
echo [VibeVoice API] conda env       : %CONDA_ENV%
echo [VibeVoice API] model root      : %VIBEVOICE_MODEL_ROOT%
echo [VibeVoice API] device          : %VIBEVOICE_DEVICE%
echo [VibeVoice API] listening on    : http://%HOST%:%PORT%
echo.

REM `call` is required for conda activate inside a .bat.
call conda activate %CONDA_ENV%
if errorlevel 1 (
    echo [ERROR] Failed to activate conda env %CONDA_ENV%. Run `conda init cmd.exe` once, then retry.
    exit /b 1
)

python -m uvicorn api.server:app --host %HOST% --port %PORT%

endlocal
