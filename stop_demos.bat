@echo off
REM One-click stop: kill every VibeVoice demo / API process started by this
REM repo, then confirm the GPU is free. Safe to run even if nothing is up.
REM
REM Strategy:
REM   1. Kill cmd windows tagged with a VibeVoice-* title (and their child
REM      python), i.e. anything launched via start_*.bat in this repo.
REM   2. As a second pass, kill any python.exe whose command line points at
REM      our demo scripts or api.server - catches processes started outside
REM      the tagged .bat wrappers (e.g. `python -m uvicorn api.server:app`).
REM   3. Print nvidia-smi memory.used so you can see VRAM come back to idle.

setlocal ENABLEDELAYEDEXPANSION

echo [stop] killing tagged gradio / api windows ...
taskkill /FI "WINDOWTITLE eq VibeVoice-ASR-Gradio" /T /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq VibeVoice-TTS-Gradio" /T /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq VibeVoice-TTS-API"    /T /F >nul 2>&1

echo [stop] killing any stray python.exe running our demos / api ...
REM PowerShell finds python.exe processes by command line pattern and kills them.
REM We match: vibevoice_*_gradio_demo, VibeVoiceCommunity\demo\gradio_demo,
REM api.server (unified uvicorn), api.openai_tts_server (standalone TTS API).
powershell -NoProfile -Command ^
  "$pat = 'vibevoice_asr_gradio_demo|vibevoice_tts_gradio_demo|VibeVoiceCommunity.demo.gradio_demo|api\.server|api\.openai_tts_server';" ^
  "Get-CimInstance Win32_Process -Filter \"name = 'python.exe'\" |" ^
  "  Where-Object { $_.CommandLine -match $pat } |" ^
  "  ForEach-Object {" ^
  "    Write-Host ('[stop] killing PID ' + $_.ProcessId);" ^
  "    try { Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop } catch {}" ^
  "  }"

echo.
echo [stop] GPU memory after cleanup:
where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo     ^(nvidia-smi not on PATH; open Task Manager to verify^)
) else (
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
)

endlocal
