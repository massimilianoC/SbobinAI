@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run the README installation steps first.
    exit /b 1
)
".venv\Scripts\python.exe" -m audio_transcript.desktop %*
exit /b %ERRORLEVEL%
