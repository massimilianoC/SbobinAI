@echo off
rem Guided transcription: copy media into the input folder, then double-click this file.
rem A console asks the spoken language, optional context and scope, then shows progress and statistics.
rem Starts the llama.cpp server from the [server] table of config.local.toml when needed.
rem With arguments the unattended pipeline runs as before (same as transcribe-batch.cmd), for example:
rem   transcribe.cmd -MaxDuration 600 -NoArchive -Label quick-check
setlocal
cd /d "%~dp0"
if "%~1"=="" (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\process-input.ps1" -Interactive
) else (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\process-input.ps1" %*
)
set "EXITCODE=%ERRORLEVEL%"
echo.
if "%EXITCODE%"=="0" (
    echo Done. Results: output\^<source^>\^<version^>\  -  overview: output\catalog.json
) else (
    echo Finished with exit code %EXITCODE%. See the log path printed above.
)
rem Keep the window open when started by double-click.
echo %CMDCMDLINE% | find /i "/c" >nul && pause
exit /b %EXITCODE%
