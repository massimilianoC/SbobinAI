@echo off
rem Unattended transcription of every media file in the input folder (no questions asked).
rem Starts the llama.cpp server from the [server] table of config.local.toml when needed, then runs the pipeline.
rem Extra arguments are passed to scripts\process-input.ps1, for example:
rem   transcribe-batch.cmd -MaxDuration 600 -NoArchive -Label quick-check
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\process-input.ps1" %*
set "EXITCODE=%ERRORLEVEL%"
echo.
if "%EXITCODE%"=="0" (
    echo Done. Results: output\^<source^>\^<version^>\  -  overview: output\catalog.json
) else (
    echo Finished with exit code %EXITCODE%. See the log path printed above.
)
rem Keep the window open when started by double-click.
rem Delayed expansion: operators such as "&&" inside the original command line
rem must not be parsed again (an immediate %CMDCMDLINE% expansion re-ran this script).
setlocal EnableDelayedExpansion
set "LAUNCH_LINE=!CMDCMDLINE!"
if /i not "!LAUNCH_LINE:/c=!"=="!LAUNCH_LINE!" pause
exit /b %EXITCODE%
