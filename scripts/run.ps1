$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Run scripts\setup.ps1 first.'
}
# Refresh PATH so a just-installed FFmpeg is available without reopening the terminal.
$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' +
            [Environment]::GetEnvironmentVariable('Path', 'User') + ';' + $env:Path
Push-Location -LiteralPath $projectRoot
try {
    & $pythonPath -m audio_transcript @args
    $commandExit = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $commandExit
