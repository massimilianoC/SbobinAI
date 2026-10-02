param([switch]$Dev)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
        & python -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python virtual environment.' }
    }
    $package = if ($Dev) { '.[dev]' } else { '.' }
    & '.\.venv\Scripts\python.exe' -m pip install -e $package
    if ($LASTEXITCODE -ne 0) { throw 'Could not install the local package.' }
    Write-Host 'Setup complete. Run scripts\run.ps1 doctor --backend mock to check media tools.'
} finally {
    Pop-Location
}
