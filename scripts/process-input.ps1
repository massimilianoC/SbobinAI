[CmdletBinding()]
param(
    [string]$InputFile,
    [string]$ConfigPath = 'config.local.toml',
    [switch]$Force
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Run scripts\setup.ps1 first.' }
$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' +
    [Environment]::GetEnvironmentVariable('Path', 'User') + ';' + $env:Path
Push-Location -LiteralPath $projectRoot
try {
    if (-not (Test-Path -LiteralPath $ConfigPath)) { throw 'Create config.local.toml from config.example.toml first.' }
    $runLogDirectory = Join-Path $projectRoot '.local\pipeline-runs'
    New-Item -ItemType Directory -Path $runLogDirectory -Force | Out-Null
    $runLog = Join-Path $runLogDirectory ((Get-Date -Format 'yyyyMMdd-HHmmss') + '.log')
    Start-Transcript -Path $runLog | Out-Null
    try {
        & $pythonPath -u -m audio_transcript doctor --config $ConfigPath
        if ($LASTEXITCODE -ne 0) { throw 'Pipeline dependency/backend validation failed.' }
        $pipelineArguments = @('-u', '-m', 'audio_transcript', 'run', '--config', $ConfigPath)
        if ($InputFile) { $pipelineArguments += @('--file', $InputFile) }
        if ($Force) { $pipelineArguments += '--force' }
        & $pythonPath @pipelineArguments
        $pipelineExit = $LASTEXITCODE
        Write-Output "Pipeline exit code: $pipelineExit. Inspect output/<job-id>/ and process/<job-id>/; execution log: $runLog"
    } finally { Stop-Transcript | Out-Null }
} finally { Pop-Location }
exit $pipelineExit
