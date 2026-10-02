[CmdletBinding()]
param(
    [string]$InputFile,
    [string]$ConfigPath = 'config.local.toml',
    # Use whatever server is already running; do not start or restart it.
    [switch]$NoServerManagement,
    [switch]$Force,
    # Bounded validation run: transcribe only the first N seconds (distinct job identity, input kept).
    [ValidateRange(1, 86400)]
    [double]$MaxDuration,
    # Keep the source in input even after a complete run (comparisons on the same recording).
    [switch]$NoArchive,
    # Short label added to the log file name, for example 'qwen3-asr-q8'.
    [ValidatePattern('^[A-Za-z0-9._-]{1,40}$')]
    [string]$Label,
    # Guided run: a console wizard asks language, context and scope, then shows progress and statistics.
    [switch]$Interactive,
    # With -Interactive: take the wizard answers from a JSON file instead of the console
    # (keys: language, context, max_minutes, confirm, files) for unattended guided runs.
    [string]$AnswersFile
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Run scripts\setup.ps1 first.' }
$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' +
    [Environment]::GetEnvironmentVariable('Path', 'User') + ';' + $env:Path
$env:PYTHONUNBUFFERED = '1'
$pipelineExit = 1

function Test-OwnedServerMatches($Server) {
    # Returns $true/$false for a running owned server, $null when none is recorded.
    $manifestPath = Join-Path $Server.state_dir 'server.json'
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) { return $null }
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    if ($null -eq (Get-Process -Id ([int]$manifest.processId) -ErrorAction SilentlyContinue)) { return $null }
    $full = { param($p) [System.IO.Path]::GetFullPath($p).TrimEnd('\') }
    return (
        [string]::Equals((& $full $manifest.modelPath), (& $full $Server.model_path), [StringComparison]::OrdinalIgnoreCase) -and
        [string]::Equals((& $full $manifest.projectorPath), (& $full $Server.projector_path), [StringComparison]::OrdinalIgnoreCase) -and
        [string]$manifest.alias -eq [string]$Server.alias -and
        [int]$manifest.port -eq [int]$Server.port -and
        [int]$manifest.configuration.contextSize -eq [int]$Server.context_size -and
        [int]$manifest.configuration.parallel -eq [int]$Server.parallel
    )
}

function Initialize-Server($Server) {
    # $Server is the JSON printed by `audio-transcript server-profile` (absolute paths).
    Write-Output ("Server profile from {0}: alias {1} | port {2} | context {3} | parallel slots {4}" -f
        $ConfigPath, $Server.alias, $Server.port, $Server.context_size, $Server.parallel)
    $serverState = Test-OwnedServerMatches $Server
    if ($serverState -eq $false) {
        Write-Output 'The owned llama.cpp server runs a different configuration; restarting it from the profile.'
        & (Join-Path $PSScriptRoot 'stop-llamacpp.ps1') -StateDirectory $Server.state_dir
        if ($LASTEXITCODE -ne 0) { throw 'Could not stop the previous llama.cpp server.' }
    }
    $startArguments = @{
        ModelPath = $Server.model_path
        ProjectorPath = $Server.projector_path
        RuntimeDirectory = $Server.runtime_dir
        Alias = $Server.alias
        StateDirectory = $Server.state_dir
        Port = [int]$Server.port
        ContextSize = [int]$Server.context_size
        Parallel = [int]$Server.parallel
    }
    if ($null -ne $Server.gpu_layers) { $startArguments.GpuLayers = [int]$Server.gpu_layers }
    & (Join-Path $PSScriptRoot 'start-llamacpp.ps1') @startArguments
    if ($LASTEXITCODE -ne 0) { throw 'llama.cpp server startup failed; see the messages above.' }
}

Push-Location -LiteralPath $projectRoot
try {
    if (-not (Test-Path -LiteralPath $ConfigPath)) { throw 'Create config.local.toml from config.example.toml first.' }
    $runLogDirectory = Join-Path $projectRoot '.local\pipeline-runs'
    New-Item -ItemType Directory -Path $runLogDirectory -Force | Out-Null
    $logName = Get-Date -Format 'yyyyMMdd-HHmmss'
    if ($Label) { $logName += "-$Label" }
    $runLog = Join-Path $runLogDirectory ($logName + '.log')
    Start-Transcript -Path $runLog | Out-Null
    try {
        if ($Interactive) {
            $pipelineArguments = @('-u', '-m', 'audio_transcript', 'wizard', '--config', $ConfigPath)
            $ignored = @()
            if ($InputFile) { $pipelineArguments += @('--file', $InputFile) }
            if ($AnswersFile) { $pipelineArguments += @('--answers', $AnswersFile) }
            if ($Force) { $ignored += '-Force' }
            if ($PSBoundParameters.ContainsKey('MaxDuration')) { $ignored += '-MaxDuration' }
            if ($NoArchive) { $ignored += '-NoArchive' }
            if ($ignored.Count -gt 0) {
                Write-Output "Note: interactive mode asks for its own parameters; ignoring $($ignored -join ', ')."
            }
        } else {
            $pipelineArguments = @('-u', '-m', 'audio_transcript', 'run', '--config', $ConfigPath)
            if ($InputFile) { $pipelineArguments += @('--file', $InputFile) }
            if ($Force) { $pipelineArguments += '--force' }
            if ($PSBoundParameters.ContainsKey('MaxDuration')) {
                $pipelineArguments += @('--max-duration', $MaxDuration.ToString([Globalization.CultureInfo]::InvariantCulture))
            }
            if ($NoArchive) { $pipelineArguments += '--no-archive-inputs' }
        }
        Write-Output "Run started (UTC): $([datetime]::UtcNow.ToString('o'))"
        Write-Output "Config: $ConfigPath; arguments: $($pipelineArguments[2..($pipelineArguments.Count - 1)] -join ' ')"
        if ($NoServerManagement) {
            Write-Output 'Server management disabled; using the running llama.cpp server as is.'
        } else {
            if (Test-Path -LiteralPath 'server.local.psd1' -PathType Leaf) {
                Write-Output 'Note: server.local.psd1 is no longer used; move its values into the [server] table of the config file (see config.example.toml) and delete it.'
            }
            $profileJson = (& $pythonPath -m audio_transcript server-profile --config $ConfigPath --optional) -join "`n"
            if ($LASTEXITCODE -ne 0) { throw 'The [server] configuration is invalid; see the message above (or use -NoServerManagement).' }
            if ($profileJson.Trim() -eq 'null') {
                Write-Output "No [server] table in $ConfigPath; using the running llama.cpp server as is."
            } else {
                Initialize-Server ($profileJson | ConvertFrom-Json)
            }
        }
        & $pythonPath -u -m audio_transcript doctor --config $ConfigPath
        if ($LASTEXITCODE -ne 0) { throw 'Pipeline dependency/backend validation failed.' }
        $clock = [Diagnostics.Stopwatch]::StartNew()
        & $pythonPath @pipelineArguments
        $pipelineExit = $LASTEXITCODE
        $clock.Stop()
        Write-Output ("Run finished (UTC): {0}; wall time {1:n1} s" -f [datetime]::UtcNow.ToString('o'), $clock.Elapsed.TotalSeconds)
        Write-Output "Pipeline exit code: $pipelineExit. Inspect output/<source>/<version>/ (see output/catalog.json) and process/<source>/<version>/; execution log: $runLog"
    } finally { Stop-Transcript | Out-Null }
} finally { Pop-Location }
exit $pipelineExit
