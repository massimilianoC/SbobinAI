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
    # (keys: language, context, max_minutes, after_success, confirm, files) for unattended guided runs.
    [string]$AnswersFile,
    # With -Interactive: language of the wizard's questions (default: the system language).
    [ValidateSet('auto', 'en', 'it', 'es', 'fr', 'de', 'pt')]
    [string]$UiLanguage,
    # After a complete, full-length transcription: keep-all (move the original to processed),
    # keep-audio (keep a FLAC of its audio, delete the original) or delete-all. Default: the
    # after_success setting of the config file (keep-all). With -Interactive it is the default answer.
    [ValidateSet('keep-all', 'keep-audio', 'delete-all')]
    [string]$AfterSuccess,
    # Console output of the pipeline: live progress display, plain status lines, a JSONL event
    # stream, or auto (live only on an interactive terminal; the default).
    [ValidateSet('auto', 'live', 'plain', 'jsonl')]
    [string]$Ui,
    # Disable the CPU/RAM/GPU resource sampling of the run.
    [switch]$NoMonitor
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Run scripts\setup.ps1 first.' }
$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' +
    [Environment]::GetEnvironmentVariable('Path', 'User') + ';' + $env:Path
$env:PYTHONUNBUFFERED = '1'
$pipelineExit = 1
$scriptStartUtc = [datetime]::UtcNow

function Test-OwnedServerMatches($Server) {
    # Returns $true/$false for a running owned server, $null when none is recorded.
    $manifestPath = Join-Path $Server.state_dir 'server.json'
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) { return $null }
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    if ($null -eq (Get-Process -Id ([int]$manifest.processId) -ErrorAction SilentlyContinue)) { return $null }
    $full = { param($p) [System.IO.Path]::GetFullPath($p).TrimEnd('\') }
    # Manifests written before backend selection have no backend/device/threads: cuda, CUDA0, 0.
    $recorded = $manifest.configuration
    $recordedBackend = if ($null -ne $recorded.PSObject.Properties['backend']) { [string]$recorded.backend } else { 'cuda' }
    $recordedDevice = if ($null -ne $recorded.PSObject.Properties['device']) { [string]$recorded.device } else { 'CUDA0' }
    $recordedThreads = if ($null -ne $recorded.PSObject.Properties['threads'] -and $null -ne $recorded.threads) { [int]$recorded.threads } else { 0 }
    $wantedThreads = if ($Server.backend -eq 'cpu' -and $null -ne $Server.threads) { [int]$Server.threads } else { 0 }
    return (
        $recordedBackend -eq [string]$Server.backend -and
        $recordedDevice -eq [string]$Server.device -and
        $recordedThreads -eq $wantedThreads -and
        [string]::Equals((& $full $manifest.modelPath), (& $full $Server.model_path), [StringComparison]::OrdinalIgnoreCase) -and
        [string]::Equals((& $full $manifest.projectorPath), (& $full $Server.projector_path), [StringComparison]::OrdinalIgnoreCase) -and
        [string]$manifest.alias -eq [string]$Server.alias -and
        [int]$manifest.port -eq [int]$Server.port -and
        [int]$manifest.configuration.contextSize -eq [int]$Server.context_size -and
        [int]$manifest.configuration.parallel -eq [int]$Server.parallel
    )
}

function Write-BackendFallbackWarning($Server) {
    # Prominent, multi-line notice when the automatic fallback chose a slower backend.
    $selection = $Server.selection
    if ($null -eq $selection -or -not [bool]$selection.fallback_used) { return }
    $bar = '=' * 78
    Write-Host $bar -ForegroundColor Yellow
    Write-Host 'WARNING: GPU backend fallback in use' -ForegroundColor Yellow
    $chosen = [string]$selection.backend
    if ($selection.backend -eq 'cpu') { $chosen += " ($($selection.threads) threads)" }
    else { $chosen += " ($($selection.device), $($selection.device_name))" }
    Write-Host "  Backend chosen : $chosen" -ForegroundColor Yellow
    foreach ($attempt in @($selection.tried)) {
        if (-not [bool]$attempt.ok) {
            Write-Host ("  Skipped {0,-7}: {1}" -f $attempt.backend, $attempt.reason) -ForegroundColor Yellow
        }
    }
    $slowdown = switch ([string]$selection.backend) {
        'vulkan' { 'about 2x slower than CUDA' }
        'cpu' { 'about 9x slower than CUDA' }
        default { $null }
    }
    if ($slowdown) {
        Write-Host "  Expected speed : $slowdown (measured on the reference machine; other hardware differs)" -ForegroundColor Yellow
    }
    Write-Host '  The transcript is not affected: identical output was measured on CUDA, Vulkan and CPU.' -ForegroundColor Yellow
    Write-Host '  To restore GPU speed install/repair the skipped backend (docs\runtime-setup.md), or set [server].backend = "cuda" to fail instead of falling back.' -ForegroundColor Yellow
    Write-Host $bar -ForegroundColor Yellow
}

function Initialize-Server($Server) {
    # $Server is the JSON printed by `audio-transcript server-profile` (absolute paths).
    Write-Output ("Server profile from {0}: alias {1} | port {2} | context {3} | parallel slots {4}" -f
        $ConfigPath, $Server.alias, $Server.port, $Server.context_size, $Server.parallel)
    $backendText = "{0} ({1})" -f $Server.backend, $Server.device
    if ($Server.backend -eq 'cpu') { $backendText = "cpu ({0} threads)" -f $Server.threads }
    elseif ($Server.selection.device_name) { $backendText += ' ' + $Server.selection.device_name }
    Write-Output ("Inference backend: {0}; requested {1}" -f $backendText, $Server.selection.requested)
    Write-BackendFallbackWarning $Server
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
        Backend = [string]$Server.backend
        Device = [string]$Server.device
    }
    if ($Server.backend -eq 'cpu' -and $null -ne $Server.threads) { $startArguments.Threads = [int]$Server.threads }
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
            if ($UiLanguage) { $pipelineArguments += @('--ui-language', $UiLanguage) }
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
            if ($UiLanguage) { Write-Output 'Note: -UiLanguage applies only to the guided wizard (-Interactive); ignored.' }
        }
        if ($AfterSuccess) { $pipelineArguments += @('--after-success', $AfterSuccess) }
        if ($Ui) { $pipelineArguments += @('--ui', $Ui) }
        if ($NoMonitor) { $pipelineArguments += '--no-monitor' }
        Write-Output "Run started (UTC): $([datetime]::UtcNow.ToString('o'))"
        Write-Output "Config: $ConfigPath; arguments: $($pipelineArguments[2..($pipelineArguments.Count - 1)] -join ' ')"
        if ($NoServerManagement) {
            Write-Output 'Server management disabled; using the running llama.cpp server as is.'
        } else {
            if (Test-Path -LiteralPath 'server.local.psd1' -PathType Leaf) {
                Write-Output 'Note: server.local.psd1 is no longer used; move its values into the [server] table of the config file (see config.example.toml) and delete it.'
            }
            $profileJson = (& $pythonPath -m audio_transcript server-profile --config $ConfigPath --optional) -join "`n"
            if ($LASTEXITCODE -ne 0) { throw 'The [server] configuration is invalid or no inference backend is usable; see the message above (or use -NoServerManagement).' }
            if ($profileJson.Trim() -eq 'null') {
                Write-Output "No [server] table in $ConfigPath; using the running llama.cpp server as is."
            } else {
                Initialize-Server ($profileJson | ConvertFrom-Json)
            }
        }
        & $pythonPath -u -m audio_transcript doctor --config $ConfigPath
        if ($LASTEXITCODE -ne 0) { throw 'Pipeline dependency/backend validation failed.' }
        $clock = [Diagnostics.Stopwatch]::StartNew()
        # Not piped: the console UI must keep a real terminal. Start-Transcript does not capture
        # native-process output, so the pipeline writes its own pipeline.log and events.jsonl.
        & $pythonPath @pipelineArguments
        $pipelineExit = $LASTEXITCODE
        $clock.Stop()
        Write-Output ("Run finished (UTC): {0}; wall time {1:n1} s" -f [datetime]::UtcNow.ToString('o'), $clock.Elapsed.TotalSeconds)
        Write-Output "Pipeline exit code: $pipelineExit. Inspect output/<source>/<version>/ (see output/catalog.json) and process/<source>/<version>/."
        $latestPath = Join-Path $projectRoot 'process\runs\latest.json'
        $latest = $null
        if (Test-Path -LiteralPath $latestPath -PathType Leaf) {
            try { $latest = Get-Content -LiteralPath $latestPath -Raw | ConvertFrom-Json } catch { $latest = $null }
        }
        $latestStart = [datetime]::MinValue
        if ($null -ne $latest -and [datetime]::TryParse([string]$latest.started_at, [Globalization.CultureInfo]::InvariantCulture, [Globalization.DateTimeStyles]::AdjustToUniversal -bor [Globalization.DateTimeStyles]::AssumeUniversal, [ref]$latestStart) -and $latestStart -ge $scriptStartUtc.AddSeconds(-2)) {
            Write-Output "Pipeline log   : $(Join-Path $latest.process_dir $latest.paths.log)"
            Write-Output "Pipeline events: $(Join-Path $latest.process_dir $latest.paths.events) (follow with: audio-transcript events --follow)"
        } else {
            Write-Output 'Pipeline log and events: not found in process\runs (a custom process_dir? look in <process_dir>\runs\latest.json).'
        }
        Write-Output "Script messages: $runLog"
    } finally { Stop-Transcript | Out-Null }
} finally { Pop-Location }
exit $pipelineExit
