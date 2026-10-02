[CmdletBinding()]
param(
    [string]$StateDirectory = 'C:\ProgramData\AI\services\qwen2-audio'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$serviceLock = $null

function Normalize-Path([string]$Path) {
    return [System.IO.Path]::GetFullPath($Path).TrimEnd('\')
}

function Write-JsonAtomically([string]$Path, [object]$Value) {
    $temporary = "$Path.$PID.tmp"
    $json = ConvertTo-Json -InputObject $Value -Depth 8
    $encoding = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::WriteAllText($temporary, ($json + [Environment]::NewLine), $encoding)
    if ([System.IO.File]::Exists($Path)) {
        $backup = "$temporary.backup"
        [System.IO.File]::Replace($temporary, $Path, $backup)
        Remove-Item -LiteralPath $backup -Force
    } else {
        [System.IO.File]::Move($temporary, $Path)
    }
}

try {
    $manifestPath = Join-Path $StateDirectory 'server.json'
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
        throw "No server ownership manifest exists at $manifestPath. Refusing to stop any process."
    }
    try {
        $serviceLock = [System.IO.File]::Open(
            (Join-Path $StateDirectory 'server.lock'),
            [System.IO.FileMode]::OpenOrCreate,
            [System.IO.FileAccess]::ReadWrite,
            [System.IO.FileShare]::None
        )
    } catch {
        throw 'Another llama.cpp start or stop operation is in progress for this state directory.'
    }
    try {
        $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    } catch {
        throw "The server ownership manifest is unreadable. Refusing to stop any process: $manifestPath"
    }
    foreach ($required in @('processId', 'executablePath', 'processCreationTimeUtcTicks', 'status')) {
        if ($null -eq $manifest.$required) {
            throw "The server ownership manifest is missing '$required'. Refusing to stop any process."
        }
    }
    if ($manifest.status -eq 'stopped') {
        Write-Output "llama.cpp is already stopped (PID $($manifest.processId))."
        exit 0
    }

    $processId = [int]$manifest.processId
    $processInfo = Get-CimInstance -ClassName Win32_Process -Filter "ProcessId = $processId" -ErrorAction SilentlyContinue
    if ($null -eq $processInfo) {
        $manifest.status = 'stopped'
        $manifest | Add-Member -MemberType NoteProperty -Name stoppedAtUtc -Value ([datetime]::UtcNow.ToString('o')) -Force
        Write-JsonAtomically -Path $manifestPath -Value $manifest
        Write-Output "llama.cpp was already stopped (PID $processId)."
        exit 0
    }
    if ([string]::IsNullOrWhiteSpace($processInfo.ExecutablePath) -or $null -eq $processInfo.CreationDate) {
        throw "Cannot verify PID $processId executable path and creation time. Refusing to stop it."
    }

    $manifestExe = Normalize-Path ([string]$manifest.executablePath)
    $currentExe = Normalize-Path ([string]$processInfo.ExecutablePath)
    $currentCreation = ([datetime]$processInfo.CreationDate).ToUniversalTime().Ticks
    if (-not [string]::Equals($manifestExe, $currentExe, [StringComparison]::OrdinalIgnoreCase) -or
        $currentCreation -ne [long]$manifest.processCreationTimeUtcTicks) {
        throw "PID $processId does not match the manifest executable and creation time. Refusing to stop an unrelated process."
    }

    # Kill through the verified process handle so a rapid PID reuse cannot target a different process.
    $process = [System.Diagnostics.Process]::GetProcessById($processId)
    try {
        $process.Refresh()
        $handleExe = Normalize-Path $process.MainModule.FileName
        $handleCreation = $process.StartTime.ToUniversalTime().Ticks
        if (-not [string]::Equals($manifestExe, $handleExe, [StringComparison]::OrdinalIgnoreCase) -or
            [math]::Abs($handleCreation - [long]$manifest.processCreationTimeUtcTicks) -gt 10000000) {
            throw "PID $processId changed while stop ownership was being verified. Refusing to stop it."
        }
        $process.Kill()
        if (-not $process.WaitForExit(30000)) {
            throw "llama.cpp PID $processId did not exit within 30 seconds."
        }
    } finally {
        $process.Dispose()
    }

    $manifest.status = 'stopped'
    $manifest | Add-Member -MemberType NoteProperty -Name stoppedAtUtc -Value ([datetime]::UtcNow.ToString('o')) -Force
    Write-JsonAtomically -Path $manifestPath -Value $manifest
    Write-Output "Stopped the manifest-owned llama.cpp process (PID $processId)."
    exit 0
} catch {
    Write-Error $_.Exception.Message -ErrorAction Continue
    exit 1
} finally {
    if ($null -ne $serviceLock) {
        $serviceLock.Dispose()
    }
}
