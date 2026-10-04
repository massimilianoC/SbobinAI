[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ModelPath,

    [Parameter(Mandatory = $true)]
    [string]$ProjectorPath,

    [string]$RuntimeDirectory = 'C:\ProgramData\AI\runtimes\llama.cpp-cuda',
    [string]$StateDirectory = 'C:\ProgramData\AI\services\qwen2-audio',
    [ValidateRange(1, 65535)]
    [int]$Port = 8088,
    [ValidateRange(1, 1800)]
    [int]$TimeoutSeconds = 180,
    [ValidateRange(1, 999)]
    [int]$GpuLayers = 99,
    [ValidateNotNullOrEmpty()]
    [string]$Alias = 'qwen2-audio-7b',
    [ValidateRange(1, 1048576)]
    [int]$ContextSize = 4096,
    [ValidateRange(1, 8)]
    [int]$Parallel = 1,
    # Inference backend of the runtime folder: cuda (default, as before), vulkan or cpu.
    [ValidateSet('cuda', 'vulkan', 'cpu')]
    [string]$Backend = 'cuda',
    # llama.cpp device (CUDA0, Vulkan1, ...). Default: the first device of the backend; none for cpu.
    [string]$Device = '',
    # CPU threads (cpu backend only). 0 = physical cores.
    [ValidateRange(0, 1024)]
    [int]$Threads = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$manifestPath = Join-Path $StateDirectory 'server.json'
$stdoutPath = Join-Path $StateDirectory 'stdout.log'
$stderrPath = Join-Path $StateDirectory 'stderr.log'
$failurePath = Join-Path $StateDirectory 'startup-error.json'
$aliasName = $Alias
$serviceLock = $null

function Resolve-ExistingFile([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Label file does not exist: $Path"
    }
    $resolved = (Resolve-Path -LiteralPath $Path).Path
    if ((Get-Item -LiteralPath $resolved).Length -le 0) {
        throw "$Label file is empty: $resolved"
    }
    return $resolved
}

function Normalize-Path([string]$Path) {
    return [System.IO.Path]::GetFullPath($Path).TrimEnd('\')
}

function ConvertTo-QuotedWindowsArgument([string]$Value) {
    $builder = [System.Text.StringBuilder]::new()
    [void]$builder.Append('"')
    $slashCount = 0
    foreach ($character in $Value.ToCharArray()) {
        if ($character -eq [char]92) {
            $slashCount++
            continue
        }
        if ($character -eq [char]34) {
            [void]$builder.Append([string]::new([char]92, (2 * $slashCount) + 1))
            [void]$builder.Append([char]34)
        } else {
            if ($slashCount -gt 0) {
                [void]$builder.Append([string]::new([char]92, $slashCount))
            }
            [void]$builder.Append($character)
        }
        $slashCount = 0
    }
    if ($slashCount -gt 0) {
        [void]$builder.Append([string]::new([char]92, 2 * $slashCount))
    }
    [void]$builder.Append('"')
    return $builder.ToString()
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

function Read-Manifest([string]$Path) {
    try {
        $manifest = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    } catch {
        throw "The existing server manifest is unreadable. Refusing to start another server: $Path"
    }
    foreach ($required in @('processId', 'executablePath', 'processCreationTimeUtcTicks', 'modelPath', 'projectorPath', 'port', 'alias', 'stdoutPath', 'stderrPath', 'configuration')) {
        if ($null -eq $manifest.$required) {
            throw "The existing server manifest is missing '$required'. Refusing to take ownership."
        }
    }
    # backend/device/threads are optional: manifests written before backend selection mean cuda/CUDA0.
    foreach ($required in @('gpuLayers', 'contextSize', 'parallel', 'mmprojOffload', 'logVerbosity', 'timeoutSeconds')) {
        if ($null -eq $manifest.configuration.$required) {
            throw "The existing server manifest configuration is missing '$required'. Refusing to take ownership."
        }
    }
    return $manifest
}

function Get-ProcessIdentity([int]$ProcessId) {
    $processInfo = Get-CimInstance -ClassName Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction SilentlyContinue
    if ($null -eq $processInfo) {
        return $null
    }
    if ([string]::IsNullOrWhiteSpace($processInfo.ExecutablePath) -or $null -eq $processInfo.CreationDate) {
        throw "Cannot verify executable path and creation time for process $ProcessId."
    }
    $creation = ([datetime]$processInfo.CreationDate).ToUniversalTime().Ticks
    return [pscustomobject]@{
        ProcessId = [int]$processInfo.ProcessId
        ExecutablePath = Normalize-Path $processInfo.ExecutablePath
        CreationTimeUtcTicks = [long]$creation
    }
}

function Test-ManifestProcess([object]$Manifest) {
    $identity = Get-ProcessIdentity -ProcessId ([int]$Manifest.processId)
    if ($null -eq $identity) {
        return $false
    }
    $expectedExe = Normalize-Path ([string]$Manifest.executablePath)
    if (-not [string]::Equals($identity.ExecutablePath, $expectedExe, [StringComparison]::OrdinalIgnoreCase) -or
        $identity.CreationTimeUtcTicks -ne [long]$Manifest.processCreationTimeUtcTicks) {
        throw "PID $($Manifest.processId) is now a different process. Refusing to adopt or stop it."
    }
    return $true
}

function Get-ListenerOwner([int]$ListenPort) {
    $getConnections = Get-Command -Name Get-NetTCPConnection -ErrorAction SilentlyContinue
    if ($null -ne $getConnections) {
        $listeners = @(Get-NetTCPConnection -State Listen -LocalPort $ListenPort -ErrorAction SilentlyContinue)
        if ($listeners.Count -gt 0) {
            return [int]$listeners[0].OwningProcess
        }
        return $null
    }

    $probe = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $ListenPort)
    try {
        $probe.Start()
        return $null
    } catch [System.Net.Sockets.SocketException] {
        return -1
    } finally {
        $probe.Stop()
    }
}

function Test-ServerReady([string]$Url) {
    # Plain .NET request: Windows PowerShell 5.1 transcripts record even caught
    # Invoke-RestMethod failures as "fatal errors" while the model is loading (503).
    $response = $null
    try {
        $request = [System.Net.WebRequest]::Create($Url)
        $request.Method = 'GET'
        $request.Timeout = 3000
        $response = $request.GetResponse()
        $reader = New-Object System.IO.StreamReader($response.GetResponseStream())
        $health = $reader.ReadToEnd() | ConvertFrom-Json
        return ($health.status -eq 'ok')
    } catch {
        return $false
    } finally {
        if ($null -ne $response) { $response.Close() }
    }
}

function Get-ConfigurationValue([object]$Configuration, [string]$Name, [object]$Default) {
    $property = $Configuration.PSObject.Properties[$Name]
    if ($null -eq $property -or $null -eq $property.Value) {
        return $Default
    }
    return $property.Value
}

function Get-PhysicalCoreCount {
    try {
        $cores = (Get-CimInstance -ClassName Win32_Processor -ErrorAction Stop | Measure-Object -Property NumberOfCores -Sum).Sum
        if ($cores -gt 0) { return [int]$cores }
    } catch {
        # fall through to the logical processor estimate
    }
    return [Math]::Max(1, [int]([Environment]::ProcessorCount / 2))
}

function Test-StartupLog([string]$BackendName, [string]$Stdout, [string]$Stderr) {
    $combined = ''
    foreach ($logPath in @($Stdout, $Stderr)) {
        if (Test-Path -LiteralPath $logPath -PathType Leaf) {
            $combined += Get-Content -LiteralPath $logPath -Raw -ErrorAction SilentlyContinue
        }
    }
    switch ($BackendName) {
        'cuda' {
            return ($combined -match '(?i)(ggml-cuda\.dll|CUDA\d+\s*:|found\s+\d+\s+CUDA\s+devices|NVIDIA.{0,80}(GPU|CUDA)|offload(?:ed|ing).{0,80}(GPU|CUDA))')
        }
        'vulkan' {
            return ($combined -match '(?i)(ggml_vulkan|ggml-vulkan\.dll|Vulkan\d+\s*:|Vulkan\d+\s+(model|compute|KV)\s+buffer|offload(?:ed|ing).{0,80}(GPU|Vulkan))')
        }
        default {
            # CPU: llama.cpp reports 0 offloaded layers and CPU buffers; never a hard failure.
            return ($combined -match '(?i)(offloaded\s+0\s*/\s*\d+\s+layers|CPU(_Mapped)?\s+model\s+buffer)')
        }
    }
}

try {
    $RuntimeDirectory = (Resolve-Path -LiteralPath $RuntimeDirectory -ErrorAction Stop).Path
    $executablePath = Resolve-ExistingFile (Join-Path $RuntimeDirectory 'llama-server.exe') 'llama-server executable'
    $ModelPath = Resolve-ExistingFile $ModelPath 'Model'
    $ProjectorPath = Resolve-ExistingFile $ProjectorPath 'Projector'
    $ModelPath = Normalize-Path $ModelPath
    $ProjectorPath = Normalize-Path $ProjectorPath
    $executablePath = Normalize-Path $executablePath

    # Per-backend readiness. cuda and vulkan need their ggml DLL and a listed device; cpu needs
    # neither. The caller (process-input.ps1) chooses the backend; this script never falls back.
    $gpuLayersEffective = $GpuLayers
    $mmprojOffloadEffective = $true
    $threadsEffective = 0
    if ($Backend -eq 'cpu') {
        if (-not [string]::IsNullOrWhiteSpace($Device) -and $Device -ne 'none') {
            throw "-Device must be 'none' (or omitted) for the cpu backend, got '$Device'."
        }
        $Device = 'none'
        $gpuLayersEffective = 0
        $mmprojOffloadEffective = $false
        $threadsEffective = if ($Threads -gt 0) { $Threads } else { Get-PhysicalCoreCount }
    } else {
        $backendLabel = if ($Backend -eq 'cuda') { 'CUDA' } else { 'Vulkan' }
        $devicePattern = "(?im)^\s*${backendLabel}\d+\s*:"
        if ([string]::IsNullOrWhiteSpace($Device)) { $Device = "${backendLabel}0" }
        if ($Device -notmatch "^${backendLabel}\d+$") {
            throw "-Device '$Device' does not belong to the $Backend backend (expected ${backendLabel}0, ${backendLabel}1, ...)."
        }
        $backendDllPath = Join-Path $RuntimeDirectory "ggml-$Backend.dll"
        if (-not (Test-Path -LiteralPath $backendDllPath -PathType Leaf)) {
            throw "The $Backend backend needs ggml-$Backend.dll, but it is missing: $backendDllPath"
        }
        $savedErrorActionPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = 'Continue'
            $deviceOutput = (& $executablePath --list-devices 2>&1 | Out-String)
            $deviceExitCode = $LASTEXITCODE
        } finally {
            $ErrorActionPreference = $savedErrorActionPreference
        }
        if ($deviceExitCode -ne 0) {
            throw "Could not enumerate $backendLabel devices (exit $deviceExitCode): $deviceOutput"
        }
        if ($deviceOutput -notmatch $devicePattern) {
            throw "No $backendLabel device was reported by llama-server --list-devices. The $Backend backend is not usable on this machine. Output: $deviceOutput"
        }
        if ($deviceOutput -notmatch "(?im)^\s*$([regex]::Escape($Device))\s*:") {
            throw "Device $Device was not reported by llama-server --list-devices. Output: $deviceOutput"
        }
        if ($Backend -eq 'cuda' -and $deviceOutput -notmatch '(?i)NVIDIA') {
            throw "No NVIDIA CUDA device was reported by llama-server --list-devices. Output: $deviceOutput"
        }
    }

    New-Item -ItemType Directory -Path $StateDirectory -Force | Out-Null
    $StateDirectory = (Resolve-Path -LiteralPath $StateDirectory).Path
    $serviceLockPath = Join-Path $StateDirectory 'server.lock'
    try {
        $serviceLock = [System.IO.File]::Open(
            $serviceLockPath,
            [System.IO.FileMode]::OpenOrCreate,
            [System.IO.FileAccess]::ReadWrite,
            [System.IO.FileShare]::None
        )
    } catch {
        throw 'Another llama.cpp start or stop operation is in progress for this state directory.'
    }
    $manifestPath = Join-Path $StateDirectory 'server.json'
    $stdoutPath = Join-Path $StateDirectory 'stdout.log'
    $stderrPath = Join-Path $StateDirectory 'stderr.log'
    $failurePath = Join-Path $StateDirectory 'startup-error.json'

    $configuration = [ordered]@{
        executablePath = $executablePath
        modelPath = $ModelPath
        projectorPath = $ProjectorPath
        port = $Port
        alias = $aliasName
        gpuLayers = $gpuLayersEffective
        contextSize = $ContextSize
        parallel = $Parallel
        mmprojOffload = $mmprojOffloadEffective
        backend = $Backend
        device = $Device
        threads = $threadsEffective
        logVerbosity = 4
        timeoutSeconds = $TimeoutSeconds
    }

    if (Test-Path -LiteralPath $manifestPath -PathType Leaf) {
        $existing = Read-Manifest $manifestPath
        if (Test-ManifestProcess $existing) {
            $sameConfiguration =
                [string]::Equals((Normalize-Path ([string]$existing.executablePath)), $executablePath, [StringComparison]::OrdinalIgnoreCase) -and
                [string]::Equals((Normalize-Path ([string]$existing.modelPath)), $ModelPath, [StringComparison]::OrdinalIgnoreCase) -and
                [string]::Equals((Normalize-Path ([string]$existing.projectorPath)), $ProjectorPath, [StringComparison]::OrdinalIgnoreCase) -and
                [int]$existing.port -eq $Port -and [string]$existing.alias -eq $aliasName -and
                [int]$existing.configuration.gpuLayers -eq $gpuLayersEffective -and
                [int]$existing.configuration.contextSize -eq $ContextSize -and
                [int]$existing.configuration.parallel -eq $Parallel -and
                [bool]$existing.configuration.mmprojOffload -eq $mmprojOffloadEffective -and
                [string](Get-ConfigurationValue $existing.configuration 'backend' 'cuda') -eq $Backend -and
                [string](Get-ConfigurationValue $existing.configuration 'device' (Get-ConfigurationValue $existing.configuration 'cudaDevice' 'CUDA0')) -eq $Device -and
                [int](Get-ConfigurationValue $existing.configuration 'threads' 0) -eq [int]$threadsEffective -and
                [int]$existing.configuration.logVerbosity -eq 4 -and
                [int]$existing.configuration.timeoutSeconds -eq $TimeoutSeconds
            if (-not $sameConfiguration) {
                throw "An owned llama.cpp server is already running with a different configuration. Stop it before changing models or settings."
            }
            $listenerOwner = Get-ListenerOwner -ListenPort $Port
            if ($null -ne $listenerOwner -and $listenerOwner -ne -1 -and $listenerOwner -ne [int]$existing.processId) {
                throw "Port $Port is served by PID $listenerOwner, not the process in the manifest. Refusing to proceed."
            }
            if ((Test-ServerReady "http://127.0.0.1:$Port/health") -and
                (Test-StartupLog -BackendName $Backend -Stdout ([string]$existing.stdoutPath) -Stderr ([string]$existing.stderrPath))) {
                $existing.status = 'ready'
                $existing.updatedAtUtc = [datetime]::UtcNow.ToString('o')
                Write-JsonAtomically -Path $manifestPath -Value $existing
                Remove-Item -LiteralPath $failurePath -Force -ErrorAction SilentlyContinue
                Write-Output "llama.cpp is already running and ready (PID $($existing.processId))."
                exit 0
            }
            Write-Output "Owned llama.cpp server is still starting; waiting for health readiness."
            $manifest = $existing
        } else {
            Remove-Item -LiteralPath $manifestPath -Force
            $manifest = $null
        }
    } else {
        $manifest = $null
    }

    if ($null -eq $manifest) {
        $listenerOwner = Get-ListenerOwner -ListenPort $Port
        if ($null -ne $listenerOwner) {
            if ($listenerOwner -eq -1) {
                throw "Port $Port is already occupied. No process identity was available; refusing to start or stop anything."
            }
            throw "Port $Port is already occupied by PID $listenerOwner. It is not owned by this service manifest; refusing to touch it."
        }

        $arguments = @(
            '-m', $ModelPath,
            '--mmproj', $ProjectorPath,
            '-ngl', [string]$gpuLayersEffective,
            '--device', $Device
        )
        if ($Backend -eq 'cpu') {
            # CPU inference: nothing offloaded, projector on the CPU, explicit thread count.
            $arguments += @('--no-mmproj-offload', '-t', [string]$threadsEffective)
        }
        $arguments += @(
            '-c', [string]$ContextSize,
            '--host', '127.0.0.1',
            '--port', [string]$Port,
            '-np', [string]$Parallel,
            '--alias', $aliasName,
            '--timeout', [string]$TimeoutSeconds,
            '-lv', '4'
        )
        $argumentLine = (($arguments | ForEach-Object { ConvertTo-QuotedWindowsArgument ([string]$_) }) -join ' ')
        Remove-Item -LiteralPath $failurePath -Force -ErrorAction SilentlyContinue
        $started = Start-Process -FilePath $executablePath -ArgumentList $argumentLine `
            -WorkingDirectory $RuntimeDirectory -RedirectStandardOutput $stdoutPath `
            -RedirectStandardError $stderrPath -WindowStyle Hidden -PassThru
        if ($null -eq $started) {
            throw 'Start-Process did not return a process handle.'
        }
        Start-Sleep -Milliseconds 250
        try {
            $processIdentity = Get-ProcessIdentity -ProcessId $started.Id
        } catch {
            if (-not $started.HasExited) {
                $started.Kill()
                $started.WaitForExit(10000) | Out-Null
            }
            throw
        }
        if ($null -eq $processIdentity) {
            throw "llama-server exited before its process identity could be recorded. See $stderrPath"
        }
        $manifest = [ordered]@{
            schemaVersion = 1
            status = 'starting'
            processId = $processIdentity.ProcessId
            executablePath = $processIdentity.ExecutablePath
            processCreationTimeUtcTicks = $processIdentity.CreationTimeUtcTicks
            modelPath = $ModelPath
            projectorPath = $ProjectorPath
            port = $Port
            alias = $aliasName
            runtimeDirectory = $RuntimeDirectory
            stateDirectory = $StateDirectory
            stdoutPath = $stdoutPath
            stderrPath = $stderrPath
            arguments = $arguments
            configuration = $configuration
            startedAtUtc = [datetime]::UtcNow.ToString('o')
            updatedAtUtc = [datetime]::UtcNow.ToString('o')
        }
        Write-JsonAtomically -Path $manifestPath -Value $manifest
        Write-Output "Started llama.cpp server (PID $($manifest.processId)); waiting for readiness."
    }

    $deadline = [datetime]::UtcNow.AddSeconds($TimeoutSeconds)
    while ([datetime]::UtcNow -lt $deadline) {
        if (-not (Test-ManifestProcess $manifest)) {
            throw "llama-server exited before health readiness. See $stderrPath"
        }
        $listenerOwner = Get-ListenerOwner -ListenPort $Port
        if ($null -ne $listenerOwner -and $listenerOwner -ne -1 -and $listenerOwner -ne [int]$manifest.processId) {
            throw "Port $Port is now served by PID $listenerOwner instead of the owned llama-server."
        }
        if (Test-ServerReady "http://127.0.0.1:$Port/health") {
            if (-not (Test-StartupLog -BackendName $Backend -Stdout ([string]$manifest.stdoutPath) -Stderr ([string]$manifest.stderrPath))) {
                if ($Backend -eq 'cpu') {
                    Write-Warning "Server health is ready but the logs do not show the expected CPU buffers. See $stdoutPath"
                } else {
                    throw "Server health is ready but logs do not confirm $Backend initialization. See $stdoutPath and $stderrPath"
                }
            }
            $manifest.status = 'ready'
            $manifest.updatedAtUtc = [datetime]::UtcNow.ToString('o')
            Write-JsonAtomically -Path $manifestPath -Value $manifest
            Remove-Item -LiteralPath $failurePath -Force -ErrorAction SilentlyContinue
            Write-Output "llama.cpp is ready at http://127.0.0.1:$Port (PID $($manifest.processId)); backend $Backend, device $Device."
            exit 0
        }
        Start-Sleep -Seconds 1
    }
    throw "Timed out after $TimeoutSeconds seconds waiting for llama.cpp health readiness. See $stderrPath"
} catch {
    $message = $_.Exception.Message
    Write-Error $message -ErrorAction Continue
    try {
        New-Item -ItemType Directory -Path $StateDirectory -Force -ErrorAction SilentlyContinue | Out-Null
        $failure = [ordered]@{
            status = 'failed'
            message = $message
            updatedAtUtc = [datetime]::UtcNow.ToString('o')
        }
        Write-JsonAtomically -Path $failurePath -Value $failure
    } catch {
        Write-Error "Could not write startup failure state: $($_.Exception.Message)" -ErrorAction Continue
    }
    exit 1
} finally {
    if ($null -ne $serviceLock) {
        $serviceLock.Dispose()
    }
}
