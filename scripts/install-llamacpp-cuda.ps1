[CmdletBinding()]
param(
    [ValidatePattern('^b[0-9]+$')]
    [string]$Tag = 'b11193',

    [ValidatePattern('^[0-9]+\.[0-9]+$')]
    [string]$CudaVersion = '13.4',

    [string]$RuntimeDirectory = 'C:\ProgramData\AI\runtimes\llama.cpp-cuda',

    [string]$CacheDirectory = (Join-Path $env:TEMP 'audio-transcript-runtime')
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$releaseApi = "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/$([uri]::EscapeDataString($Tag))"
$releaseRepository = 'https://github.com/ggml-org/llama.cpp'
$binaryAssetName = "llama-$Tag-bin-win-cuda-$CudaVersion-x64.zip"
$cudaAssetName = "cudart-llama-bin-win-cuda-$CudaVersion-x64.zip"
$manifestName = 'runtime-manifest.json'

function Normalize-Path([string]$Path) {
    return [System.IO.Path]::GetFullPath($Path).TrimEnd('\')
}

function Get-ReleaseAsset([object]$Release, [string]$Name) {
    $matches = @($Release.assets | Where-Object { $_.name -ceq $Name })
    if ($matches.Count -ne 1) {
        throw "Official release $Tag does not contain exactly one asset named '$Name'."
    }
    $asset = $matches[0]
    if ($asset.digest -notmatch '^sha256:([0-9a-fA-F]{64})$') {
        throw "GitHub release API did not provide a SHA-256 digest for '$Name'; refusing an unverified download."
    }
    return [pscustomobject]@{
        Name = [string]$asset.name
        Url = [string]$asset.browser_download_url
        Sha256 = ([regex]::Match([string]$asset.digest, '^sha256:([0-9a-fA-F]{64})$')).Groups[1].Value.ToLowerInvariant()
        Size = [long]$asset.size
    }
}

function Assert-GpuRuntime([string]$Directory) {
    $executable = Join-Path $Directory 'llama-server.exe'
    if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
        throw "CUDA runtime does not contain llama-server.exe: $executable"
    }
    if (-not (Test-Path -LiteralPath (Join-Path $Directory 'ggml-cuda.dll') -PathType Leaf)) {
        throw "CUDA runtime is missing ggml-cuda.dll in $Directory."
    }

    $oldPath = $env:Path
    $oldErrorActionPreference = $ErrorActionPreference
    try {
        $env:Path = "$Directory;$oldPath"
        $ErrorActionPreference = 'Continue'
        $output = (& $executable --list-devices 2>&1 | Out-String)
        $exitCode = $LASTEXITCODE
    } catch {
        throw "Could not run llama-server --list-devices: $($_.Exception.Message)"
    } finally {
        $env:Path = $oldPath
        $ErrorActionPreference = $oldErrorActionPreference
    }
    if ($exitCode -ne 0) {
        throw "llama-server --list-devices failed with exit code $exitCode. Output: $output"
    }
    if ($output -notmatch '(?im)\bCUDA\s*\d+\s*:' -or $output -notmatch '(?i)NVIDIA') {
        throw "No NVIDIA CUDA device was reported by llama-server. CPU or Vulkan fallback is not accepted. Output: $output"
    }
    Write-Output ($output.Trim())
}

function Get-TargetServerProcesses([string]$Directory) {
    $expectedExecutable = Normalize-Path (Join-Path $Directory 'llama-server.exe')
    try {
        $processes = @(Get-CimInstance -ClassName Win32_Process -Filter "Name = 'llama-server.exe'" -ErrorAction Stop)
    } catch {
        throw "Could not check whether llama-server is using the target directory: $($_.Exception.Message)"
    }

    $matching = @()
    foreach ($process in $processes) {
        if ([string]::IsNullOrWhiteSpace([string]$process.ExecutablePath)) {
            throw "Cannot verify the executable path of llama-server PID $($process.ProcessId); refusing to modify a possibly active runtime."
        }
        if ([string]::Equals(
                (Normalize-Path ([string]$process.ExecutablePath)),
                $expectedExecutable,
                [StringComparison]::OrdinalIgnoreCase
            )) {
            $matching += $process
        }
    }
    return $matching
}

function Read-RuntimeManifest([string]$Path) {
    try {
        return (Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json)
    } catch {
        throw "Existing runtime manifest is unreadable; refusing to overwrite the runtime: $Path"
    }
}

function Test-ManifestMatches([object]$Manifest, [object[]]$Assets) {
    if ($Manifest.schemaVersion -ne 1 -or $Manifest.repository -ne $releaseRepository -or
        $Manifest.tag -ne $Tag -or $Manifest.cudaVersion -ne $CudaVersion) {
        return $false
    }
    foreach ($asset in $Assets) {
        $recorded = @($Manifest.assets | Where-Object { $_.name -ceq $asset.Name })
        if ($recorded.Count -ne 1 -or $recorded[0].sha256 -ne $asset.Sha256 -or $recorded[0].url -ne $asset.Url) {
            return $false
        }
    }
    return $true
}

function Test-RuntimeFilesMatchManifest([string]$Directory, [object]$Manifest) {
    $records = @($Manifest.runtimeFiles)
    if ($records.Count -eq 0) {
        return $false
    }
    foreach ($record in $records) {
        if ([string]::IsNullOrWhiteSpace([string]$record.name) -or
            [string]::IsNullOrWhiteSpace([string]$record.sha256)) {
            return $false
        }
        $filePath = Join-Path $Directory ([string]$record.name)
        $root = Normalize-Path $Directory
        $normalizedFilePath = Normalize-Path $filePath
        if (-not $normalizedFilePath.StartsWith(($root + '\'), [StringComparison]::OrdinalIgnoreCase)) {
            return $false
        }
        if (-not (Test-Path -LiteralPath $filePath -PathType Leaf)) {
            return $false
        }
        if ((Get-FileHash -LiteralPath $filePath -Algorithm SHA256).Hash.ToLowerInvariant() -ne
            ([string]$record.sha256).ToLowerInvariant()) {
            return $false
        }
    }
    return $true
}

function Set-UsersReadExecute([string]$Directory) {
    $icacls = Join-Path $env:SystemRoot 'System32\icacls.exe'
    if (-not (Test-Path -LiteralPath $icacls -PathType Leaf)) {
        throw 'icacls.exe was not found; cannot ensure shared runtime read access.'
    }
    $oldErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $output = (& $icacls $Directory '/grant' '*S-1-5-32-545:(OI)(CI)RX' '/T' '/C' 2>&1 | Out-String)
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldErrorActionPreference
    }
    if ($exitCode -ne 0) {
        throw "Could not grant BUILTIN\Users read/execute access to the shared runtime (exit $exitCode). No elevation was attempted. $output"
    }
}

function Get-FlatRuntimeFiles([string[]]$Roots) {
    $files = @()
    foreach ($index in 0..($Roots.Count - 1)) {
        $root = $Roots[$index]
        foreach ($file in (Get-ChildItem -LiteralPath $root -File -Recurse)) {
            $isRuntimeBinary = $file.Extension -in @('.dll', '.exe')
            $isDocumentation = $file.Name -match '^(?i:LICENSE|COPYING|NOTICE|PATENTS|AUTHORS|COPYRIGHT|README)(?:[-_.].*)?$'
            if ($isRuntimeBinary -or $isDocumentation) {
                $files += [pscustomobject]@{
                    File = $file
                    Root = $root
                    Archive = if ($index -eq 0) { 'binary' } else { 'cuda-runtime' }
                    IsRuntimeBinary = $isRuntimeBinary
                }
            }
        }
    }
    if (-not ($files | Where-Object { $_.IsRuntimeBinary -and $_.File.Name -ceq 'llama-server.exe' })) {
        throw 'The official binary archive did not contain llama-server.exe.'
    }
    return $files
}

function Copy-FlatRuntimeFiles([object[]]$Files, [string]$Destination) {
    $byName = @{}
    foreach ($entry in $Files) {
        $file = $entry.File
        if (-not $entry.IsRuntimeBinary) {
            $relative = $file.FullName.Substring($entry.Root.Length).TrimStart([char[]]@([char]92, [char]47))
            $licenseRoot = Join-Path (Join-Path $Destination 'licenses') $entry.Archive
            $licensePath = [System.IO.Path]::GetFullPath((Join-Path $licenseRoot $relative))
            $licensePrefix = (Normalize-Path $licenseRoot) + '\'
            if (-not $licensePath.StartsWith($licensePrefix, [StringComparison]::OrdinalIgnoreCase)) {
                throw "Archive documentation path escapes the license directory: $($file.FullName)"
            }
            $licenseParent = Split-Path -Parent $licensePath
            New-Item -ItemType Directory -Path $licenseParent -Force | Out-Null
            Copy-Item -LiteralPath $file.FullName -Destination $licensePath
            continue
        }
        if ($byName.ContainsKey($file.Name)) {
            $previous = $byName[$file.Name]
            $oldHash = (Get-FileHash -LiteralPath $previous.FullName -Algorithm SHA256).Hash
            $newHash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
            if ($oldHash -ne $newHash) {
                throw "Release archives contain conflicting files named '$($file.Name)'; refusing to choose one silently."
            }
            continue
        }
        $byName[$file.Name] = $file
        Copy-Item -LiteralPath $file.FullName -Destination (Join-Path $Destination $file.Name)
    }
}

try {
    if ([string]::IsNullOrWhiteSpace($env:TEMP)) {
        throw 'TEMP is not set; pass -CacheDirectory explicitly.'
    }
    $RuntimeDirectory = [System.IO.Path]::GetFullPath($RuntimeDirectory)
    $CacheDirectory = [System.IO.Path]::GetFullPath($CacheDirectory)
    $runtimeParent = Split-Path -Parent $RuntimeDirectory
    if ([string]::IsNullOrWhiteSpace($runtimeParent) -or $RuntimeDirectory -eq $runtimeParent) {
        throw "RuntimeDirectory must be a child folder path: $RuntimeDirectory"
    }

    $release = Invoke-RestMethod -Uri $releaseApi -Headers @{
        'Accept' = 'application/vnd.github+json'
        'User-Agent' = 'audio-transcript-runtime-installer'
        'X-GitHub-Api-Version' = '2022-11-28'
    } -TimeoutSec 60
    $assets = @(
        Get-ReleaseAsset -Release $release -Name $binaryAssetName
        Get-ReleaseAsset -Release $release -Name $cudaAssetName
    )

    $running = @(Get-TargetServerProcesses -Directory $RuntimeDirectory)
    $manifestPath = Join-Path $RuntimeDirectory $manifestName
    if (Test-Path -LiteralPath $RuntimeDirectory -PathType Container) {
        if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
            if ($running.Count -gt 0) {
                throw "llama-server PID $($running[0].ProcessId) is running from this directory, which has no installer manifest. Refusing to modify it."
            }
            throw "Runtime directory already exists without a matching installer manifest: $RuntimeDirectory. Choose an empty RuntimeDirectory; nothing was overwritten."
        }
        $manifest = Read-RuntimeManifest -Path $manifestPath
        if (-not (Test-ManifestMatches -Manifest $manifest -Assets $assets)) {
            if ($running.Count -gt 0) {
                throw "llama-server PID $($running[0].ProcessId) is using this runtime. Refusing to overwrite or replace it."
            }
            throw "RuntimeDirectory contains a different llama.cpp build. Choose an empty directory; upgrades never overwrite installed files."
        }
        if (-not (Test-RuntimeFilesMatchManifest -Directory $RuntimeDirectory -Manifest $manifest)) {
            throw "Installed runtime files do not match their manifest. Refusing to repair or overwrite them in place."
        }
        Assert-GpuRuntime -Directory $RuntimeDirectory | Out-Null
        Write-Output "Verified existing llama.cpp CUDA runtime $Tag (CUDA $CudaVersion) at $RuntimeDirectory."
        exit 0
    }
    if ($running.Count -gt 0) {
        throw "llama-server PID $($running[0].ProcessId) is using the target runtime path. Refusing to overwrite a running runtime."
    }

    New-Item -ItemType Directory -Path $CacheDirectory -Force | Out-Null
    try {
        New-Item -ItemType Directory -Path $runtimeParent -Force | Out-Null
    } catch {
        throw "Cannot create shared runtime parent '$runtimeParent'. Grant write access and rerun; this script does not elevate automatically."
    }
    $runtimeParent = (Resolve-Path -LiteralPath $runtimeParent).Path
    $runtimeStage = Join-Path $runtimeParent ('.llama.cpp-cuda-stage-' + [guid]::NewGuid().ToString('N'))
    $extractRoot = Join-Path $CacheDirectory ('llama.cpp-cuda-extract-' + [guid]::NewGuid().ToString('N'))
    $binaryExtract = Join-Path $extractRoot 'binary'
    $cudaExtract = Join-Path $extractRoot 'cuda'
    New-Item -ItemType Directory -Path $runtimeStage | Out-Null
    New-Item -ItemType Directory -Path $binaryExtract | Out-Null
    New-Item -ItemType Directory -Path $cudaExtract | Out-Null

    try {
        $zipPaths = @()
        foreach ($asset in $assets) {
            $zipPath = Join-Path $CacheDirectory $asset.Name
            if (-not (Test-Path -LiteralPath $zipPath -PathType Leaf) -or
                (Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $asset.Sha256) {
                Remove-Item -LiteralPath $zipPath -Force -ErrorAction SilentlyContinue
                Invoke-WebRequest -Uri $asset.Url -OutFile $zipPath -TimeoutSec 3600 -UseBasicParsing
            }
            $actualHash = (Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($actualHash -ne $asset.Sha256) {
                throw "SHA-256 mismatch for '$($asset.Name)'. Expected $($asset.Sha256); got $actualHash."
            }
            $zipPaths += $zipPath
        }

        Expand-Archive -LiteralPath $zipPaths[0] -DestinationPath $binaryExtract
        Expand-Archive -LiteralPath $zipPaths[1] -DestinationPath $cudaExtract
        $flatFiles = @(Get-FlatRuntimeFiles -Roots @($binaryExtract, $cudaExtract))
        Copy-FlatRuntimeFiles -Files $flatFiles -Destination $runtimeStage

        $stagePrefix = (Normalize-Path $runtimeStage) + '\'
        $runtimeFiles = @(Get-ChildItem -LiteralPath $runtimeStage -File -Recurse |
            Where-Object { $_.Name -cne $manifestName } | Sort-Object FullName | ForEach-Object {
            [ordered]@{
                name = $_.FullName.Substring($stagePrefix.Length)
                sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
                size = [long]$_.Length
            }
        })
        $installManifest = [ordered]@{
            schemaVersion = 1
            repository = $releaseRepository
            tag = $Tag
            cudaVersion = $CudaVersion
            assets = @($assets | ForEach-Object {
                [ordered]@{ name = $_.Name; url = $_.Url; sha256 = $_.Sha256; size = $_.Size }
            })
            runtimeFiles = $runtimeFiles
            installedAtUtc = [datetime]::UtcNow.ToString('o')
        }
        $manifestJson = ConvertTo-Json -InputObject $installManifest -Depth 8
        $manifestEncoding = [System.Text.UTF8Encoding]::new($false)
        [System.IO.File]::WriteAllText(
            (Join-Path $runtimeStage $manifestName),
            ($manifestJson + [Environment]::NewLine),
            $manifestEncoding
        )
        Set-UsersReadExecute -Directory $runtimeStage

        if (Test-Path -LiteralPath $RuntimeDirectory) {
            throw "RuntimeDirectory appeared during installation. Refusing to merge or overwrite it: $RuntimeDirectory"
        }
        $resolvedStage = Normalize-Path (Resolve-Path -LiteralPath $runtimeStage).Path
        $resolvedParent = Normalize-Path (Resolve-Path -LiteralPath $runtimeParent).Path
        $resolvedTarget = Normalize-Path $RuntimeDirectory
        $targetParent = Normalize-Path (Split-Path -Parent $resolvedTarget)
        $stagePrefix = $resolvedParent + '\'
        if (-not $resolvedStage.StartsWith($stagePrefix, [StringComparison]::OrdinalIgnoreCase) -or
            $resolvedTarget -eq $resolvedParent -or $targetParent -ne $resolvedParent) {
            throw "Resolved staging or destination paths escaped their intended runtime parent. Stage: $resolvedStage; target: $resolvedTarget; parent: $resolvedParent"
        }
        Move-Item -LiteralPath $resolvedStage -Destination $resolvedTarget
        Assert-GpuRuntime -Directory $RuntimeDirectory | Out-Null
        Write-Output "Installed llama.cpp $Tag with CUDA $CudaVersion at $RuntimeDirectory."
    } finally {
        foreach ($path in @($runtimeStage, $extractRoot)) {
            if (Test-Path -LiteralPath $path) {
                $resolvedPath = Normalize-Path $path
                $allowedParent = if ($path -eq $runtimeStage) { $runtimeParent } else { $CacheDirectory }
                $allowedPrefix = (Normalize-Path $allowedParent) + '\'
                if (-not $resolvedPath.StartsWith($allowedPrefix, [StringComparison]::OrdinalIgnoreCase)) {
                    throw "Refusing to clean a staging path outside its expected parent: $resolvedPath"
                }
                Remove-Item -LiteralPath $resolvedPath -Recurse -Force
            }
        }
    }
} catch {
    Write-Error $_.Exception.Message -ErrorAction Continue
    exit 1
}
