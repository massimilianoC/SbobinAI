[CmdletBinding()]
param(
    [string]$Destination
)

# Downloads the MIT-licensed Silero VAD ONNX model (https://github.com/snakers4/silero-vad)
# from a pinned release tag and verifies its SHA-256 before installing it.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$tag = 'v6.2.3'
$expectedHash = '1A153A22F4509E292A94E67D6F9B85E8DEB25B4988682B7E174C65279D8788E3'
$url = "https://raw.githubusercontent.com/snakers4/silero-vad/$tag/src/silero_vad/data/silero_vad.onnx"

if ([string]::IsNullOrWhiteSpace($Destination)) {
    $repositoryRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
    $Destination = Join-Path $repositoryRoot 'models\silero_vad.onnx'
}
$Destination = [System.IO.Path]::GetFullPath($Destination)
if (Test-Path -LiteralPath $Destination -PathType Leaf) {
    $existingHash = (Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash
    if ($existingHash -ne $expectedHash) {
        throw "A different file already exists at $Destination (SHA-256 $existingHash). Refusing to overwrite it; move it away or pass -Destination."
    }
    Write-Output "Silero VAD model already installed and verified: $Destination"
} else {
    $directory = Split-Path -Parent $Destination
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
    $temporary = "$Destination.$PID.download"
    try {
        Invoke-WebRequest -Uri $url -OutFile $temporary -UseBasicParsing
        $actualHash = (Get-FileHash -LiteralPath $temporary -Algorithm SHA256).Hash
        if ($actualHash -ne $expectedHash) {
            throw "SHA-256 mismatch for the downloaded model: expected $expectedHash, got $actualHash."
        }
        Move-Item -LiteralPath $temporary -Destination $Destination
    } finally {
        Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
    }
    Write-Output "Installed Silero VAD model: $Destination"
}
$provenancePath = Join-Path (Split-Path -Parent $Destination) 'PROVENANCE.txt'
$provenance = @(
    "File:    $([System.IO.Path]::GetFileName($Destination))",
    "Source:  $url",
    "Tag:     $tag",
    "SHA-256: $expectedHash",
    'License: MIT (Silero Team)',
    "Fetched: $([datetime]::UtcNow.ToString('o'))"
)
if (-not (Test-Path -LiteralPath $provenancePath -PathType Leaf)) {
    [System.IO.File]::WriteAllLines($provenancePath, $provenance, [System.Text.UTF8Encoding]::new($false))
}
Write-Output "Source:  $url"
Write-Output "Tag:     $tag"
Write-Output "SHA-256: $expectedHash"
Write-Output 'License: MIT (Silero Team)'
