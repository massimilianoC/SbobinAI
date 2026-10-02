$ErrorActionPreference = 'Stop'
& winget install --id Gyan.FFmpeg --exact --accept-package-agreements --accept-source-agreements --silent --disable-interactivity
if ($LASTEXITCODE -ne 0 -and -not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    throw 'FFmpeg installation failed. Check winget output or supply explicit executable paths.'
}
$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' +
            [Environment]::GetEnvironmentVariable('Path', 'User') + ';' + $env:Path
& ffmpeg -version | Select-Object -First 1
