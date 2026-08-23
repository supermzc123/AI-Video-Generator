param(
    [string]$FfmpegSourceDirectory = $env:AIVIDEO_FFMPEG_SOURCE_DIR
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
$resourceRoot = [IO.Path]::GetFullPath((Join-Path $projectRoot "desktop\src-tauri\resources"))
$controlPlaneRoot = [IO.Path]::GetFullPath((Join-Path $resourceRoot "control-plane"))
$ffmpegRoot = [IO.Path]::GetFullPath((Join-Path $resourceRoot "ffmpeg"))
$installerRoot = [IO.Path]::GetFullPath((Join-Path $resourceRoot "installers"))
$buildRoot = [IO.Path]::GetFullPath((Join-Path $projectRoot "build\pyinstaller"))
$manifest = Get-Content -Raw (Join-Path $PSScriptRoot "release-assets.json") | ConvertFrom-Json

function Get-Sha256([string]$Path) {
    $stream = [IO.File]::OpenRead($Path)
    try {
        $algorithm = [Security.Cryptography.SHA256]::Create()
        try {
            return ([BitConverter]::ToString($algorithm.ComputeHash($stream))).Replace("-", "").ToLowerInvariant()
        } finally {
            $algorithm.Dispose()
        }
    } finally {
        $stream.Dispose()
    }
}

foreach ($target in @($controlPlaneRoot, $ffmpegRoot, $installerRoot, $buildRoot)) {
    if (-not $target.StartsWith($projectRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to prepare resources outside the project: $target"
    }
}

New-Item -ItemType Directory -Path $controlPlaneRoot, $ffmpegRoot, $installerRoot, $buildRoot -Force | Out-Null
Get-ChildItem -LiteralPath $controlPlaneRoot -Force |
    Where-Object Name -ne ".gitkeep" |
    Remove-Item -Recurse -Force
Get-ChildItem -LiteralPath $ffmpegRoot -Force |
    Where-Object Name -ne ".gitkeep" |
    Remove-Item -Recurse -Force
Get-ChildItem -LiteralPath $installerRoot -Force |
    Where-Object Name -ne ".gitkeep" |
    Remove-Item -Recurse -Force

$installerScripts = Join-Path $installerRoot "scripts"
$installerNodes = Join-Path $installerRoot "comfyui_nodes\ai_video_generator_nodes"
New-Item -ItemType Directory -Path $installerScripts, $installerNodes -Force | Out-Null
foreach ($name in @(
    "install-avg-comfyui-nodes.ps1",
    "install-h3-motion-context.ps1",
    "install-h3-turbo-nodes.ps1"
)) {
    Copy-Item -LiteralPath (Join-Path $projectRoot "scripts\$name") -Destination $installerScripts
}
Copy-Item -LiteralPath (Join-Path $projectRoot "comfyui_nodes\ai_video_generator_nodes\__init__.py") `
    -Destination $installerNodes
Copy-Item -LiteralPath (Join-Path $projectRoot "src\ai_video_generator\services\conditioning_io.py") `
    -Destination (Join-Path $installerNodes "conditioning_io.py")

$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python environment is missing: $python"
}

& $python -m PyInstaller `
    --noconfirm `
    --clean `
    --onedir `
    --name aivideo-api `
    --distpath (Join-Path $buildRoot "dist") `
    --workpath (Join-Path $buildRoot "work") `
    --specpath (Join-Path $buildRoot "spec") `
    --collect-all faster_whisper `
    --collect-all ctranslate2 `
    --collect-all tokenizers `
    --collect-all av `
    --collect-data ai_video_generator `
    (Join-Path $PSScriptRoot "pyinstaller-entry.py")
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE" }

Copy-Item -Path (Join-Path $buildRoot "dist\aivideo-api\*") `
    -Destination $controlPlaneRoot -Recurse -Force
if (-not (Test-Path -LiteralPath (Join-Path $controlPlaneRoot "aivideo-api.exe"))) {
    throw "Packaged control plane executable was not produced"
}

if (-not $FfmpegSourceDirectory) {
    $ffmpegCommand = Get-Command ffmpeg.exe -ErrorAction SilentlyContinue
    if ($null -eq $ffmpegCommand) {
        throw "FFmpeg is unavailable; set AIVIDEO_FFMPEG_SOURCE_DIR"
    }
    $FfmpegSourceDirectory = Split-Path $ffmpegCommand.Source
}
$ffmpegExe = Join-Path $FfmpegSourceDirectory "ffmpeg.exe"
$ffprobeExe = Join-Path $FfmpegSourceDirectory "ffprobe.exe"
foreach ($item in @($ffmpegExe, $ffprobeExe)) {
    if (-not (Test-Path -LiteralPath $item -PathType Leaf)) {
        throw "Required release tool is missing: $item"
    }
}

$actualFfmpeg = Get-Sha256 $ffmpegExe
$actualFfprobe = Get-Sha256 $ffprobeExe
if ($actualFfmpeg -ne $manifest.ffmpeg_sha256 -or $actualFfprobe -ne $manifest.ffprobe_sha256) {
    throw "FFmpeg binaries do not match the pinned release manifest"
}
Copy-Item -LiteralPath $ffmpegExe, $ffprobeExe -Destination $ffmpegRoot -Force
$ffmpegLicense = Join-Path (Split-Path $FfmpegSourceDirectory) "LICENSE"
if (-not (Test-Path -LiteralPath $ffmpegLicense -PathType Leaf)) {
    throw "FFmpeg LICENSE is missing beside the selected build"
}
Copy-Item -LiteralPath $ffmpegLicense -Destination (Join-Path $ffmpegRoot "LICENSE.txt") -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot "release-assets.json") -Destination $ffmpegRoot -Force

Write-Output "Prepared packaged control plane and pinned FFmpeg resources."
