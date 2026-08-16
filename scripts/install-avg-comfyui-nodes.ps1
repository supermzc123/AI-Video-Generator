param(
    [Parameter(Mandatory = $true)]
    [string]$ComfyUIRoot,

    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = "Stop"
$resolvedComfy = (Resolve-Path -LiteralPath $ComfyUIRoot).Path
$resolvedProject = (Resolve-Path -LiteralPath $ProjectRoot).Path
$python = Join-Path $resolvedComfy "venv\Scripts\python.exe"
$source = Join-Path $resolvedProject "comfyui_nodes\ai_video_generator_nodes"
$destination = Join-Path $resolvedComfy "custom_nodes\ai_video_generator_nodes"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "ComfyUI Python does not exist: $python"
}
if (-not (Test-Path -LiteralPath $source -PathType Container)) {
    throw "AVG custom node source does not exist: $source"
}

& $python -m pip install --no-deps -e $resolvedProject
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install AVG Worker dependencies into ComfyUI Python"
}

if (Test-Path -LiteralPath $destination) {
    $item = Get-Item -LiteralPath $destination -Force
    $targets = @($item.Target | ForEach-Object { [System.IO.Path]::GetFullPath($_) })
    if ($item.LinkType -ne "Junction" -or $targets -notcontains $source) {
        throw "Custom node destination already exists and is not the expected junction: $destination"
    }
} else {
    New-Item -ItemType Junction -Path $destination -Target $source | Out-Null
}

Push-Location $resolvedComfy
try {
    $probe = "import ai_video_generator, safetensors; print('AVG Python dependencies verified')"
    & $python -s -c $probe
    if ($LASTEXITCODE -ne 0) {
        throw "AVG Python dependency verification failed"
    }
} finally {
    Pop-Location
}

Write-Output "AVG ComfyUI nodes installed. Restart ComfyUI once to register them."
