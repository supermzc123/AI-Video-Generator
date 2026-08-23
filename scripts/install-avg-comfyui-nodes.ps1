param(
    [Parameter(Mandatory = $true)]
    [string]$ComfyUIRoot,

    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = "Stop"
$resolvedComfy = (Resolve-Path -LiteralPath $ComfyUIRoot).Path
$resolvedProject = (Resolve-Path -LiteralPath $ProjectRoot).Path
$pythonCandidates = @(
    (Join-Path $resolvedComfy "venv\Scripts\python.exe"),
    (Join-Path $resolvedComfy ".venv\Scripts\python.exe"),
    (Join-Path (Split-Path -Parent $resolvedComfy) "python_embeded\python.exe"),
    (Join-Path $resolvedComfy "python_embeded\python.exe")
)
$python = $pythonCandidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
$source = Join-Path $resolvedProject "comfyui_nodes\ai_video_generator_nodes"
$destination = Join-Path $resolvedComfy "custom_nodes\ai_video_generator_nodes"

if (-not $python) {
    throw "ComfyUI Python was not found in venv, .venv, or python_embeded"
}
if (-not (Test-Path -LiteralPath $source -PathType Container)) {
    throw "AVG custom node source does not exist: $source"
}

$projectMetadata = Join-Path $resolvedProject "pyproject.toml"
$standaloneRuntime = Join-Path $source "conditioning_io.py"
if (Test-Path -LiteralPath $projectMetadata -PathType Leaf) {
    & $python -m pip install --no-deps -e $resolvedProject
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install AVG Worker package into ComfyUI Python"
    }
} elseif (-not (Test-Path -LiteralPath $standaloneRuntime -PathType Leaf)) {
    throw "Packaged AVG conditioning runtime is missing"
}

if (Test-Path -LiteralPath $destination) {
    $item = Get-Item -LiteralPath $destination -Force
    $targets = @($item.Target | ForEach-Object { [System.IO.Path]::GetFullPath($_) })
    $filesToVerify = @("__init__.py")
    if (Test-Path -LiteralPath $standaloneRuntime -PathType Leaf) {
        $filesToVerify += "conditioning_io.py"
    }
    $filesMatch = $true
    foreach ($name in $filesToVerify) {
        $sourceFile = Join-Path $source $name
        $installedFile = Join-Path $destination $name
        if (-not (Test-Path -LiteralPath $installedFile -PathType Leaf) -or
            (Get-FileHash -Algorithm SHA256 -LiteralPath $installedFile).Hash -ne
            (Get-FileHash -Algorithm SHA256 -LiteralPath $sourceFile).Hash) {
            $filesMatch = $false
            break
        }
    }
    if (($item.LinkType -ne "Junction" -or $targets -notcontains $source) -and
        -not $filesMatch) {
        throw "Custom node destination exists but does not match this application version: $destination"
    }
} else {
    Copy-Item -LiteralPath $source -Destination $destination -Recurse
}

Push-Location $resolvedComfy
try {
    $probe = "import safetensors; print('AVG Python dependencies verified')"
    & $python -s -c $probe
    if ($LASTEXITCODE -ne 0) {
        throw "AVG Python dependency verification failed"
    }
} finally {
    Pop-Location
}

Write-Output "AVG ComfyUI nodes installed and verified. Restart ComfyUI once to register them."
