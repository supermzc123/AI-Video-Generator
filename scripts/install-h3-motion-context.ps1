param(
    [Parameter(Mandatory = $true)]
    [string]$ComfyUIRoot,

    [string]$Proxy = ""
)

$ErrorActionPreference = "Stop"
$PinnedCommit = "725a731e644c669601799da1eb63f4e7497c628f"
$Repository = "https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context.git"
$ExpectedHashes = @{
    "__init__.py" = "92D4E0304D94F09226D6415F7218E9F23748FA94B85DEA18AB30574C1BF8A263"
    "nodes.py" = "318425778B9AC01D3D4620316F20A630E1405E4377AB37330C719FFF3C6A4AB4"
    "patch_layout.py" = "A489FB5D9BDF3F67FF6B847BAAB04CE2D4FFBA62F7095ED1E262A11BDEDE210B"
    "patch_payload.py" = "0DA2D1D6D718A6AA8349C52CB530EAA15CDB5D771B984D8678FAAF00B035C4BF"
    "probe_node.py" = "DCB75AB3992CA605C25C8278050BD40328847158AE06088F1B53C0806D85B559"
    "pyproject.toml" = "72238F9C03CCF91BF2ABC91EE660C34F65F02871E2C96837025A93A07110A7C8"
}

$resolvedRoot = (Resolve-Path -LiteralPath $ComfyUIRoot).Path
$customNodes = Join-Path $resolvedRoot "custom_nodes"
$destination = Join-Path $customNodes "ComfyUI-H3-Motion-Context"
if (-not (Test-Path -LiteralPath $customNodes -PathType Container)) {
    throw "ComfyUI custom_nodes directory does not exist: $customNodes"
}
if (Test-Path -LiteralPath $destination) {
    $actualCommit = (& git -C $destination rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $actualCommit -ne $PinnedCommit) {
        throw "Existing H3 Motion Context is not the pinned version: $destination"
    }
    foreach ($entry in $ExpectedHashes.GetEnumerator()) {
        $sourcePath = Join-Path $destination $entry.Key
        if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf) -or
            (Get-FileHash -Algorithm SHA256 -LiteralPath $sourcePath).Hash -ne $entry.Value) {
            throw "Existing H3 Motion Context failed file verification: $($entry.Key)"
        }
    }
    Write-Output "H3 Motion Context is already installed at pinned commit $PinnedCommit (verified)."
    exit 0
}

$patchPatterns = @(
    "MiniMaxH3MotionContext",
    "motion_context_index",
    "PackedLayout\.__init__",
    "MiniMaxH3\.extra_conds"
)
$conflicts = @()
foreach ($package in Get-ChildItem -LiteralPath $customNodes -Directory) {
    if ($package.Name.StartsWith(".") -or $package.FullName -eq $destination) {
        continue
    }
    $pythonFiles = Get-ChildItem -LiteralPath $package.FullName -Recurse -File `
        -Filter "*.py" -ErrorAction SilentlyContinue
    if ($pythonFiles -and (Select-String -LiteralPath $pythonFiles.FullName `
            -Pattern $patchPatterns -Quiet)) {
        $conflicts += $package.Name
    }
}
if ($conflicts) {
    $names = ($conflicts | Sort-Object -Unique) -join ", "
    throw "Conflicting H3 Motion Context patch owner detected: $names. Disable it manually first."
}

$stagingName = ".h3-motion-context-installing-$([guid]::NewGuid().ToString('N'))"
$staging = Join-Path $customNodes $stagingName
$installed = $false
try {
    $gitArgs = @("clone", "--filter=blob:none", "--no-checkout", $Repository, $staging)
    if ($Proxy) {
        $gitArgs = @("-c", "http.proxy=$Proxy") + $gitArgs
    }
    & git @gitArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to clone H3 Motion Context"
    }

    & git -C $staging checkout --detach $PinnedCommit
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to check out pinned H3 Motion Context commit"
    }
    $actualCommit = (& git -C $staging rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $actualCommit -ne $PinnedCommit) {
        throw "H3 Motion Context checkout did not resolve to pinned commit $PinnedCommit"
    }

    foreach ($entry in $ExpectedHashes.GetEnumerator()) {
        $sourcePath = Join-Path $staging $entry.Key
        $actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $sourcePath).Hash
        if ($actualHash -ne $entry.Value) {
            throw "H3 Motion Context hash mismatch for $($entry.Key): $actualHash"
        }
    }

    Move-Item -LiteralPath $staging -Destination $destination
    $installed = $true
}
finally {
    if (-not $installed -and (Test-Path -LiteralPath $staging)) {
        $resolvedStaging = (Resolve-Path -LiteralPath $staging).Path
        $resolvedCustomNodes = (Resolve-Path -LiteralPath $customNodes).Path
        if ((Split-Path -Parent $resolvedStaging) -ne $resolvedCustomNodes) {
            throw "Refusing to clean unexpected staging path: $resolvedStaging"
        }
        Remove-Item -LiteralPath $resolvedStaging -Recurse -Force
    }
}

Write-Output "Installed H3 Motion Context at pinned commit $PinnedCommit (verified)."
Write-Output "Restart ComfyUI manually, then verify all five MiniMaxH3MotionContext nodes in /object_info."
