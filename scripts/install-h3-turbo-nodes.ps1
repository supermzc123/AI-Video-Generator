param(
    [Parameter(Mandatory = $true)]
    [string]$ComfyUIRoot,

    [string]$Proxy = ""
)

$ErrorActionPreference = "Stop"
$PinnedCommit = "4274783a23afcfdbea3b4876cb79effd6c510785"
$PinnedInitSha256 = "7B01887788919948C5765989E10F17C00400D591748853CC383CCE85B923BF36"
$Repository = "https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo.git"
$resolvedRoot = (Resolve-Path -LiteralPath $ComfyUIRoot).Path
$customNodes = Join-Path $resolvedRoot "custom_nodes"
$destination = Join-Path $customNodes "ComfyUI-MiniMax-H3-Turbo-official"

if (-not (Test-Path -LiteralPath $customNodes -PathType Container)) {
    throw "ComfyUI custom_nodes directory does not exist: $customNodes"
}

$conflicts = Get-ChildItem -LiteralPath $customNodes -Directory | Where-Object {
    $_.FullName -ne $destination -and
    (Test-Path -LiteralPath (Join-Path $_.FullName "__init__.py")) -and
    (Select-String -LiteralPath (Join-Path $_.FullName "__init__.py") `
        -Pattern 'MiniMaxH3TurboLoRA|MiniMaxH3TurboSampler' -Quiet)
}
if ($conflicts) {
    $names = ($conflicts.Name | Sort-Object) -join ", "
    throw "Conflicting Turbo node implementation detected: $names. Disable it manually first."
}

if (Test-Path -LiteralPath $destination) {
    $actualCommit = (& git -C $destination rev-parse HEAD).Trim()
    $initPath = Join-Path $destination "__init__.py"
    if ($LASTEXITCODE -ne 0 -or $actualCommit -ne $PinnedCommit -or
        -not (Test-Path -LiteralPath $initPath -PathType Leaf) -or
        (Get-FileHash -Algorithm SHA256 -LiteralPath $initPath).Hash -ne $PinnedInitSha256) {
        throw "Existing official Turbo nodes are not the pinned verified version: $destination"
    }
    Write-Output "Official MiniMax H3 Turbo nodes are already installed at commit $PinnedCommit (verified)."
    exit 0
}

$staging = Join-Path $customNodes ".h3-turbo-installing-$([guid]::NewGuid().ToString('N'))"
$installed = $false
try {
    $gitArgs = @("clone", "--filter=blob:none", "--no-checkout", $Repository, $staging)
    if ($Proxy) {
        $gitArgs = @("-c", "http.proxy=$Proxy") + $gitArgs
    }
    & git @gitArgs
    if ($LASTEXITCODE -ne 0) { throw "Failed to clone official MiniMax H3 Turbo nodes" }

    & git -C $staging checkout --detach $PinnedCommit
    if ($LASTEXITCODE -ne 0) { throw "Failed to check out pinned Turbo commit" }

    $actualCommit = (& git -C $staging rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $actualCommit -ne $PinnedCommit) {
        throw "Official Turbo checkout did not resolve to pinned commit $PinnedCommit"
    }
    $actualInitSha256 = (Get-FileHash -Algorithm SHA256 `
        -LiteralPath (Join-Path $staging "__init__.py")).Hash
    if ($actualInitSha256 -ne $PinnedInitSha256) {
        throw "Official Turbo node hash mismatch: $actualInitSha256"
    }
    Move-Item -LiteralPath $staging -Destination $destination
    $installed = $true
} finally {
    if (-not $installed -and (Test-Path -LiteralPath $staging)) {
        $resolvedStaging = (Resolve-Path -LiteralPath $staging).Path
        if ((Split-Path -Parent $resolvedStaging) -ne $customNodes) {
            throw "Refusing to clean unexpected Turbo staging path: $resolvedStaging"
        }
        Remove-Item -LiteralPath $resolvedStaging -Recurse -Force
    }
}

Write-Output "Installed official MiniMax H3 Turbo nodes at commit $PinnedCommit (verified)"
