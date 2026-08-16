param(
    [Parameter(Mandatory = $true)]
    [string]$ComfyUIRoot
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
    throw "Destination already exists: $destination"
}

git clone --no-checkout $Repository $destination
git -C $destination checkout --detach $PinnedCommit

$actualCommit = git -C $destination rev-parse HEAD
if ($LASTEXITCODE -ne 0 -or $actualCommit -ne $PinnedCommit) {
    throw "Official Turbo checkout did not resolve to pinned commit $PinnedCommit"
}
$actualInitSha256 = (Get-FileHash -Algorithm SHA256 `
    -LiteralPath (Join-Path $destination "__init__.py")).Hash
if ($actualInitSha256 -ne $PinnedInitSha256) {
    throw "Official Turbo node hash mismatch: $actualInitSha256"
}

Write-Output "Installed official MiniMax H3 Turbo nodes at commit $PinnedCommit (verified)"
