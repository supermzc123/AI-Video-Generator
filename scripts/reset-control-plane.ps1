param(
    [string]$DataRoot = (Join-Path $PSScriptRoot "..\data")
)

$resolvedRoot = [System.IO.Path]::GetFullPath($DataRoot)
$database = Join-Path $resolvedRoot "control-plane.db"
if (-not (Test-Path -LiteralPath $database)) {
    Write-Host "No control-plane database exists at $database"
    exit 0
}

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$backup = "$database.$stamp.old"
Move-Item -LiteralPath $database -Destination $backup
foreach ($sidecar in @("$database-wal", "$database-shm")) {
    if (Test-Path -LiteralPath $sidecar) {
        Move-Item -LiteralPath $sidecar -Destination "$sidecar.$stamp.old"
    }
}
Write-Host "Archived old control-plane database to $backup"
Write-Host "Start the control plane to create a clean schema. Media, models, and workflows were not touched."
