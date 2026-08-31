param(
    [ValidateRange(5, 120)]
    [int]$TimeoutSeconds = 30
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
$apiExecutable = Join-Path $projectRoot ".venv\Scripts\aivideo-api.exe"
$apiPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$healthUrl = "http://127.0.0.1:8000/api/v1/health"
$logRoot = Join-Path $projectRoot "data\logs"
$currentSchemaVersion = "6"

if (-not (Test-Path -LiteralPath $apiPython -PathType Leaf) -and -not (Test-Path -LiteralPath $apiExecutable -PathType Leaf)) {
    throw "Backend runtime does not exist. Run scripts\setup.ps1 first."
}

# Resolve the same data root the backend will use. It may be outside the
# repository (the default is the per-user application data directory).
$databasePath = $null
if (Test-Path -LiteralPath $apiPython -PathType Leaf) {
    $resolvedDataRoot = (& $apiPython -c "from ai_video_generator.config import Settings; print(Settings().data_root)")
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($resolvedDataRoot)) {
        throw "Unable to resolve the backend data root."
    }
    $databasePath = Join-Path ([string]$resolvedDataRoot).Trim() "control-plane.db"
} else {
    # Packaged launchers use the configured repository data directory.
    $databasePath = Join-Path $projectRoot "data\control-plane.db"
}

function Get-ControlPlaneSchemaVersion {
    if (-not (Test-Path -LiteralPath $databasePath -PathType Leaf)) {
        return $null
    }
    if (-not (Test-Path -LiteralPath $apiPython -PathType Leaf)) {
        return $null
    }
    $script = @'
import sqlite3, sys
try:
    with sqlite3.connect(sys.argv[1], timeout=2) as db:
        row = db.execute("SELECT value FROM schema_metadata WHERE key='schema_version'").fetchone()
        print(row[0] if row else "")
except Exception:
    print("__invalid__")
'@
    $previousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $result = (& $apiPython -c $script $databasePath 2>$null | Select-Object -Last 1)
    } finally {
        $ErrorActionPreference = $previousErrorAction
    }
    if ($null -eq $result) { return "__invalid__" }
    return ([string]$result).Trim()
}

function Remove-IncompatibleControlPlane {
    $schema = Get-ControlPlaneSchemaVersion
    # A locked/unreadable database must never be deleted speculatively.
    if ($null -eq $schema -or $schema -eq "" -or $schema -eq "__invalid__" -or $schema -eq $currentSchemaVersion) {
        return
    }

    Remove-Item -LiteralPath $databasePath -Force
    foreach ($sidecar in @("$databasePath-wal", "$databasePath-shm")) {
        if (Test-Path -LiteralPath $sidecar -PathType Leaf) {
            Remove-Item -LiteralPath $sidecar -Force
        }
    }
    Write-Output "Removed incompatible test control-plane schema $schema at $databasePath"
    Write-Output "A clean schema $currentSchemaVersion database will be created on startup."
}

Remove-IncompatibleControlPlane

function Test-ProjectBackendProcess {
    param([Parameter(Mandatory = $true)][int]$ProcessId)

    $processInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId"
    if ($null -eq $processInfo) {
        return $false
    }
    $executablePath = [string]$processInfo.ExecutablePath
    $commandLine = [string]$processInfo.CommandLine
    $venvRoot = Join-Path $projectRoot ".venv"
    $usesExactProjectLauncher = $commandLine.IndexOf(
        $apiExecutable,
        [System.StringComparison]::OrdinalIgnoreCase
    ) -ge 0
    $usesSourceLauncher = $commandLine.IndexOf(
        "ai_video_generator.main",
        [System.StringComparison]::OrdinalIgnoreCase
    ) -ge 0
    $usesProjectVenv = (
        $executablePath.StartsWith($venvRoot, [System.StringComparison]::OrdinalIgnoreCase) -and
        (
            $executablePath.EndsWith("aivideo-api.exe", [System.StringComparison]::OrdinalIgnoreCase) -or
            $commandLine.Contains("ai_video_generator") -or
            $commandLine.Contains("aivideo-api")
        )
    )
    return $usesExactProjectLauncher -or $usesSourceLauncher -or $usesProjectVenv
}

$listeners = @(
    Get-NetTCPConnection -LocalAddress "127.0.0.1" -LocalPort 8000 -State Listen `
        -ErrorAction SilentlyContinue
)
$ownerIds = @($listeners | Select-Object -ExpandProperty OwningProcess -Unique)

# The launcher can leave a Python reloader/child process listening after its
# parent exits. Stop every project-owned control-plane process, not only the
# process reported by the socket, so a restart cannot silently keep old code.
$projectProcessIds = @(
    Get-CimInstance Win32_Process |
        Where-Object {
            (Test-ProjectBackendProcess -ProcessId ([int]$_.ProcessId)) -and
            ([string]$_.CommandLine).Contains("ai_video_generator.main")
        } |
        Select-Object -ExpandProperty ProcessId -Unique
)
$ownerIds = @($ownerIds + $projectProcessIds | Sort-Object -Unique)

foreach ($ownerId in $ownerIds) {
    if (-not (Test-ProjectBackendProcess -ProcessId $ownerId)) {
        $owner = Get-CimInstance Win32_Process -Filter "ProcessId = $ownerId"
        throw "Port 8000 is owned by an unrelated process (PID $ownerId): $($owner.CommandLine)"
    }
}

foreach ($ownerId in $ownerIds) {
    Write-Output "Stopping AI Video Generator backend (PID $ownerId)..."
    Stop-Process -Id $ownerId -Force
}

$stopDeadline = [DateTime]::UtcNow.AddSeconds(10)
do {
    $stillListening = Get-NetTCPConnection -LocalAddress "127.0.0.1" -LocalPort 8000 `
        -State Listen -ErrorAction SilentlyContinue
    if ($null -eq $stillListening) {
        break
    }
    Start-Sleep -Milliseconds 200
} while ([DateTime]::UtcNow -lt $stopDeadline)

if ($null -ne $stillListening) {
    throw "Port 8000 did not become available after stopping the old backend."
}

New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$stdoutLog = Join-Path $logRoot "control-plane-$timestamp.stdout.log"
$stderrLog = Join-Path $logRoot "control-plane-$timestamp.stderr.log"

$startArguments = @{
    WorkingDirectory = $projectRoot
    WindowStyle = "Hidden"
    RedirectStandardOutput = $stdoutLog
    RedirectStandardError = $stderrLog
    PassThru = $true
}
if (Test-Path -LiteralPath $apiPython -PathType Leaf) {
    $process = Start-Process -FilePath $apiPython -ArgumentList @("-m", "ai_video_generator.main") @startArguments
} else {
    $process = Start-Process -FilePath $apiExecutable @startArguments
}

Write-Output "Starting AI Video Generator backend (PID $($process.Id))..."
$deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
do {
    if ($process.HasExited) {
        throw "Backend exited during startup. See $stderrLog"
    }
    try {
        $health = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 2
        if ($health.status -eq "ok") {
            Write-Output "Backend is online: $healthUrl"
            Write-Output "Logs: $stdoutLog and $stderrLog"
            exit 0
        }
    } catch {
        # The listener may not be ready yet.
    }
    Start-Sleep -Milliseconds 500
} while ([DateTime]::UtcNow -lt $deadline)

throw "Backend health check timed out after $TimeoutSeconds seconds. See $stderrLog"
