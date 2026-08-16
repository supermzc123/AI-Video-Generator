param(
    [ValidateRange(5, 120)]
    [int]$TimeoutSeconds = 30,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
$desktopRoot = Join-Path $projectRoot "desktop"
$frontendUrl = "http://127.0.0.1:1420/"
$apiHealthUrl = "http://127.0.0.1:8000/api/v1/health"
$restartApiScript = Join-Path $PSScriptRoot "restart-api.ps1"
$viteEntry = Join-Path $desktopRoot "node_modules\vite\bin\vite.js"
$logRoot = Join-Path $projectRoot "data\logs"

function Test-ApiReady {
    try {
        $health = Invoke-RestMethod -Uri $apiHealthUrl -TimeoutSec 2
        return $health.status -eq "ok"
    } catch {
        return $false
    }
}

function Test-FrontendReady {
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri $frontendUrl -TimeoutSec 2
        return $response.StatusCode -eq 200 -and $response.Content.Contains("AI Video Generator")
    } catch {
        return $false
    }
}

function Test-ProjectFrontendProcess {
    param([Parameter(Mandatory = $true)][int]$ProcessId)

    $processInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId"
    if ($null -eq $processInfo) {
        return $false
    }

    $commandLine = [string]$processInfo.CommandLine
    return (
        $commandLine.IndexOf($viteEntry, [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
        (
            $commandLine.IndexOf($desktopRoot, [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -and
            $commandLine.IndexOf("vite", [System.StringComparison]::OrdinalIgnoreCase) -ge 0
        )
    )
}

Write-Output "Starting AI Video Generator..."
Write-Output ""

if (-not (Test-ApiReady)) {
    Write-Output "The control plane is offline. Starting it now..."
    & $restartApiScript -TimeoutSeconds $TimeoutSeconds
    if ($LASTEXITCODE -ne 0) {
        throw "The control plane failed to start."
    }
} else {
    Write-Output "Control plane is online: $apiHealthUrl"
}

if (-not (Test-FrontendReady)) {
    $listeners = @(
        Get-NetTCPConnection -LocalAddress "127.0.0.1" -LocalPort 1420 -State Listen `
            -ErrorAction SilentlyContinue
    )
    $ownerIds = @($listeners | Select-Object -ExpandProperty OwningProcess -Unique)
    foreach ($ownerId in $ownerIds) {
        if (-not (Test-ProjectFrontendProcess -ProcessId $ownerId)) {
            $owner = Get-CimInstance Win32_Process -Filter "ProcessId = $ownerId"
            throw "Port 1420 is owned by an unrelated process (PID $ownerId): $($owner.CommandLine)"
        }
    }

    if (-not (Test-Path -LiteralPath $viteEntry -PathType Leaf)) {
        throw "Frontend dependencies are missing. Run 'npm install' in $desktopRoot first."
    }

    $nodeCommand = Get-Command node.exe -ErrorAction SilentlyContinue
    if ($null -eq $nodeCommand) {
        throw "Node.js is not available on PATH. Install Node.js 20 or newer first."
    }

    New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $stdoutLog = Join-Path $logRoot "frontend-$timestamp.stdout.log"
    $stderrLog = Join-Path $logRoot "frontend-$timestamp.stderr.log"
    $frontendProcess = Start-Process `
        -FilePath $nodeCommand.Source `
        -ArgumentList @($viteEntry, "--host", "127.0.0.1", "--port", "1420") `
        -WorkingDirectory $desktopRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutLog `
        -RedirectStandardError $stderrLog `
        -PassThru

    Write-Output "Starting the web interface (PID $($frontendProcess.Id))..."
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    do {
        if ($frontendProcess.HasExited) {
            throw "The web interface exited during startup. See $stderrLog"
        }
        if (Test-FrontendReady) {
            Write-Output "Web interface is online: $frontendUrl"
            Write-Output "Frontend logs: $stdoutLog and $stderrLog"
            break
        }
        Start-Sleep -Milliseconds 500
    } while ([DateTime]::UtcNow -lt $deadline)

    if (-not (Test-FrontendReady)) {
        throw "The web interface health check timed out after $TimeoutSeconds seconds. See $stderrLog"
    }
} else {
    Write-Output "Web interface is online: $frontendUrl"
}

if (-not $NoBrowser) {
    Start-Process $frontendUrl
}

Write-Output ""
Write-Output "AI Video Generator is ready."
exit 0
