param(
    [int]$PollSeconds = 60,
    [int]$MaxHours = 24
)

$ErrorActionPreference = "Continue"
$api = "http://127.0.0.1:8000/api/v1"
$projects = @(
    "4eb809ef-4277-45a6-97de-c6f6e63c7cd9",
    "5b752f5c-252a-4bce-8a60-9b555d7cbfe1",
    "ba461488-f1d5-46c5-a401-55d1ba060e47",
    "833d1e87-88f1-4ee2-a956-86dca8f372b5"
)
$log = Join-Path (Resolve-Path (Join-Path $PSScriptRoot "..\data\logs")) "overnight-monitor.log"
$deadline = (Get-Date).AddHours($MaxHours)

function Write-MonitorLog([string]$message) {
    "$(Get-Date -Format o) $message" | Add-Content -LiteralPath $log -Encoding UTF8
}

function Invoke-Api([string]$method, [string]$url) {
    try { return Invoke-RestMethod -Method $method -Uri $url -TimeoutSec 20 } catch { Write-MonitorLog "API error $method $url : $($_.Exception.Message)"; return $null }
}

while ((Get-Date) -lt $deadline) {
    $unfinished = $false
    foreach ($projectId in $projects) {
        $status = Invoke-Api "GET" "$api/projects/$projectId/execution-status"
        if ($null -eq $status) { $unfinished = $true; continue }
        $groups = @($status.tasks | Group-Object state)
        $summary = (($groups | ForEach-Object { "$($_.Name)=$($_.Count)" }) -join ",")
        Write-MonitorLog "$projectId $summary"

        foreach ($task in @($status.tasks | Where-Object { $_.state -eq "failed" -and $_.attempt -lt $_.max_attempts })) {
            $encoded = [Uri]::EscapeDataString([string]$task.task_id)
            $retried = Invoke-Api "POST" "$api/tasks/$encoded/run"
            if ($null -ne $retried) { Write-MonitorLog "retried $($task.task_id)" }
        }

        foreach ($task in @($status.tasks | Where-Object { $_.state -in @("ready", "queued", "running", "blocked") })) {
            $unfinished = $true
        }
        if ($null -ne $status.generation_batch -and $status.generation_batch.dispatch_requested -ne $true) {
            $started = Invoke-Api "POST" "$api/projects/$projectId/generation/start"
            if ($null -ne $started) { Write-MonitorLog "requested dispatch $projectId" }
        }
    }

    if (-not $unfinished) {
        Write-MonitorLog "all monitored projects reached terminal task states; shutting down"
        Stop-Process -Name "node" -Force -ErrorAction SilentlyContinue
        Stop-Process -Name "python" -Force -ErrorAction SilentlyContinue
        shutdown.exe /s /t 30 /c "AI Video Generator overnight projects completed"
        exit 0
    }
    Start-Sleep -Seconds $PollSeconds
}

Write-MonitorLog "monitor deadline reached; leaving machine running for manual inspection"
