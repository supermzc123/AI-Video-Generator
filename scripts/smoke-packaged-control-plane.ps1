$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
$executable = Join-Path $projectRoot "desktop\src-tauri\resources\control-plane\aivideo-api.exe"
if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
    throw "Packaged control plane is missing: $executable"
}

$listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, 0)
$listener.Start()
$port = ([Net.IPEndPoint]$listener.LocalEndpoint).Port
$listener.Stop()
$nonce = [Guid]::NewGuid().ToString("N")
$smokeRoot = Join-Path ([IO.Path]::GetTempPath()) "aivideo-control-plane-smoke-$nonce"
$original = @{
    AIVIDEO_API_PORT = $env:AIVIDEO_API_PORT
    AIVIDEO_INSTANCE_NONCE = $env:AIVIDEO_INSTANCE_NONCE
    AIVIDEO_DATA_ROOT = $env:AIVIDEO_DATA_ROOT
    AIVIDEO_RUNTIME_SETTINGS_PATH = $env:AIVIDEO_RUNTIME_SETTINGS_PATH
    PATH = $env:PATH
}
$process = $null
try {
    $env:AIVIDEO_API_PORT = [string]$port
    $env:AIVIDEO_INSTANCE_NONCE = $nonce
    $env:AIVIDEO_DATA_ROOT = $smokeRoot
    $env:AIVIDEO_RUNTIME_SETTINGS_PATH = Join-Path $smokeRoot "settings.json"
    $env:PATH = "$env:SystemRoot\System32"
    $process = Start-Process -FilePath $executable -WindowStyle Hidden -PassThru
    $deadline = [DateTime]::UtcNow.AddSeconds(30)
    $health = $null
    while ([DateTime]::UtcNow -lt $deadline) {
        if ($process.HasExited) { throw "Packaged control plane exited with $($process.ExitCode)" }
        try {
            $health = Invoke-RestMethod "http://127.0.0.1:$port/api/v1/health" -TimeoutSec 2
            break
        } catch {
            Start-Sleep -Milliseconds 150
        }
    }
    if ($null -eq $health) { throw "Packaged control plane did not become healthy" }
    if ($health.service_id -ne "io.github.supermzc123.aivideogenerator.control-plane") {
        throw "Packaged control plane returned the wrong service identity"
    }
    if ($health.instance_nonce -ne $nonce) {
        throw "Packaged control plane returned the wrong instance nonce"
    }
    $capabilities = Invoke-RestMethod "http://127.0.0.1:$port/api/v1/postprocessing/capabilities"
    $whisper = $capabilities.profiles | Where-Object { $_.profile.profile_id -eq "transcription:faster-whisper" }
    if ($null -eq $whisper -or $whisper.available -ne $true) {
        throw "Packaged control plane is missing the Faster Whisper executor"
    }
    Write-Output "Packaged control plane smoke test passed on port $port."
} finally {
    if ($null -ne $process -and -not $process.HasExited) {
        Stop-Process -Id $process.Id -Force
        $process.WaitForExit()
    }
    $env:AIVIDEO_API_PORT = $original.AIVIDEO_API_PORT
    $env:AIVIDEO_INSTANCE_NONCE = $original.AIVIDEO_INSTANCE_NONCE
    $env:AIVIDEO_DATA_ROOT = $original.AIVIDEO_DATA_ROOT
    $env:AIVIDEO_RUNTIME_SETTINGS_PATH = $original.AIVIDEO_RUNTIME_SETTINGS_PATH
    $env:PATH = $original.PATH
    if (Test-Path -LiteralPath $smokeRoot) {
        Remove-Item -LiteralPath $smokeRoot -Recurse -Force
    }
}
