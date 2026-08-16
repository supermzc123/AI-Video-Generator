param(
    [Parameter(Mandatory = $true)][string[]]$Files,
    [string]$CertificateThumbprint = $env:AIVIDEO_SIGNING_CERT_THUMBPRINT,
    [string]$TimestampUrl = "http://timestamp.digicert.com",
    [switch]$Release
)
$ErrorActionPreference = "Stop"
if (-not $CertificateThumbprint) {
    if ($Release) { throw "A trusted Authenticode certificate is required for a public Release" }
    Write-Warning "No signing certificate configured; artifacts are unsigned internal-test builds."
    exit 0
}
foreach ($file in $Files) {
    $resolved = (Resolve-Path -LiteralPath $file).Path
    & signtool.exe sign /sha1 $CertificateThumbprint /fd SHA256 /tr $TimestampUrl /td SHA256 $resolved
    if ($LASTEXITCODE -ne 0) { throw "Signing failed: $resolved" }
    & signtool.exe verify /pa /all $resolved
    if ($LASTEXITCODE -ne 0) { throw "Signature verification failed: $resolved" }
}
