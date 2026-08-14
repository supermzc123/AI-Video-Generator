$ErrorActionPreference = "Stop"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is required. Install it before running this script."
}

uv sync --extra dev --python 3.12
Write-Output "Project environment is ready. Copy .env.example to .env and set ComfyUI paths."
