#!/usr/bin/env bash
set -euo pipefail

prefix="${AIVIDEO_WORKER_PREFIX:-/opt/aivideo-worker}"
if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this installer as root." >&2
  exit 1
fi
id -u aivideo >/dev/null 2>&1 || useradd --system --home /var/lib/aivideo-worker --shell /usr/sbin/nologin aivideo
install -d -o aivideo -g aivideo /var/lib/aivideo-worker /etc/aivideo-worker "$prefix"
python3 -m venv "$prefix"
"$prefix/bin/pip" install --upgrade pip
"$prefix/bin/pip" install 'ai-video-generator[worker]'
install -m 0644 "$(dirname "$0")/aivideo-worker.service" /etc/systemd/system/aivideo-worker.service
systemctl daemon-reload
echo "Edit /etc/systemd/system/aivideo-worker.service and install /etc/aivideo-worker/worker-token before enabling the experimental Worker."
