#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${APP_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PORT="${PORT:-8000}"

# Fast-iteration mode:
# - auto reload on code changes
# - skip heavy startup preloads/warmups
export STARTUP_BACKGROUND_PRELOAD=0
export STARTUP_PRELOAD_ANALYZE=0
export STARTUP_PRELOAD_VTO=0
export QWEN_IMAGE_EDIT_PRELOAD=0
export QWEN_IMAGE_EDIT_PRELOAD_WARMUP=0
export ANALYZE_PRELOAD_MINICPM=0
export ANALYZE_PRELOAD_FLUX_RUNNER=0

export APP_DIR
export PORT
export LOG_FILE="${LOG_FILE:-/tmp/uvicorn_${PORT}_dev_fast.log}"
export PID_FILE="${PID_FILE:-/tmp/uvicorn_${PORT}_dev_fast.pid}"

echo "[dev-fast] APP_DIR=${APP_DIR}"
echo "[dev-fast] PORT=${PORT}"
echo "[dev-fast] LOG_FILE=${LOG_FILE}"

"${SCRIPT_DIR}/server_ctl.sh" dev-reload
