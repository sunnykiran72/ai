#!/usr/bin/env bash

set -euo pipefail

ACTION="${1:-restart}"

APP_DIR="${APP_DIR:-/workspace/hybrid_vto_v1_latest_v1}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
LOG_FILE="${LOG_FILE:-/tmp/uvicorn_${PORT}.log}"
PID_FILE="${PID_FILE:-/tmp/uvicorn_${PORT}.pid}"
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:${PORT}/health}"
STARTUP_TIMEOUT_SEC="${STARTUP_TIMEOUT_SEC:-900}"

VENV_UVICORN="${APP_DIR}/.venv/bin/uvicorn"

_now() {
  date "+%Y-%m-%d %H:%M:%S"
}

_info() {
  echo "[$(_now)] $*"
}

_warn() {
  echo "[$(_now)] WARNING: $*" >&2
}

_err() {
  echo "[$(_now)] ERROR: $*" >&2
}

_pids_for_port() {
  pgrep -f "uvicorn main:app.*--port ${PORT}" || true
}

_is_running() {
  [[ -n "$(_pids_for_port)" ]]
}

_check_prereqs() {
  if [[ ! -d "${APP_DIR}" ]]; then
    _err "APP_DIR not found: ${APP_DIR}"
    exit 1
  fi
  if [[ ! -x "${VENV_UVICORN}" ]]; then
    _err "uvicorn not found/executable: ${VENV_UVICORN}"
    exit 1
  fi
}

status_server() {
  local pids
  pids="$(_pids_for_port)"
  if [[ -z "${pids}" ]]; then
    _info "status: DOWN (no uvicorn main:app on port ${PORT})"
  else
    _info "status: RUNNING pid(s): ${pids}"
  fi

  if command -v curl >/dev/null 2>&1; then
    local body
    body="$(curl -fsS -m 4 "${HEALTH_URL}" 2>/dev/null || true)"
    if [[ -n "${body}" ]]; then
      _info "health: ${body}"
    else
      _warn "health check failed at ${HEALTH_URL}"
    fi
  fi
}

stop_server() {
  local pids
  pids="$(_pids_for_port)"
  if [[ -z "${pids}" ]]; then
    _info "no running uvicorn process found for port ${PORT}"
    rm -f "${PID_FILE}" || true
    return 0
  fi

  _info "stopping pid(s): ${pids}"
  kill ${pids} || true

  local i
  for i in {1..20}; do
    sleep 1
    if ! _is_running; then
      _info "stopped"
      rm -f "${PID_FILE}" || true
      return 0
    fi
  done

  _warn "forcing kill for remaining pid(s): $(_pids_for_port)"
  kill -9 $(_pids_for_port) || true
  rm -f "${PID_FILE}" || true
}

wait_ready() {
  local start_ts now elapsed
  start_ts="$(date +%s)"

  while true; do
    if command -v curl >/dev/null 2>&1; then
      if curl -fsS -m 4 "${HEALTH_URL}" >/dev/null 2>&1; then
        _info "health is ready: ${HEALTH_URL}"
        return 0
      fi
    fi

    if ! _is_running; then
      _err "process exited before health became ready"
      _warn "last logs:"
      tail -n 80 "${LOG_FILE}" || true
      return 1
    fi

    now="$(date +%s)"
    elapsed="$((now - start_ts))"
    if (( elapsed >= STARTUP_TIMEOUT_SEC )); then
      _err "startup timeout after ${STARTUP_TIMEOUT_SEC}s"
      _warn "last logs:"
      tail -n 120 "${LOG_FILE}" || true
      return 1
    fi

    sleep 4
  done
}

start_server() {
  _check_prereqs

  if _is_running; then
    _info "already running pid(s): $(_pids_for_port)"
    wait_ready
    return 0
  fi

  _info "starting uvicorn on ${HOST}:${PORT}"
  cd "${APP_DIR}"

  nohup "${VENV_UVICORN}" main:app \
    --app-dir "${APP_DIR}" \
    --host "${HOST}" \
    --port "${PORT}" \
    > "${LOG_FILE}" 2>&1 &

  local pid="$!"
  echo "${pid}" > "${PID_FILE}"
  _info "started pid=${pid}, log=${LOG_FILE}"

  wait_ready
}

start_server_reload() {
  _check_prereqs
  stop_server

  _info "starting uvicorn in reload mode on ${HOST}:${PORT}"
  cd "${APP_DIR}"

  nohup "${VENV_UVICORN}" main:app \
    --app-dir "${APP_DIR}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --reload \
    > "${LOG_FILE}" 2>&1 &

  local pid="$!"
  echo "${pid}" > "${PID_FILE}"
  _info "started reload pid=${pid}, log=${LOG_FILE}"

  wait_ready
}

logs_server() {
  _info "tailing logs: ${LOG_FILE}"
  tail -f "${LOG_FILE}"
}

case "${ACTION}" in
  start)
    start_server
    ;;
  stop)
    stop_server
    ;;
  restart)
    stop_server
    start_server
    ;;
  dev-reload)
    start_server_reload
    ;;
  status)
    status_server
    ;;
  logs)
    logs_server
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|dev-reload|status|logs}" >&2
    exit 2
    ;;
esac
