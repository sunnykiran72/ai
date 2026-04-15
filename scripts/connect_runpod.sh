#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  connect_runpod.sh --target <ip:port|ssh_command> [options]

Examples:
  connect_runpod.sh --target "157.157.221.177:14621"
  connect_runpod.sh --target "ssh root@157.157.221.177 -p 14621 -i ~/.ssh/id_ed25519"
  connect_runpod.sh --target "157.157.221.177:14621" --name runpod-tryon --dry-run

Options:
  --target <value>       Required. Either:
                         1) ip:port
                         2) full SSH command copied from RunPod UI
  --name <host_alias>    SSH config host alias. Default: runpod-current
  --key <path>           Private key path. Default: ~/.ssh/id_ed25519
  --user <username>      SSH user. Default: root
  --connect              Connect immediately after writing config (default)
  --no-connect           Only update config and print next commands
  --dry-run              Parse and print without writing config
  -h, --help             Show help
EOF
}

TARGET=""
HOST_ALIAS="runpod-current"
KEY_PATH="${HOME}/.ssh/id_ed25519"
SSH_USER="root"
DO_CONNECT="1"
DRY_RUN="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      TARGET="${2:-}"
      shift 2
      ;;
    --name)
      HOST_ALIAS="${2:-}"
      shift 2
      ;;
    --key)
      KEY_PATH="${2:-}"
      shift 2
      ;;
    --user)
      SSH_USER="${2:-}"
      shift 2
      ;;
    --connect)
      DO_CONNECT="1"
      shift
      ;;
    --no-connect)
      DO_CONNECT="0"
      shift
      ;;
    --dry-run)
      DRY_RUN="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${TARGET}" ]]; then
  echo "Error: --target is required." >&2
  usage
  exit 1
fi

expand_path() {
  local p="$1"
  if [[ "$p" == ~* ]]; then
    printf "%s%s" "${HOME}" "${p:1}"
  else
    printf "%s" "$p"
  fi
}

trim_quotes() {
  local s="$1"
  s="${s%\"}"
  s="${s#\"}"
  s="${s%\'}"
  s="${s#\'}"
  printf "%s" "$s"
}

parse_target() {
  local t="$1"
  local parsed_host=""
  local parsed_port=""
  local parsed_user="${SSH_USER}"
  local parsed_key="${KEY_PATH}"

  # Case 1: ip:port
  if [[ "$t" =~ ^([0-9]{1,3}(\.[0-9]{1,3}){3}):([0-9]{1,5})$ ]]; then
    parsed_host="${BASH_REMATCH[1]}"
    parsed_port="${BASH_REMATCH[3]}"
    echo "${parsed_host}|${parsed_port}|${parsed_user}|${parsed_key}"
    return 0
  fi

  # Case 2: full SSH command
  if [[ "$t" =~ ssh[[:space:]]+([^@[:space:]]+)@([^[:space:]]+) ]]; then
    parsed_user="${BASH_REMATCH[1]}"
    parsed_host="${BASH_REMATCH[2]}"
  fi

  if [[ "$t" =~ -p[[:space:]]+([0-9]{1,5}) ]]; then
    parsed_port="${BASH_REMATCH[1]}"
  fi

  if [[ "$t" =~ -i[[:space:]]+([^[:space:]]+) ]]; then
    parsed_key="$(trim_quotes "${BASH_REMATCH[1]}")"
  fi

  if [[ -n "$parsed_host" && -n "$parsed_port" ]]; then
    echo "${parsed_host}|${parsed_port}|${parsed_user}|${parsed_key}"
    return 0
  fi

  echo "Unable to parse --target. Pass either ip:port or full SSH command." >&2
  exit 1
}

IFS='|' read -r HOST PORT USERNAME KEY_FILE <<<"$(parse_target "${TARGET}")"
KEY_FILE="$(expand_path "${KEY_FILE}")"

if [[ ! -f "${KEY_FILE}" ]]; then
  echo "Key file not found: ${KEY_FILE}" >&2
  exit 1
fi

mkdir -p "${HOME}/.ssh"
chmod 700 "${HOME}/.ssh"
chmod 600 "${KEY_FILE}" || true

SSH_CONFIG="${HOME}/.ssh/config"
TMP_CONFIG="$(mktemp)"

if [[ -f "${SSH_CONFIG}" ]]; then
  cp "${SSH_CONFIG}" "${TMP_CONFIG}"
else
  : > "${TMP_CONFIG}"
fi

# Remove existing host block for the alias (simple non-indented Host blocks).
awk -v alias="${HOST_ALIAS}" '
  BEGIN { skip=0 }
  /^Host[[:space:]]+/ {
    split($0, a, /[[:space:]]+/)
    if (a[2] == alias) { skip=1; next }
    skip=0
  }
  skip == 0 { print }
' "${TMP_CONFIG}" > "${TMP_CONFIG}.clean"
mv "${TMP_CONFIG}.clean" "${TMP_CONFIG}"

cat >> "${TMP_CONFIG}" <<EOF

Host ${HOST_ALIAS}
  HostName ${HOST}
  User ${USERNAME}
  Port ${PORT}
  IdentityFile ${KEY_FILE}
  IdentitiesOnly yes
  ServerAliveInterval 30
  ServerAliveCountMax 3
  TCPKeepAlive yes
  ConnectTimeout 10
  StrictHostKeyChecking accept-new
EOF

echo "Parsed target:"
echo "  Host: ${HOST}"
echo "  Port: ${PORT}"
echo "  User: ${USERNAME}"
echo "  Key:  ${KEY_FILE}"
echo "  Alias:${HOST_ALIAS}"
echo
echo "Quick tests:"
echo "  nc -vz ${HOST} ${PORT}"
echo "  ssh ${HOST_ALIAS}"
echo "  scp -P ${PORT} -i ${KEY_FILE} <local> ${USERNAME}@${HOST}:/workspace/"
echo

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Dry run enabled. No files changed."
  rm -f "${TMP_CONFIG}"
  exit 0
fi

cp "${TMP_CONFIG}" "${SSH_CONFIG}"
rm -f "${TMP_CONFIG}"

echo "Updated ${SSH_CONFIG} with Host ${HOST_ALIAS}."

if command -v nc >/dev/null 2>&1; then
  if nc -w 5 -vz "${HOST}" "${PORT}" >/dev/null 2>&1; then
    echo "TCP check: reachable"
  else
    echo "TCP check: timeout/refused (you can still try ssh, but network may be blocked)"
  fi
fi

if [[ "${DO_CONNECT}" == "1" ]]; then
  exec ssh "${HOST_ALIAS}"
else
  echo "Skipping SSH connect (--no-connect set)."
fi
