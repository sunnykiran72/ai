#!/usr/bin/env bash

set -euo pipefail

APP_DIR="${APP_DIR:-/workspace/hybrid_vto_v1_latest_v1}"
BRANCH="${BRANCH:-wardobe_and_tryon}"

cd "${APP_DIR}"
git pull origin "${BRANCH}"
chmod +x scripts/server_ctl.sh
./scripts/server_ctl.sh restart
./scripts/server_ctl.sh status
