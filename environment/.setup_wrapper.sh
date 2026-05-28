#!/bin/bash
set -euo pipefail
# Prevent host VIRTUAL_ENV from confusing uv inside the container.
unset VIRTUAL_ENV
export UV_SYSTEM_PYTHON=1 UV_BREAK_SYSTEM_PACKAGES=1
# Use uv from /opt/uv/bin (bind-mounted from host, avoids
# making /usr/local/bin read-only which blocks script installs).
export PATH=/opt/uv/bin:$PATH
export UV_LINK_MODE=copy
# Kill hung downloads after 5 min so uv retries instead of
# blocking forever on CLOSE-WAIT sockets.
export UV_HTTP_TIMEOUT=300
# Lock down the cache dir: save it so wrappers can restore it
# even if setup.sh overrides UV_CACHE_DIR.
export _SWARM_UV_CACHE="$UV_CACHE_DIR"
# Run a command with a 10-min timeout; retry up to 3 times.
_timed_run() {
  local attempt
  for attempt in 1 2 3; do
    if timeout 600 "$@"; then return 0; fi
    echo "[swarm] command timed out or failed (attempt $attempt/3), retrying..." >&2
  done
  echo "[swarm] command failed after 3 attempts: $*" >&2
  return 1
}
# Redirect all pip variants to uv pip.
_fast_pip() {
  local args=()
  for a in "$@"; do
    case "$a" in
      --no-cache-dir|--no-cache) continue ;;
    esac
    args+=("$a")
  done
  _timed_run env UV_CACHE_DIR="$_SWARM_UV_CACHE" /opt/uv/bin/uv pip "${args[@]}"
}
pip()  { _fast_pip "$@"; }
pip3() { _fast_pip "$@"; }
export -f _timed_run _fast_pip pip pip3
# Intercept direct uv calls to strip --no-cache and force shared cache.
uv() {
  local args=()
  for a in "$@"; do
    case "$a" in
      --no-cache-dir|--no-cache) continue ;;
    esac
    args+=("$a")
  done
  _timed_run env UV_CACHE_DIR="$_SWARM_UV_CACHE" /opt/uv/bin/uv "${args[@]}"
}
export -f uv
python3() {
  if [ "${1:-}" = "-m" ] && [ "${2:-}" = "pip" ]; then
    shift 2
    _fast_pip "$@"
  else
    command python3 "$@"
  fi
}
export -f python3
source /scratch/projects/researchgym/workspaces/workspaces/2512.16245/environment/setup.sh
