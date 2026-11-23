#!/usr/bin/env bash
# Red-team audit job (temporary; restored to the gym's original at the end of the audit).
set -uo pipefail
cd "$(dirname "$0")/.."
bash redteam/run_all.sh
