#!/usr/bin/env bash
# Hardening verification (iteration 11): every variant / shortcut through the hidden test, from a clean slate,
# then the rank proxy + paired bootstrap summary (redteam/v2/summary.{md,json}).
set -uo pipefail
cd "$(dirname "$0")/../.."
rm -rf redteam/v2/runs redteam/v2/_work
python redteam/v2/harness.py --force
python redteam/v2/analyze.py
