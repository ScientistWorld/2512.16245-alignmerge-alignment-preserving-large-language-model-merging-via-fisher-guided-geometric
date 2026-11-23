#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Reference reproduction from a clean slate (nothing else runs beside it): method + ablations, baselines,
# train slice, hidden test.
rm -f checkpoints/method/*.pt checkpoints/baseline/*.pt
bash scripts/download.sh
bash scripts/method.sh
bash scripts/evaluate.sh checkpoints scoring/scores.json
bash scripts/baseline.sh
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
python validate.py --compare || echo "[run.sh] validate.py reported errors"

# Hardening verification for this iteration (variants and shortcuts through the hidden test).
bash redteam/v2/run_all.sh
