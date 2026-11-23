#!/usr/bin/env bash
# Visible train slice (the calibration set). Scores every checkpoint under checkpoints/ plus the fixed baselines and
# writes scoring/scores_train.json. Optional $1: checkpoint root.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT_DIR="${1:-checkpoints}"
python -m eval.train.evaluate --checkpoint-root "$ROOT_DIR" --output scoring/scores_train.json
