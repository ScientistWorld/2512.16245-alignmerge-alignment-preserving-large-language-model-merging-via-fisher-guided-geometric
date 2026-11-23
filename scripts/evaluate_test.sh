#!/usr/bin/env bash
# Hidden test. Run with no arguments: scores every checkpoint under checkpoints/ (plus the fixed baselines and the
# expected submission) on the hidden test set and writes scoring/scores_test.json. Optional $1: checkpoint root.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT_DIR="${1:-checkpoints}"
rm -f scoring/scores_test.json
python -m eval.test.evaluate --checkpoint-root "$ROOT_DIR" --output scoring/scores_test.json
