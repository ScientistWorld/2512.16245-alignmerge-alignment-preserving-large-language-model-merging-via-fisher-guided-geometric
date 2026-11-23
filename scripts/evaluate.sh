#!/usr/bin/env bash
# Standard evaluator: same as the train-slice evaluator, written to scoring/scores.json (or $2).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT_DIR="${1:-checkpoints}"
OUTPUT="${2:-scoring/scores.json}"
python -m eval.train.evaluate --checkpoint-root "$ROOT_DIR" --output "$OUTPUT"
