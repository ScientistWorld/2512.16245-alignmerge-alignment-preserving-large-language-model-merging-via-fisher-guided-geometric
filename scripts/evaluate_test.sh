#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
INPUT="${1:-checkpoints/method/metrics.json}"
OUTPUT="${2:-scoring/scores_test.json}"
python -m eval.test.evaluate --input "$INPUT" --output "$OUTPUT" --train-scores scoring/scores_train.json
