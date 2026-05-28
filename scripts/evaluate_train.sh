#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
INPUT="${1:-checkpoints/method/metrics.json}"
OUTPUT="${2:-scoring/scores_train.json}"
python -m eval.train.evaluate --input "$INPUT" --output "$OUTPUT"
