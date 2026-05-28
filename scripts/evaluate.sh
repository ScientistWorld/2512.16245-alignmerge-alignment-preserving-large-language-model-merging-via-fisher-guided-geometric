#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
INPUT="${1:-checkpoints/method/metrics.json}"
OUTPUT="${2:-scoring/scores.json}"
python -m eval.evaluate_outputs --input "$INPUT" --output "$OUTPUT"

