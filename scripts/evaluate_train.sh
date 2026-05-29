#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
SOURCE="${1:-checkpoints}"
OUTPUT="${2:-scoring/scores_train.json}"

if [[ -d "$SOURCE" ]]; then
  if [[ -f "$SOURCE/method/metrics.json" ]]; then
    INPUT="$SOURCE/method/metrics.json"
    CHECKPOINT_DIR="$SOURCE/method"
  elif [[ -f "$SOURCE/metrics.json" ]]; then
    INPUT="$SOURCE/metrics.json"
    CHECKPOINT_DIR="$SOURCE"
  else
    echo "Could not find metrics.json under checkpoint source: $SOURCE" >&2
    exit 1
  fi
else
  INPUT="$SOURCE"
  CHECKPOINT_DIR="$(dirname "$INPUT")"
fi

python -m eval.train.evaluate --input "$INPUT" --checkpoint-dir "$CHECKPOINT_DIR" --output "$OUTPUT"
