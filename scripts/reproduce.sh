#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
scripts/method.sh
scripts/evaluate.sh checkpoints/method/metrics.json scoring/scores.json
python validate.py --compare

