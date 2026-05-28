#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m baseline.simple.run_baselines --out-dir checkpoints/baseline --seed 13
scripts/evaluate.sh checkpoints/baseline/metrics.json scoring/baseline_scores.json
