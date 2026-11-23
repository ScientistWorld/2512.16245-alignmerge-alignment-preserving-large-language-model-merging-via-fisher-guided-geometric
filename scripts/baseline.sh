#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m baseline.simple.run_baselines --out-dir checkpoints/baseline
scripts/evaluate.sh checkpoints scoring/baseline_scores.json
