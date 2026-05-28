#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m baseline.simple.run_baselines --out-dir checkpoints/baseline --seed 13
python -m eval.evaluate_outputs --input checkpoints/baseline/metrics.json --output scoring/baseline_scores.json

