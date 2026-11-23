#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Reference reproduction from a clean slate: AlignMerge + ablations, baselines, train slice, hidden test.
# (The hardening verification of iteration 11 is `bash redteam/v2/run_all.sh`; see HARDENING.md.)
rm -f checkpoints/method/*.pt checkpoints/baseline/*.pt
bash scripts/reproduce.sh
bash scripts/baseline.sh
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
