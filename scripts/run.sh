#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

bash scripts/reproduce.sh
bash scripts/baseline.sh
bash scripts/evaluate_train.sh checkpoints
bash scripts/evaluate_test.sh checkpoints
