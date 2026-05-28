#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m method.alignmerge.train_and_merge --out-dir checkpoints/method --seed 13

