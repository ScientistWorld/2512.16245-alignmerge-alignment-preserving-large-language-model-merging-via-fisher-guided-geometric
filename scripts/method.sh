#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m method.alignmerge.merge --out-dir checkpoints/method
