#!/usr/bin/env bash
set -euo pipefail
# Estimated download size: 0 bytes. Every artifact this gym uses is committed:
#   data/merge_inputs/*.pt  frozen source checkpoints (built by eval/test/build_world.py)
#   data/calibration.pt     the labelled calibration set
# No external downloads are required.
cd "$(dirname "$0")/.."
mkdir -p data/downloads
for f in data/merge_inputs/instruction_anchor.pt data/merge_inputs/safety_anchor.pt data/merge_inputs/task_expert.pt data/calibration.pt; do
  [[ -f "$f" ]] || { echo "missing committed artifact: $f" >&2; exit 1; }
done
printf 'No external downloads required; committed artifacts present.\n'
