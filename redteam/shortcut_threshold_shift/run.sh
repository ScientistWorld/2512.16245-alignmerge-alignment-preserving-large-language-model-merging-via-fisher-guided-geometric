#!/usr/bin/env bash
# SHORTCUT probe (operating point): reference AlignMerge merges (seeds 13, 21, 34) with only the alignment
# decision threshold moved, chosen on the visible training split (shift_threshold.py).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../common/sandbox.sh"
for SEED in 13 21 34; do
  rt_begin "$HERE/seed$SEED"
  rt_reuse "reference/seed$SEED"
  PYTHONPATH="$W" python "$HERE/shift_threshold.py" checkpoints/method/alignmerge.pt | tee "$HERE/seed$SEED/shift.log"
  bash scripts/evaluate_train.sh > /dev/null
  rt_handoff_and_test
  rt_collect
done
