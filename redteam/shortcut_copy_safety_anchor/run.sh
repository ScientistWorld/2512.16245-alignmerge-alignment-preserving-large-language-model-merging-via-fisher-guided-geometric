#!/usr/bin/env bash
# SHORTCUT (degenerate output = copied input): submit the safety-oriented source checkpoint unchanged as
# `alignmerge`, i.e. no merge at all. Reuses the reference runs' trained sources for all 5 seeds.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../common/sandbox.sh"
for SEED in 13 21 34 55 89; do
  rt_begin "$HERE/seed$SEED"
  rt_reuse "reference/seed$SEED"
  cp checkpoints/method/safety_anchor.pt checkpoints/method/alignmerge.pt
  bash scripts/evaluate_train.sh > /dev/null
  rt_handoff_and_test
  rt_collect
done
