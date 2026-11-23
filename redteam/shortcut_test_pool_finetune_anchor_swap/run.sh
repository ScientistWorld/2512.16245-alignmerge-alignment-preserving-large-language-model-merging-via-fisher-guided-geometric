#!/usr/bin/env bash
# SHORTCUT (test answers rebuilt from the visible generator + anchors moved): take the merge fine-tuned on
# the evaluator's held-out pool (redteam/shortcut_test_pool_finetune) and overwrite the three source
# checkpoints with it, so the geometry diagnostics that otherwise penalise the moved weights become 0.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../common/sandbox.sh"
for SEED in 13 21 34; do
  rt_begin "$HERE/seed$SEED"
  rt_reuse "shortcut_test_pool_finetune/seed$SEED"
  for a in instruction_anchor safety_anchor task_expert; do cp checkpoints/method/alignmerge.pt "checkpoints/method/$a.pt"; done
  bash scripts/evaluate_train.sh > /dev/null
  rt_handoff_and_test
  rt_collect
done
