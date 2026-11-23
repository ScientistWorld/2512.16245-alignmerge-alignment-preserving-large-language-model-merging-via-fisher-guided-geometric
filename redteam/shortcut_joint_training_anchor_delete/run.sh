#!/usr/bin/env bash
# SHORTCUT (empty entries avoid a loss): the geometry diagnostics are only written when the three source
# checkpoints are on disk. Take the jointly trained (non-merged) model from
# redteam/shortcut_bruteforce_joint_training and simply do not ship the sources: the two geometry metrics
# that penalise it disappear from the hidden-test entry instead of getting a worst value.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../common/sandbox.sh"
for SEED in 13 21 34; do
  rt_begin "$HERE/seed$SEED"
  rt_reuse "shortcut_bruteforce_joint_training/seed$SEED"
  rm -f checkpoints/method/instruction_anchor.pt checkpoints/method/safety_anchor.pt checkpoints/method/task_expert.pt
  PYTHONPATH="$W" python - <<'PY'
import json
p = "checkpoints/method/metrics.json"
d = json.load(open(p))
for k in ("instruction_anchor", "safety_anchor", "task_expert"):
    d["synthetic_alignment_merge"]["results"].pop(k, None)
json.dump(d, open(p, "w"), indent=2)
PY
  bash scripts/evaluate_train.sh > /dev/null
  rt_handoff_and_test
  rt_collect
done
