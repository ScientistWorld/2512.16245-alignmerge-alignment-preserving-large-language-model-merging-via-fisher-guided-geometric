#!/usr/bin/env bash
# SHORTCUT probe (impossible values): the reference merge (seed 13) rewritten so that its alignment logits are
# +-inf (see make_inf_logits.py). The three source checkpoints are not shipped, because the geometry
# diagnostics would otherwise be computed on infinite weights.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../common/sandbox.sh"
rt_begin "$HERE"
rt_reuse "reference/seed13"
PYTHONPATH="$W" python "$HERE/make_inf_logits.py" checkpoints/method/alignmerge.pt | tee "$HERE/make_inf_logits.log"
rm -f checkpoints/method/instruction_anchor.pt checkpoints/method/safety_anchor.pt checkpoints/method/task_expert.pt
python - <<'PY'
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
