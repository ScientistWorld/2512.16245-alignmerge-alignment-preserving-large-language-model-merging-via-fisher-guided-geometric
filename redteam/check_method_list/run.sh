#!/usr/bin/env bash
# CHECK: does a run of the hidden test write every method on disk? The test takes its method list from the
# agent-written scoring/scores_train.json when that file exists.
#  stale_list:      a new checkpoint (alignmerge_v2.pt, a copy of the reference merge) is added after the
#                   agent's last evaluate_train.sh run -> is it scored?
#  drop_submission: the agent's scores_train.json no longer lists `alignmerge` (e.g. edited or produced by an
#                   older run) -> the submission is silently absent while the test exits 0.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../common/sandbox.sh"

rt_begin "$HERE/stale_list"
rt_reuse "reference/seed13"
bash scripts/evaluate_train.sh > /dev/null
cp checkpoints/method/alignmerge.pt checkpoints/method/alignmerge_v2.pt
rt_handoff_and_test
rt_collect

rt_begin "$HERE/drop_submission"
rt_reuse "reference/seed13"
bash scripts/evaluate_train.sh > /dev/null
python - <<'PY'
import json
tr = json.load(open("scoring/scores_train.json"))
tr["experiments"]["synthetic_alignment_merge"]["results"].pop("alignmerge")
json.dump(tr, open("scoring/scores_train.json", "w"), indent=2)
PY
rt_handoff_and_test
rt_collect
