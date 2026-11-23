#!/usr/bin/env bash
# CHECK: Bayes-optimal ceiling of every accuracy-type metric on the hidden test slice.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../common/sandbox.sh"
rt_make_pristine
cd "$RT_PRISTINE"
PYTHONPATH="$RT_PRISTINE" python "$HERE/bayes.py" "$HERE/bayes_ceiling.json"
