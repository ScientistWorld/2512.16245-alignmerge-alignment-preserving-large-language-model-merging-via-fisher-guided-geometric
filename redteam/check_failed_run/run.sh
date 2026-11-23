#!/usr/bin/env bash
# CHECK: what does the hidden test write for a failed run? The agent's method crashes before writing any
# checkpoint; the hidden test is then run as usual.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../common/sandbox.sh"
rt_begin "$HERE"
cat > scripts/method.sh <<'EOS'
#!/usr/bin/env bash
echo "method crashed" >&2
exit 1
EOS
bash scripts/method.sh || true
bash scripts/evaluate_train.sh > /dev/null 2>&1 || true
rt_handoff_and_test
rt_collect
ls scoring > "$HERE/scoring_dir_after_test.txt"
