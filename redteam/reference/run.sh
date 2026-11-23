#!/usr/bin/env bash
# Reference AlignMerge method, 5 training seeds (data seed fixed at the gym's 13).
# seed13 runs the gym's own scripts/method.sh unchanged (must reproduce scoring/scores_test.json exactly);
# the other seeds run the same pipeline with only model-init / minibatch randomness changed.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../common/sandbox.sh"

rt_begin "$HERE/seed13"
bash scripts/method.sh
bash scripts/evaluate_train.sh > /dev/null
rt_handoff_and_test
rt_collect

for SEED in 21 34 55 89; do
  rt_begin "$HERE/seed$SEED"
  rt_install_rtlib
  cat > scripts/method.sh <<EOS
#!/usr/bin/env bash
set -euo pipefail
cd "\$(dirname "\$0")/.."
python -m method.alignmerge.rt_variant --variant reference --out-dir checkpoints/method --train-seed $SEED
EOS
  bash scripts/method.sh
  bash scripts/evaluate_train.sh > /dev/null
  rt_handoff_and_test
  rt_collect
done
