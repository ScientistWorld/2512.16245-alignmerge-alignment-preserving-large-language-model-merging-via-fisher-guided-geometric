#!/usr/bin/env bash
# Better variant: sources trained 4x longer on 8x more generator data, merged by the brute-force coefficient grid selected on the training split.
# Method: python -m method.alignmerge.rt_variant --variant more_data_grid (see redteam/common/rtlib.py),
# run for training seeds 13, 21, 34 (data seed fixed at 13); each seed is put in place, run, and hidden-tested.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../common/sandbox.sh"
for SEED in ${RT_SEEDS:-13 21 34}; do
  rt_begin "$HERE/seed$SEED"
  rt_install_rtlib
  cat > scripts/method.sh <<EOS
#!/usr/bin/env bash
set -euo pipefail
cd "\$(dirname "\$0")/.."
python -m method.alignmerge.rt_variant --variant more_data_grid --out-dir checkpoints/method --train-seed $SEED
EOS
  bash scripts/method.sh
  bash scripts/evaluate_train.sh > /dev/null
  rt_handoff_and_test
  rt_collect
done
