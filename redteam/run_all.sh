#!/usr/bin/env bash
# Runs every red-team variant / shortcut / check in dependency order inside the gym container, then
# writes redteam/summary.{md,json}. Each run.sh is independent; a failing run is logged and skipped.
set -uo pipefail
RT="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$RT/logs"
python -c "import sys, torch; print('python', sys.version.split()[0], 'torch', torch.__version__)" | tee "$RT/logs/environment.txt"
RUNS=(
  check_bayes_ceiling
  reference
  weak_no_align weak_less_training weak_naive_average
  better_more_training better_more_data better_merge_plus better_more_data_grid_merge bruteforce_grid_merge
  shortcut_bruteforce_joint_training shortcut_test_pool_finetune
  shortcut_copy_safety_anchor shortcut_copy_safety_anchor_anchor_swap shortcut_anchor_swap
  shortcut_test_pool_finetune_anchor_swap shortcut_joint_training_anchor_delete
  shortcut_easier_data shortcut_forged_scores_crash
  check_method_list check_failed_run
  shortcut_constant_unsafe shortcut_inf_logits shortcut_threshold_shift
)
for r in "${RUNS[@]}"; do
  start=$(date +%s)
  if bash "$RT/$r/run.sh" > "$RT/logs/$r.log" 2>&1; then status=ok; else status=FAILED; fi
  echo "[run_all] $r: $status ($(( $(date +%s) - start ))s)" | tee -a "$RT/logs/run_all.txt"
done
python "$RT/common/rank.py" > "$RT/logs/rank.txt" 2>&1 || echo "[run_all] rank.py FAILED" | tee -a "$RT/logs/run_all.txt"
rm -rf "$RT/_work"
echo "[run_all] done" | tee -a "$RT/logs/run_all.txt"
