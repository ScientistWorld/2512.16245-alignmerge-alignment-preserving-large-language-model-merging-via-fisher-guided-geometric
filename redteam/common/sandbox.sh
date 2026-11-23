#!/usr/bin/env bash
# Shared helpers for redteam/<name>/run.sh.
#
# Each run gets a scratch copy of the gym laid out exactly as a future agent receives it
# (data/, baseline/, method/, eval/train/, scripts/{baseline,download,evaluate_train}.sh and the
# four scoring/*.md documents, taken from the commit that started this iteration). The run then
# installs its method, runs it plus evaluate_train.sh as the agent would, and finally simulates
# the hand-off: the gym's own eval/ and scripts/evaluate_test.sh are put back, every other file
# stays as the agent left it, and `bash scripts/evaluate_test.sh` runs with no arguments.
# The resulting scoring/scores_test.json (and the exit code) is copied into redteam/<name>/.

RT_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RT_GYM_COMMIT="${RT_GYM_COMMIT:-94feace}"   # "Start of Iteration 10" = the gym under audit
RT_WORK="$RT_REPO/redteam/_work"
RT_PRISTINE="$RT_WORK/pristine"

rt_make_pristine() {
  if [[ ! -f "$RT_PRISTINE/.ok" ]]; then
    rm -rf "$RT_PRISTINE"
    mkdir -p "$RT_PRISTINE"
    if ! (cd "$RT_REPO" && git -c safe.directory='*' archive "$RT_GYM_COMMIT" data eval scripts method baseline scoring) | tar -x -C "$RT_PRISTINE"; then
      echo "[rt] git archive failed; copying working tree instead" >&2
      for d in data eval scripts method baseline scoring; do cp -r "$RT_REPO/$d" "$RT_PRISTINE/"; done
      find "$RT_PRISTINE" -name __pycache__ -type d -prune -exec rm -rf {} +
      (cd "$RT_REPO" && git -c safe.directory='*' show "$RT_GYM_COMMIT:scripts/run.sh") > "$RT_PRISTINE/scripts/run.sh" 2>/dev/null || true
    fi
    touch "$RT_PRISTINE/.ok"
  fi
}

# rt_begin <run_dir>: create the agent's workspace W and cd into it.
rt_begin() {
  RT_RUN_DIR="$1"
  mkdir -p "$RT_RUN_DIR"
  RT_RUN_DIR="$(cd "$RT_RUN_DIR" && pwd)"
  RT_NAME="${RT_RUN_DIR#$RT_REPO/redteam/}"
  RT_NAME="${RT_NAME//\//__}"
  rt_make_pristine
  W="$RT_WORK/runs/$RT_NAME"
  rm -rf "$W"
  mkdir -p "$W/scripts" "$W/scoring" "$W/eval"
  cp -r "$RT_PRISTINE/data" "$RT_PRISTINE/baseline" "$RT_PRISTINE/method" "$W/"
  cp -r "$RT_PRISTINE/eval/train" "$W/eval/train"
  for s in baseline download evaluate_train; do cp "$RT_PRISTINE/scripts/$s.sh" "$W/scripts/"; done
  for d in TARGETS CONSTRAINTS DIRECTION EXPERIMENTS; do cp "$RT_PRISTINE/scoring/$d.md" "$W/scoring/"; done
  # the gym's own method runner, as an improve-mode agent would start from it
  cp "$RT_PRISTINE/scripts/method.sh" "$W/scripts/method.sh"
  rm -f "$RT_RUN_DIR/scores_test.json" "$RT_RUN_DIR/scores_train.json" "$RT_RUN_DIR/test_exit_code.txt"
  cd "$W"
  echo "[rt] $RT_NAME: workspace $W"
}

# rt_install_rtlib: put the shared red-team variant module into the agent's method package.
rt_install_rtlib() {
  cp "$RT_REPO/redteam/common/rtlib.py" "$W/method/alignmerge/rt_variant.py"
}

# rt_reuse <run_subdir>: reuse another run's trained checkpoints, e.g. `rt_reuse reference/seed13`
# (only for shortcuts that change what is written or scored, not how the model is trained).
rt_reuse() {
  mkdir -p "$W/checkpoints/method"
  cp "$RT_REPO/redteam/$1/checkpoints/"* "$W/checkpoints/method/"
}

# rt_handoff_and_test: gym's eval/ and evaluate_test.sh put back, hidden test run with no arguments.
rt_handoff_and_test() {
  cd "$W"
  rm -rf "$W/eval"
  cp -r "$RT_PRISTINE/eval" "$W/eval"
  cp "$RT_PRISTINE/scripts/evaluate_test.sh" "$W/scripts/evaluate_test.sh"
  RT_TEST_RC=0
  bash scripts/evaluate_test.sh > "$W/hidden_test.log" 2>&1 || RT_TEST_RC=$?
  echo "[rt] $RT_NAME: hidden test exit code $RT_TEST_RC"
  tail -n 3 "$W/hidden_test.log" | sed 's/^/[rt]   /'
}

# rt_collect: copy scores, exit code and checkpoints back into redteam/<name>/.
rt_collect() {
  echo "$RT_TEST_RC" > "$RT_RUN_DIR/test_exit_code.txt"
  [[ -f "$W/scoring/scores_test.json" ]] && cp "$W/scoring/scores_test.json" "$RT_RUN_DIR/scores_test.json"
  [[ -f "$W/scoring/scores_train.json" ]] && cp "$W/scoring/scores_train.json" "$RT_RUN_DIR/scores_train.json"
  tail -n 40 "$W/hidden_test.log" > "$RT_RUN_DIR/hidden_test_tail.log" 2>/dev/null || true
  if [[ -d "$W/checkpoints/method" ]]; then
    rm -rf "$RT_RUN_DIR/checkpoints"
    mkdir -p "$RT_RUN_DIR/checkpoints"
    cp "$W/checkpoints/method/"*.pt "$W/checkpoints/method/metrics.json" "$RT_RUN_DIR/checkpoints/" 2>/dev/null || true
  fi
  echo "[rt] $RT_NAME: collected into $RT_RUN_DIR"
}
