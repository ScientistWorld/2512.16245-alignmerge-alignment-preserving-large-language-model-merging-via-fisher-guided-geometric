# Design Notes

## Checked

- Read `eval/train/evaluate.py`, `eval/test/evaluate.py`, `scripts/evaluate_train.sh`, `scripts/evaluate_test.sh`, and `scoring/reference.json`.
- Confirmed the train/test scripts invoke their own slice packages and write `scoring/scores_train.json` / `scoring/scores_test.json`.
- Confirmed the evaluators discover method names from `metrics.json` and checkpoint files rather than hardcoding a method default.
- Confirmed scoring is attached to method artifacts: both slices load `*.pt` checkpoints and fail if a checkpoint for a discovered method is missing.
- Grepped for train/test cross-imports and paper-agreement metric names in active scoring entries.
- Checked `bash -n` for both train/test shell wrappers.
- Confirmed `scoring/scores_train.json` and `scoring/scores_test.json` do not exist.

## Changed

- Replaced the exact 50/50 complement split with disjoint 45% train and 45% test slices. The test slice excludes train indices but uses a separate held-out selection seed, leaving 10% unused so the visible train split does not imply the exact test examples.
- Added stress metrics on harder regions of the same scientific tasks:
  - `task_boundary_accuracy` for task examples near angular/radius decision boundaries.
  - `alignment_overlap_accuracy` for safe/unsafe examples near distribution overlap.
  - `unsafe_overlap_false_negative_rate` for the safety-critical false-negative mode inside the overlap region.
- Added probability-quality metrics (`task_nll`, `alignment_nll`, `task_ece`, `alignment_ece`) so a method cannot optimize only argmax accuracy while producing brittle confidence scores.
- Made `alignment_subspace_drift` and `fisher_geodesic_loss` slice-specific by estimating Fisher/sketch geometry on each slice rather than reusing identical train-data diagnostics for both train and test.
- Set the illustrative `llama3_placeholder_from_paper` experiment to a tiny schema-valid weight and zero coefficients. It remains as paper context but no longer materially competes with the synthetic train/test gym or introduces active metrics the split evaluators do not compute.
- Added required `description` fields to all metric entries and updated active coefficients to emphasize the synthetic alignment-merge objective.

## Runtime Concerns For Runner Iteration

- I did not run the Python evaluators or generate score numbers, per this iteration's restriction. The runner should execute both scripts and inspect the actual train/test gap for saturation.
- The original paper markdown path supplied in the prompt was not present at `/home/user/scratch/AgentWorkspaces/papers/arxiv_mds/2512.16245/2512.16245.md`, so metric choices were based on the existing workspace design and synthetic benchmark files.
- The stress subsets are 35% of each 45% slice. If runner results show high variance or saturation, tune `STRESS_FRACTION` or the stress coefficients before freezing the benchmark.
- The placeholder experiment still exists in `reference.json` for paper context with zero metric coefficients and a `0.001` weight only to satisfy the current reference schema. If the scoring harness requires every reference experiment to appear in score files regardless of effective coefficient, remove that context experiment in a later schema cleanup or teach the scorer to ignore inactive metrics.
