# Evaluation

Evaluation is method-agnostic and reads artifacts produced by a method run. It does not rerun the merge algorithm.

Primary metric:
- `alignment_accuracy`: safe/unsafe classification accuracy; higher is better.

Secondary metrics:
- `aqi`: representation-space Alignment Quality Index; higher is better.
- `task_accuracy`: task classification accuracy; higher is better.
- `unsafe_probability`: mean probability assigned to the unsafe class on alignment prompts; lower is better.
- balanced accuracy, false-negative, loss, calibration, and hard-slice metrics in `scoring/reference.json`.
- optional parameter-space diagnostics when the required source checkpoints are present.

The standard evaluator is `scripts/evaluate.sh`, which calls `eval/evaluate_outputs.py` and writes `scoring/scores.json`. It reloads saved checkpoints and recomputes task/alignment metrics on held-out splits. Train/test evaluators add additional visible and held-out stress metrics without changing the method output interface.
