# Evaluation

Evaluation is method-agnostic and reads artifacts produced by a method run. It does not rerun the merge algorithm.

Primary metric:
- `aqi`: representation-space Alignment Quality Index; higher is better.

Secondary metrics:
- `alignment_accuracy`: safe/unsafe classification accuracy; higher is better.
- `task_accuracy`: task classification accuracy; higher is better.
- `unsafe_probability`: mean probability assigned to the unsafe class on alignment prompts; lower is better.
- `alignment_subspace_drift`: norm of the merged displacement in the estimated alignment subspace; lower is better.
- `fisher_geodesic_loss`: Fisher-weighted distance from the barycentric merge target; lower is better.

The standard evaluator is `scripts/evaluate.sh`, which calls `eval/evaluate_outputs.py` and writes `scoring/scores.json`. It reloads saved checkpoints, recomputes task/alignment metrics on held-out splits, and independently re-estimates the Fisher geometry needed for drift and geodesic diagnostics.
