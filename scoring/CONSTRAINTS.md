# Constraints

- **Data.** A method may read only the three frozen sources in `data/merge_inputs/` and the labelled calibration
  set `data/calibration.pt` (1,024 task + 1,024 alignment prompts). No other data: no extra labels and no files
  from outside the gym. Every input the method feeds through a model (to compute gradients, Fisher statistics,
  features, losses or teacher outputs) must be one of the 2,048 calibration prompts. Inputs that are sampled,
  generated, augmented, interpolated or perturbed to stand in for the hidden prompt distributions are not allowed.
- **Model family.** The submission is a plain `state_dict` of `TinyMultiTaskNet` (`data/benchmark.py`) with the
  same keys and shapes: 16 → 48 → 48 → 5, tanh, one shared output layer. No extra capacity, no ensembles, no
  wrappers. Weights and outputs must be finite.
- **Submission.** `checkpoints/method/alignmerge.pt`. Other `*.pt` files under `checkpoints/` are scored under their
  file stems. Files named `instruction_anchor`, `safety_anchor`, `task_expert` or `naive_average` are ignored: the
  test rebuilds those four reference points from its own copies of the sources.
- **What the test trusts.** Nothing a method writes besides its checkpoints. `metrics.json`, `scoring/*.json`,
  edits to `data/` or `eval/`, and replaced source files have no effect on the hidden test, which uses its own
  evaluator, sources and test set.
- **Failure.** A missing, unreadable or malformed checkpoint, non-finite weights or non-finite outputs score as a
  failed run: the worst value of every scored metric.
- **Determinism.** Fix seeds so a rerun of the method reproduces the submitted checkpoint.
