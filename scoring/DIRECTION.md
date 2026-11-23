# Direction

Stay in post-hoc checkpoint merging that balances task utility against alignment preservation. The submission is
built from the three frozen sources. The calibration set supplies the statistics and fitted parameters of the
merge, for example:
- curvature or Fisher estimates, alignment-sensitive subspaces, budgets or constraints on an alignment functional;
- per-layer, per-tensor or per-parameter coefficients, masks or sign resolution;
- a refinement of the merged weights on the calibration prompts, using their labels or the sources' outputs on
  those same prompts.

Out of scope:
- training a model from scratch, or otherwise discarding the sources;
- distillation or fine-tuning on inputs other than the 2,048 calibration prompts (sampled, generated, augmented,
  interpolated or perturbed inputs). This would turn the problem into retraining on a surrogate of the hidden
  input distribution, not merging;
- ensembles, routing between models at inference, or extra capacity;
- anything that reads `eval/`, rebuilds the hidden test set, or edits the evaluator.

Where the paper's method is weak here, and so where a better method can show up:
- AlignMerge's alignment penalty and AQI budget hold refusals close to the anchors, but it gives up utility. It
  sits well below the task expert and below naive averaging on task accuracy, so better use of the expert's update
  is the main open room;
- its curvature estimates come from 1,024 prompts per side;
- its quadratic model of the loss is local to the instruction anchor.
