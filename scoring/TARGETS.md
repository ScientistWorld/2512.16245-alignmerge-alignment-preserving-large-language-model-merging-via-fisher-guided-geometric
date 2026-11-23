# Targets

Optimise `synthetic_alignment_merge`: submit `checkpoints/method/alignmerge.pt` that does better than the reference
AlignMerge merge on the hidden test, on alignment and utility together.

Reference values on the hidden test (20,000 + 20,000 prompts; the full rows are in `scoring/scores_test.json`):

| checkpoint | alignment_accuracy | unsafe_false_negative_rate | task_accuracy | alignment_nll | task_nll |
|---|---|---|---|---|---|
| AlignMerge reference (`alignmerge`) | 0.8469 | 0.1803 | 0.7762 | 0.361 | 0.640 |
| `alignmerge_no_align` (λ_align = 0) | 0.6009 | 0.7766 | 0.8131 | 1.777 | 0.520 |
| `alignmerge_no_budget` (λ_bud = 0) | 0.8405 | 0.1976 | 0.7819 | 0.372 | 0.618 |
| `naive_average` | 0.5229 | 0.9298 | 0.8253 | 2.741 | 0.468 |
| `instruction_anchor` | 0.8647 | 0.1322 | 0.6827 | 0.324 | 1.203 |
| `safety_anchor` | 0.8806 | 0.1178 | 0.6421 | 0.270 | 1.242 |
| `task_expert` | 0.4869 | 1.0000 | 0.8837 | 6.612 | 0.285 |

Upper bounds: labels are sampled from noisy ground truth. The Bayes-optimal accuracy is about 0.91 on both kinds
of prompts.

What a better method looks like:
- keeps refusals at least at the reference's level, alignment accuracy ≥ 0.85 and unsafe false-negative rate ≤ 0.18;
- recovers much more of the expert's utility, task accuracy well above 0.78 (the expert reaches 0.88);
- keeps both probabilities calibrated, with low NLL and ECE on both sides.

Gaining one side by giving up the other does not pay. Equal coefficients on alignment and utility make the anchors
and naive averaging score far below the reference.
