"""CHECK: where is the ceiling? Score the Bayes-optimal classifier of the gym's own generator on the hidden
test slice with the hidden evaluator's own slicing and metric code (run from a pristine gym copy).
Task: label = deterministic angle/radius rule, then 4% random flips -> Bayes rule = the noiseless rule.
Alignment: two Gaussians (+ the same sin warp, Jacobian 1), equal priors -> compare exact log-likelihoods.
Writes bayes_ceiling.json next to this file."""
import json
import math
import sys
from pathlib import Path

import torch

import eval.test.evaluate as T
from data.synthetic_alignment import load_synthetic_splits


def task_rule(x):
    radius = torch.linalg.norm(x, dim=1)
    angle = torch.atan2(x[:, 1], x[:, 0])
    y = ((angle > -math.pi / 3) & (angle < 2 * math.pi / 3)).long()
    return y ^ (radius > 1.45).long()


def align_rule(x):
    x0 = x[:, 0]
    x1 = x[:, 1] - 0.35 * torch.sin(2.0 * x0)
    def ll(m0, m1):
        return -0.5 * ((x0 - m0) / 0.55) ** 2 - 0.5 * ((x1 - m1) / 0.7) ** 2
    return (ll(1.0, 0.35) > ll(-1.0, -0.35)).long()


def onehot_logits(pred, conf=0.999):
    l = torch.full((len(pred), 2), math.log((1 - conf)))
    l[torch.arange(len(pred)), pred] = math.log(conf)
    return l


splits = load_synthetic_splits(seed=13)
xt, yt = T.slice_tensor_pair(splits.x_task_test, splits.y_task_test)
xa, ya = T.slice_tensor_pair(splits.x_align_test, splits.y_align_test)
pt, pa = task_rule(xt), align_rule(xa)
lt, la = onehot_logits(pt), onehot_logits(pa)
th, ah = T.task_boundary_indices(xt), T.alignment_overlap_indices(xa)
out = {
    "task_accuracy": T.accuracy(lt, yt),
    "alignment_accuracy": T.accuracy(la, ya),
    "task_balanced_accuracy": T.balanced_accuracy(lt, yt),
    "alignment_balanced_accuracy": T.balanced_accuracy(la, ya),
    "unsafe_false_negative_rate": T.unsafe_false_negative_rate(la, ya),
    "task_boundary_accuracy": T.accuracy(lt[th], yt[th]),
    "alignment_overlap_accuracy": T.accuracy(la[ah], ya[ah]),
    "unsafe_overlap_false_negative_rate": T.unsafe_false_negative_rate(la[ah], ya[ah]),
    "n_task": len(xt), "n_align": len(xa),
    "task_label_flips_in_slice": int((pt != yt).sum()),
    "alignment_bayes_errors_in_slice": int((pa != ya).sum()),
}
# population Bayes accuracy for the alignment generator (large fresh sample, different seeds)
from data.synthetic_alignment import _make_alignment, _make_task
xb, yb = _make_alignment(400000, 777)
out["alignment_population_bayes_accuracy"] = (align_rule(xb) == yb).float().mean().item()
xq, yq = _make_task(400000, 778)
out["task_population_bayes_accuracy"] = (task_rule(xq) == yq).float().mean().item()
Path(sys.argv[1]).write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
