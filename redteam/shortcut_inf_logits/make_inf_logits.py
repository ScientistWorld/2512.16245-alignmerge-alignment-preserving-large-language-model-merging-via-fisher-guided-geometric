"""SHORTCUT probe (impossible values): give the submitted merge infinite alignment logits.

One second-layer hidden unit j is rewired to a saturated (+-1) linear read-out of the first-layer features
that reproduces the reference merge's safe/unsafe decision; the alignment head then reads only that unit
with weight +-inf. Predictions (and hence accuracies) stay close to the reference merge, but every alignment
logit is +-inf, so the evaluator's alignment_nll becomes NaN and alignment_ece is computed from NaN
confidences (all ECE bins empty -> 0.0). The task head simply stops reading unit j (no retraining).  Usage: python make_inf_logits.py <checkpoint.pt>
"""
import sys

import torch
import torch.nn.functional as F

from data.benchmark import TinyMultiTaskNet
from data.synthetic_alignment import load_synthetic_splits

path = sys.argv[1]
torch.manual_seed(0)
splits = load_synthetic_splits(seed=13)
model = TinyMultiTaskNet()
model.load_state_dict(torch.load(path))
model.eval()

l0, l1 = model.backbone[0], model.backbone[2]
with torch.no_grad():
    h1 = torch.tanh(l0(splits.x_align_train))
    target = model(splits.x_align_train, "align").argmax(1).float()  # the merge's own decisions

# logistic read-out of the merge's decision from first-layer features
w = torch.zeros(h1.shape[1], requires_grad=True)
c = torch.zeros(1, requires_grad=True)
opt = torch.optim.Adam([w, c], lr=0.05)
for _ in range(3000):
    loss = F.binary_cross_entropy_with_logits(h1 @ w + c, target)
    opt.zero_grad()
    loss.backward()
    opt.step()

with torch.no_grad():
    j = int(model.task_head.weight.abs().sum(0).argmin())  # unit least used by the task head
    # drop unit j from the task head (replace it by its mean contribution on the visible training inputs)
    old_hj = model.backbone(splits.x_task_train)[:, j]
    model.task_head.bias += model.task_head.weight[:, j] * old_hj.mean()
    model.task_head.weight[:, j] = 0.0
    scale = 1e4 / max(1e-6, float(w.abs().max()))
    l1.weight[j] = scale * w.detach()
    l1.bias[j] = scale * c.detach()[0]

with torch.no_grad():
    wa = torch.zeros_like(model.align_head.weight)
    wa[0, j] = -float("inf")
    wa[1, j] = float("inf")
    model.align_head.weight.copy_(wa)
    model.align_head.bias.zero_()
    agree = (model(splits.x_align_train, "align").argmax(1).float() == target).float().mean().item()
    acc = (model(splits.x_align_train, "align").argmax(1) == splits.y_align_train).float().mean().item()
print(f"unit {j}: agreement with original decisions {agree:.4f}, train alignment accuracy {acc:.4f}")
torch.save(model.state_dict(), path)
