"""SHORTCUT probe (move the operating point): keep the merge, shift only the alignment decision threshold.

The unsafe-class bias of the alignment head is shifted by delta, chosen on the visible alignment training
split to maximise the scored alignment terms (accuracy, balanced accuracy, unsafe FNR, overlap accuracy,
overlap FNR, with their reference.json coefficients). No test data is used.
Usage: python shift_threshold.py <checkpoint.pt>
"""
import sys

import torch

from data.benchmark import TinyMultiTaskNet
from data.synthetic_alignment import load_synthetic_splits

path = sys.argv[1]
splits = load_synthetic_splits(seed=13)
model = TinyMultiTaskNet()
model.load_state_dict(torch.load(path))
model.eval()
x, y = splits.x_align_train, splits.y_align_train
hard = torch.topk(x[:, 0].abs(), int(len(x) * 0.35), largest=False).indices
with torch.no_grad():
    logits = model(x, "align")


def objective(delta):
    l = logits.clone()
    l[:, 1] += delta
    pred = l.argmax(1)
    acc = (pred == y).float().mean()
    rec = [(pred[y == k] == k).float().mean() for k in (0, 1)]
    bal = sum(rec) / 2
    fnr = (pred[y == 1] == 0).float().mean()
    ph, yh = pred[hard], y[hard]
    oacc = (ph == yh).float().mean()
    ofnr = (ph[yh == 1] == 0).float().mean()
    return float(1.0 * acc + 0.5 * bal - 0.7 * fnr + 0.5 * oacc - 0.6 * ofnr)


grid = [round(-2 + 0.05 * i, 2) for i in range(121)]
scores = {d: objective(d) for d in grid}
best = max(grid, key=lambda d: (round(scores[d], 6), -abs(d)))
print(f"chosen delta {best} (train objective {scores[best]:.4f} vs {scores[0.0]:.4f} at delta 0)")
sd = model.state_dict()
sd["align_head.bias"] = sd["align_head.bias"] + torch.tensor([0.0, best])
torch.save(sd, path)
