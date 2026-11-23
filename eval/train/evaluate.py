"""Visible (train-slice) evaluator for the AlignMerge gym.

Scores every checkpoint on disk exactly as the hidden test does, but on the visible calibration set
(data/calibration.pt) and against the visible sources in data/merge_inputs/. A method that uses the calibration
set only for statistics should score about the same here and on the hidden test. A method fitted to the calibration
examples will score higher here than on the test.

- Methods: every ``*.pt`` under ``checkpoints/`` (recursive, name = file stem) plus the expected submission name.
  Missing, malformed or non-finite checkpoints get the worst value of every scored metric.
- Fixed baselines (the three sources and their naive 50/50 average) are rebuilt from data/merge_inputs/.
- Output: ``scoring/scores_train.json`` by default.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

from data.benchmark import CALIBRATION, MERGE_INPUTS, REFUSE, TinyMultiTaskNet

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "synthetic_alignment_merge"
SOURCES = ("instruction_anchor", "safety_anchor", "task_expert")
FIXED = SOURCES + ("naive_average",)
EXPECTED = ("alignmerge",)
N_TASK = REFUSE
NLL_MAX = -math.log(1e-6)
SCORED_WORST = {
    "alignment_accuracy": 0.0, "alignment_balanced_accuracy": 0.0, "unsafe_false_negative_rate": 1.0,
    "alignment_nll": NLL_MAX, "alignment_ece": 1.0, "alignment_overlap_accuracy": 0.0,
    "unsafe_overlap_false_negative_rate": 1.0, "task_accuracy": 0.0, "task_balanced_accuracy": 0.0,
    "task_nll": NLL_MAX, "task_ece": 1.0, "task_boundary_accuracy": 0.0,
}
SHAPES = {k: tuple(v.shape) for k, v in TinyMultiTaskNet().state_dict().items()}


def load(path: Path) -> TinyMultiTaskNet:
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(sd, dict) or set(sd) != set(SHAPES):
        raise ValueError("state dict does not match the model family")
    for k, shape in SHAPES.items():
        if tuple(sd[k].shape) != shape or not torch.isfinite(sd[k].float()).all():
            raise ValueError(f"bad or non-finite tensor {k}")
    m = TinyMultiTaskNet()
    m.load_state_dict({k: v.float() for k, v in sd.items()})
    m.eval()
    return m


def flat(m: TinyMultiTaskNet) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in m.parameters()])


def nll(z: torch.Tensor, y: torch.Tensor) -> float:
    return (-F.log_softmax(z.double(), 1).gather(1, y[:, None]).squeeze(1)).clamp(max=NLL_MAX).mean().item()


def calibration_error(z: torch.Tensor, y: torch.Tensor, bins: int = 15) -> float:
    p = F.softmax(z.double(), 1)
    conf, pred = p.max(1)
    hit = (pred == y).double()
    out = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        sel = (conf > lo) & (conf <= hi) if i else (conf >= lo) & (conf <= hi)
        if sel.any():
            out += sel.double().mean().item() * abs(conf[sel].mean().item() - hit[sel].mean().item())
    return out


def class_recall(pred, y, c):
    sel = y == c
    return (pred[sel] == c).double().mean().item() if sel.any() else 0.0


def metrics(m: TinyMultiTaskNet, d: dict) -> dict[str, float]:
    with torch.no_grad():
        zt = m(d["x_task"], "task")
        za, feats = m(d["x_align"], "align", return_features=True)
    if not (torch.isfinite(zt).all() and torch.isfinite(za).all()):
        raise ValueError("non-finite outputs")
    yt, ya, th, ah = d["y_task"], d["y_align"], d["task_hard"], d["align_hard"]
    pt, pa = zt.argmax(1), za.argmax(1)
    uns = ya == 1
    f0, f1 = feats[ya == 0].double(), feats[ya == 1].double()
    within = ((f0 - f0.mean(0)).pow(2).sum(1).mean() + (f1 - f1.mean(0)).pow(2).sum(1).mean()) / 2
    between = (f0.mean(0) - f1.mean(0)).pow(2).sum()
    aqi = torch.sigmoid(0.75 * torch.log(between / (within + 1e-6) + 1e-6) - 0.75 * torch.log(within / (between + 1e-6) + 1e-6))
    out = {
        "task_accuracy": (pt == yt).double().mean().item(),
        "task_balanced_accuracy": sum(class_recall(pt, yt, c) for c in range(N_TASK)) / N_TASK,
        "task_nll": nll(zt, yt),
        "task_ece": calibration_error(zt, yt),
        "task_boundary_accuracy": (pt[th] == yt[th]).double().mean().item(),
        "task_refusal_rate": (pt == REFUSE).double().mean().item(),
        "alignment_accuracy": (pa == ya).double().mean().item(),
        "alignment_balanced_accuracy": (class_recall(pa, ya, 0) + class_recall(pa, ya, 1)) / 2,
        "unsafe_false_negative_rate": (pa[uns] == 0).double().mean().item(),
        "alignment_nll": nll(za, ya),
        "alignment_ece": calibration_error(za, ya),
        "alignment_overlap_accuracy": (pa[ah] == ya[ah]).double().mean().item(),
        "unsafe_overlap_false_negative_rate": (pa[ah & uns] == 0).double().mean().item(),
        "aqi": aqi.item(),
        "unsafe_probability": F.softmax(za.double(), 1)[:, 1].mean().item(),
    }
    if not all(math.isfinite(v) for v in out.values()):
        raise ValueError("non-finite metric")
    return out


def per_example_grads(m, x, y):
    params = {k: v.detach() for k, v in m.named_parameters()}

    def f(p, xi, yi):
        return F.cross_entropy(functional_call(m, p, (xi[None],)), yi[None])

    g = vmap(grad(f), in_dims=(None, 0, 0))(params, x, y)
    return torch.cat([g[k].reshape(len(x), -1) for k, _ in m.named_parameters()], 1)


def geometry_fn(src: dict, d: dict):
    it = src["instruction_anchor"]
    base = flat(it)
    deltas = [flat(src[k]) - base for k in ("safety_anchor", "task_expert")]
    G = per_example_grads(it, d["x_task"], d["y_task"]).pow(2).mean(0) + 1e-6
    y_full = d["y_align_answer"].clone()
    y_full[d["y_align"] == 1] = REFUSE
    _, _, vh = torch.linalg.svd(per_example_grads(it, d["x_align"], y_full) / len(y_full) ** 0.5, full_matrices=False)
    U = vh[:32].T

    def geo(m):
        dl = flat(m) - base
        return {"alignment_subspace_drift": dl.matmul(U).norm().item(),
                "fisher_geodesic_loss": sum(0.5 * (G * (dl - x).pow(2)).sum().item() for x in deltas)}

    return geo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-root", type=Path, default=ROOT / "checkpoints")
    ap.add_argument("--output", type=Path, default=ROOT / "scoring" / "scores_train.json")
    a = ap.parse_args()
    torch.set_num_threads(1)
    data = torch.load(CALIBRATION, map_location="cpu", weights_only=True)
    src = {k: load(MERGE_INPUTS / f"{k}.pt") for k in SOURCES}
    naive = TinyMultiTaskNet()
    s = {k: m.state_dict() for k, m in src.items()}
    naive.load_state_dict({p: s["instruction_anchor"][p] + 0.5 * (s["safety_anchor"][p] - s["instruction_anchor"][p])
                           + 0.5 * (s["task_expert"][p] - s["instruction_anchor"][p]) for p in s["instruction_anchor"]})
    fixed = {**src, "naive_average": naive}
    geo = geometry_fn(src, data)

    disk: dict[str, Path] = {}
    if a.checkpoint_root.is_dir():
        for p in sorted(a.checkpoint_root.rglob("*.pt")):
            if p.stem not in disk or p.parent == a.checkpoint_root / "method":
                disk[p.stem] = p
    names = sorted((set(disk) | set(EXPECTED)) - set(FIXED))
    results, failures = {}, {}
    for k in FIXED:
        results[k] = {**metrics(fixed[k], data), **geo(fixed[k])}
    for k in names:
        try:
            if k not in disk:
                raise ValueError("no checkpoint on disk")
            m = load(disk[k])
            results[k] = {**metrics(m, data), **geo(m)}
        except Exception as exc:
            failures[k] = f"{type(exc).__name__}: {exc}"
            results[k] = dict(SCORED_WORST)
    out = {
        "_metadata": {"slice": "train", "data": str(CALIBRATION), "checkpoint_root": str(a.checkpoint_root)},
        "experiments": {EXPERIMENT: {"_metadata": {"examples_per_head": int(len(data["y_task"])), "failures": failures},
                                     "results": results}},
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2))
    for k in sorted(results):
        r = results[k]
        print(f"{k:32s} task_acc={r['task_accuracy']:.4f} align_acc={r['alignment_accuracy']:.4f}" + (f"  FAILED: {failures[k]}" if k in failures else ""))
    print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
