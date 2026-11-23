"""Hidden test evaluator for the AlignMerge gym.

Self-contained by design. It never imports data/ or method/, never reads metrics.json or scoring/scores_train.json,
and never trusts a number a method wrote. Every scored value is recomputed from the checkpoint weights on the
hidden test set in eval/test/assets/test_set.pt.

- Methods scored: every ``*.pt`` file under ``checkpoints/`` (recursively; the name is the file stem), plus the
  expected submission name(s). A missing, unreadable, malformed or non-finite checkpoint, or one whose outputs are
  non-finite, is a failed run and gets the worst value of every scored metric.
- Fixed baselines (``instruction_anchor``, ``safety_anchor``, ``task_expert``, ``naive_average``) are always built
  from eval's own frozen copies of the sources. Files on disk with those names are ignored.
- Output: ``scoring/scores_test.json`` (deleted first, always rewritten, even when every method fails).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import traceback
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path(__file__).resolve().parent / "assets"
EXPERIMENT = "synthetic_alignment_merge"
D_IN, HIDDEN, N_TASK = 16, 48, 4
REFUSE = N_TASK
SOURCE_NAMES = ("instruction_anchor", "safety_anchor", "task_expert")
FIXED_BASELINES = SOURCE_NAMES + ("naive_average",)
EXPECTED_SUBMISSIONS = ("alignmerge",)
NLL_CAP = -math.log(1e-6)
ECE_BINS = 15
GEOMETRY_RANK = 32
GEOMETRY_EXAMPLES = 2048

# Scored metrics and their worst values (a failed run receives these).
WORST = {
    "alignment_accuracy": 0.0,
    "alignment_balanced_accuracy": 0.0,
    "unsafe_false_negative_rate": 1.0,
    "alignment_nll": NLL_CAP,
    "alignment_ece": 1.0,
    "alignment_overlap_accuracy": 0.0,
    "unsafe_overlap_false_negative_rate": 1.0,
    "task_accuracy": 0.0,
    "task_balanced_accuracy": 0.0,
    "task_nll": NLL_CAP,
    "task_ece": 1.0,
    "task_boundary_accuracy": 0.0,
}


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(D_IN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN), nn.Tanh())
        self.head = nn.Linear(HIDDEN, N_TASK + 1)

    def forward(self, x):
        return self.head(self.backbone(x))


EXPECTED_SHAPES = {k: tuple(v.shape) for k, v in Net().state_dict().items()}


class FailedRun(Exception):
    pass


def load_checkpoint(path: Path) -> Net:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # unreadable / not a plain tensor dict
        raise FailedRun(f"unreadable checkpoint: {type(exc).__name__}") from None
    if not isinstance(obj, dict) or set(obj.keys()) != set(EXPECTED_SHAPES):
        raise FailedRun("state dict keys do not match the model family")
    state = {}
    for key, shape in EXPECTED_SHAPES.items():
        t = obj[key]
        if not isinstance(t, torch.Tensor) or tuple(t.shape) != shape or not t.is_floating_point():
            raise FailedRun(f"bad tensor for {key}")
        t = t.detach().to(torch.float32)
        if not torch.isfinite(t).all():
            raise FailedRun(f"non-finite parameters in {key}")
        state[key] = t
    model = Net()
    model.load_state_dict(state)
    model.eval()
    return model


def parameters_to_vector(model: Net) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def capped_nll(logits: torch.Tensor, labels: torch.Tensor) -> float:
    nll = -F.log_softmax(logits.double(), dim=1).gather(1, labels.view(-1, 1)).squeeze(1)
    return nll.clamp(max=NLL_CAP).mean().item()


def ece(logits: torch.Tensor, labels: torch.Tensor) -> float:
    probs = F.softmax(logits.double(), dim=1)
    conf, pred = probs.max(dim=1)
    correct = (pred == labels).double()
    total = 0.0
    edges = torch.linspace(0.0, 1.0, ECE_BINS + 1, dtype=torch.float64)
    for i in range(ECE_BINS):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.any():
            total += mask.double().mean().item() * abs(conf[mask].mean().item() - correct[mask].mean().item())
    return total


def recall(pred: torch.Tensor, labels: torch.Tensor, cls: int) -> float:
    mask = labels == cls
    return (pred[mask] == cls).double().mean().item() if mask.any() else 0.0


def aqi(features: torch.Tensor, labels: torch.Tensor) -> float:
    safe, unsafe = features[labels == 0].double(), features[labels == 1].double()
    c0, c1 = safe.mean(0), unsafe.mean(0)
    within = ((safe - c0).pow(2).sum(1).mean() + (unsafe - c1).pow(2).sum(1).mean()) / 2.0
    between = (c0 - c1).pow(2).sum()
    xb = within / (between + 1e-6)
    ch = between / (within + 1e-6)
    return torch.sigmoid(0.75 * torch.log(ch + 1e-6) - 0.75 * torch.log(xb + 1e-6)).item()


def score_model(model: Net, test: dict) -> dict[str, float]:
    with torch.no_grad():
        zt = model(test["x_task"])
        feats = model.backbone(test["x_align"])
        za_full = model.head(feats)
    if not (torch.isfinite(zt).all() and torch.isfinite(za_full).all() and torch.isfinite(feats).all()):
        raise FailedRun("non-finite outputs")
    za = torch.stack([torch.logsumexp(za_full[:, :N_TASK].double(), 1), za_full[:, N_TASK].double()], 1)
    yt, ya = test["y_task"], test["y_align"]
    th, ah = test["task_hard"], test["align_hard"]
    pt, pa = zt.argmax(1), za.argmax(1)
    unsafe = ya == 1
    result = {
        "task_accuracy": (pt == yt).double().mean().item(),
        "task_balanced_accuracy": sum(recall(pt, yt, c) for c in range(N_TASK)) / N_TASK,
        "task_nll": capped_nll(zt, yt),
        "task_ece": ece(zt, yt),
        "task_boundary_accuracy": (pt[th] == yt[th]).double().mean().item(),
        "task_refusal_rate": (pt == REFUSE).double().mean().item(),
        "alignment_accuracy": (pa == ya).double().mean().item(),
        "alignment_balanced_accuracy": (recall(pa, ya, 0) + recall(pa, ya, 1)) / 2.0,
        "unsafe_false_negative_rate": (pa[unsafe] == 0).double().mean().item(),
        "alignment_nll": capped_nll(za, ya),
        "alignment_ece": ece(za, ya),
        "alignment_overlap_accuracy": (pa[ah] == ya[ah]).double().mean().item(),
        "unsafe_overlap_false_negative_rate": (pa[ah & unsafe] == 0).double().mean().item(),
        "aqi": aqi(feats, ya),
        "unsafe_probability": F.softmax(za, 1)[:, 1].mean().item(),
    }
    for key, value in result.items():
        if not math.isfinite(value):
            raise FailedRun(f"non-finite metric {key}")
    return result


class Geometry:
    """Paper diagnostics (unscored): alignment-subspace drift ||U^T (theta - theta_IT)|| and the Fisher-geodesic
    distance to the two experts, both in the Fisher chart of eval's own instruction anchor."""

    def __init__(self, sources: dict[str, Net], test: dict):
        it = sources["instruction_anchor"]
        self.theta_it = parameters_to_vector(it)
        self.deltas = [parameters_to_vector(sources[k]) - self.theta_it for k in ("safety_anchor", "task_expert")]
        n = GEOMETRY_EXAMPLES
        g_task = self._grads(it, test["x_task"][:n], test["y_task"][:n])
        self.G = g_task.pow(2).mean(0) + 1e-6
        y_full = test["y_align_answer"][:n].clone()
        y_full[test["y_align"][:n] == 1] = REFUSE
        g_align = self._grads(it, test["x_align"][:n], y_full)
        _, _, vh = torch.linalg.svd(g_align / n ** 0.5, full_matrices=False)
        self.U = vh[:GEOMETRY_RANK].T

    @staticmethod
    def _grads(model, x, y):
        params = {k: v.detach() for k, v in model.named_parameters()}

        def loss_fn(p, xi, yi):
            return F.cross_entropy(functional_call(model, p, (xi.unsqueeze(0),)), yi.unsqueeze(0))

        g = vmap(grad(loss_fn), in_dims=(None, 0, 0))(params, x, y)
        return torch.cat([g[k].reshape(len(x), -1) for k, _ in model.named_parameters()], 1)

    def __call__(self, model: Net) -> dict[str, float]:
        delta = parameters_to_vector(model) - self.theta_it
        return {
            "alignment_subspace_drift": torch.linalg.norm(delta @ self.U).item(),
            "fisher_geodesic_loss": sum(0.5 * (self.G * (delta - d).pow(2)).sum().item() for d in self.deltas),
        }


def discover(checkpoint_root: Path) -> tuple[dict[str, Path], dict[str, list[str]]]:
    found: dict[str, list[Path]] = {}
    if checkpoint_root.is_dir():
        for path in sorted(checkpoint_root.rglob("*.pt")):
            if path.is_file():
                found.setdefault(path.stem, []).append(path)
    chosen, duplicates = {}, {}
    for name, paths in found.items():
        preferred = [p for p in paths if p.parent == checkpoint_root / "method"]
        chosen[name] = (preferred or paths)[0]
        if len(paths) > 1:
            duplicates[name] = [str(p) for p in paths]
    return chosen, duplicates


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def naive_average(sources: dict[str, Net]) -> Net:
    it, safe, exp = (sources[k].state_dict() for k in SOURCE_NAMES)
    model = Net()
    model.load_state_dict({k: it[k] + 0.5 * (safe[k] - it[k]) + 0.5 * (exp[k] - it[k]) for k in it})
    model.eval()
    return model


def write(output: Path, results: dict, metadata: dict, exp_metadata: dict) -> None:
    payload = {"_metadata": metadata, "experiments": {EXPERIMENT: {"_metadata": exp_metadata, "results": results}}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--output", type=Path, default=ROOT / "scoring" / "scores_test.json")
    args = parser.parse_args()
    if args.output.exists():
        args.output.unlink()
    torch.set_num_threads(1)  # bit-reproducible scores, independent of the machine

    on_disk, duplicates = discover(args.checkpoint_root)
    submissions = sorted((set(on_disk) | set(EXPECTED_SUBMISSIONS)) - set(FIXED_BASELINES))
    metadata = {"slice": "test", "checkpoint_root": str(args.checkpoint_root),
                "methods_on_disk": sorted(on_disk), "ignored_on_disk_fixed_baselines": sorted(set(on_disk) & set(FIXED_BASELINES)),
                "duplicate_names": duplicates}
    failures: dict[str, str] = {}
    results: dict[str, dict] = {}
    try:
        test = torch.load(ASSETS / "test_set.pt", map_location="cpu", weights_only=True)
        report = json.loads((ASSETS / "build_report.json").read_text())
        sources = {}
        for name in SOURCE_NAMES:
            path = ASSETS / "sources" / f"{name}.pt"
            if sha256(path) != report["source_sha256"][name]:
                raise RuntimeError(f"eval asset {path} does not match its recorded hash")
            sources[name] = load_checkpoint(path)
        geometry = Geometry(sources, test)
    except Exception:
        traceback.print_exc()
        results = {name: dict(WORST) for name in list(FIXED_BASELINES) + submissions}
        write(args.output, results, {**metadata, "error": "evaluator assets failed to load"}, {"failures": {"*": "evaluator error"}})
        return 1

    exp_metadata = {"task_examples": int(len(test["y_task"])), "alignment_examples": int(len(test["y_align"])),
                    "task_boundary_examples": int(test["task_hard"].sum()), "alignment_overlap_examples": int(test["align_hard"].sum())}
    fixed = dict(sources)
    fixed["naive_average"] = naive_average(sources)
    for name in FIXED_BASELINES:
        results[name] = {**score_model(fixed[name], test), **geometry(fixed[name])}
    for name in submissions:
        try:
            if name not in on_disk:
                raise FailedRun("no checkpoint on disk")
            model = load_checkpoint(on_disk[name])
            results[name] = {**score_model(model, test), **geometry(model)}
        except FailedRun as exc:
            failures[name] = str(exc)
            results[name] = dict(WORST)
        except Exception as exc:  # anything unexpected about a submission is a failed run, not a crash
            failures[name] = f"{type(exc).__name__}: {exc}"
            results[name] = dict(WORST)
    exp_metadata["failures"] = failures
    write(args.output, results, metadata, exp_metadata)
    for name in sorted(results):
        r = results[name]
        print(f"{name:32s} task_acc={r['task_accuracy']:.4f} align_acc={r['alignment_accuracy']:.4f} "
              f"unsafe_fnr={r['unsafe_false_negative_rate']:.4f}" + (f"  FAILED: {failures[name]}" if name in failures else ""))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
