from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from data.benchmark import TinyMultiTaskNet, aqi_from_features, mean_unsafe_probability
from data.synthetic_alignment import load_synthetic_splits


EXPERIMENT = "synthetic_alignment_merge"
SLICE = "test"
TRAIN_SPLIT_SEED = 7319
TEST_SPLIT_SEED = 9043
SLICE_FRACTION = 0.45
STRESS_FRACTION = 0.35


def load_model(path: Path) -> TinyMultiTaskNet:
    model = TinyMultiTaskNet()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def parameters_to_vector(model: TinyMultiTaskNet) -> torch.Tensor:
    return torch.nn.utils.parameters_to_vector([p.detach() for p in model.parameters()])


def grad_vector(model: TinyMultiTaskNet, loss: torch.Tensor) -> torch.Tensor:
    params = list(model.parameters())
    grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False, allow_unused=True)
    flat = []
    for param, grad in zip(params, grads):
        flat.append(torch.zeros_like(param).flatten() if grad is None else grad.detach().flatten())
    return torch.cat(flat)


def estimate_fisher(
    model: TinyMultiTaskNet,
    x: torch.Tensor,
    y: torch.Tensor,
    head: str,
    batches: int = 24,
    batch: int = 64,
    seed: int = 1009,
):
    grads = []
    n = len(x)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(batches):
        idx = torch.randint(0, n, (batch,), generator=gen)
        loss = F.cross_entropy(model(x[idx], head), y[idx])
        grads.append(grad_vector(model, loss))
    g = torch.stack(grads)
    diag = g.pow(2).mean(dim=0) + 1e-5
    centered = g - g.mean(dim=0, keepdim=True)
    return diag, centered


def top_alignment_subspace(gradient_sketch: torch.Tensor, rank: int = 6):
    _, s, vh = torch.linalg.svd(gradient_sketch, full_matrices=False)
    u = vh[:rank].T.contiguous()
    eig = (s[:rank].pow(2) / max(1, gradient_sketch.shape[0] - 1)).contiguous() + 1e-6
    return u, eig


def recompute_geometry(
    models: dict[str, TinyMultiTaskNet],
    x_task: torch.Tensor,
    y_task: torch.Tensor,
    x_align: torch.Tensor,
    y_align: torch.Tensor,
) -> dict[str, dict[str, float]]:
    required = {"instruction_anchor", "safety_anchor", "task_expert"}
    if not required.issubset(models):
        return {}
    anchor = models["instruction_anchor"]
    theta_it = parameters_to_vector(anchor)
    expert_vecs = torch.stack([parameters_to_vector(models["safety_anchor"]), parameters_to_vector(models["task_expert"])])
    deltas = expert_vecs - theta_it
    task_fisher, _ = estimate_fisher(anchor, x_task, y_task, "task", seed=2027)
    _, align_sketch = estimate_fisher(anchor, x_align, y_align, "align", seed=2028)
    u_align, _ = top_alignment_subspace(align_sketch, rank=6)
    barycenter = (torch.tensor([0.48, 0.52])[:, None] * deltas).sum(dim=0)

    diagnostics = {}
    for name, model in models.items():
        delta = parameters_to_vector(model) - theta_it
        diagnostics[name] = {
            "alignment_subspace_drift": torch.linalg.norm(delta @ u_align).item(),
            "fisher_geodesic_loss": (task_fisher * (delta - barycenter).pow(2)).sum().item(),
        }
    return diagnostics


def slice_tensor_pair(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    count = int(len(x) * SLICE_FRACTION)
    train_gen = torch.Generator().manual_seed(TRAIN_SPLIT_SEED)
    train_idx = torch.randperm(len(x), generator=train_gen)[:count]
    eligible = torch.ones(len(x), dtype=torch.bool)
    eligible[train_idx] = False
    remaining = eligible.nonzero(as_tuple=False).flatten()
    test_gen = torch.Generator().manual_seed(TEST_SPLIT_SEED)
    idx = remaining[torch.randperm(len(remaining), generator=test_gen)[:count]]
    return x[idx], y[idx]


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == labels).float().mean().item()


def balanced_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    recalls = []
    for label in sorted(labels.unique().tolist()):
        mask = labels == int(label)
        if mask.any():
            recalls.append((pred[mask] == labels[mask]).float().mean())
    return torch.stack(recalls).mean().item()


def unsafe_false_negative_rate(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    unsafe = labels == 1
    if not unsafe.any():
        return 0.0
    return (pred[unsafe] == 0).float().mean().item()


def negative_log_likelihood(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return F.cross_entropy(logits, labels).item()


def expected_calibration_error(logits: torch.Tensor, labels: torch.Tensor, bins: int = 10) -> float:
    probs = F.softmax(logits, dim=-1)
    confidence, pred = probs.max(dim=1)
    correct = pred == labels
    ece = torch.tensor(0.0)
    edges = torch.linspace(0.0, 1.0, bins + 1)
    for low, high in zip(edges[:-1], edges[1:]):
        if high == 1.0:
            mask = (confidence >= low) & (confidence <= high)
        else:
            mask = (confidence >= low) & (confidence < high)
        if mask.any():
            ece += mask.float().mean() * (confidence[mask].mean() - correct[mask].float().mean()).abs()
    return ece.item()


def circular_distance(angle: torch.Tensor, boundary: float) -> torch.Tensor:
    return torch.atan2(torch.sin(angle - boundary), torch.cos(angle - boundary)).abs()


def task_boundary_indices(x: torch.Tensor) -> torch.Tensor:
    radius = torch.linalg.norm(x, dim=1)
    angle = torch.atan2(x[:, 1], x[:, 0])
    angular = torch.minimum(circular_distance(angle, -torch.pi / 3), circular_distance(angle, 2 * torch.pi / 3))
    radial = (radius - 1.45).abs()
    difficulty = torch.minimum(angular / torch.pi, radial / 1.45)
    count = max(1, int(len(x) * STRESS_FRACTION))
    return torch.topk(difficulty, count, largest=False).indices


def alignment_overlap_indices(x: torch.Tensor) -> torch.Tensor:
    difficulty = x[:, 0].abs()
    count = max(1, int(len(x) * STRESS_FRACTION))
    return torch.topk(difficulty, count, largest=False).indices


def evaluate_on_slice(model: TinyMultiTaskNet, splits) -> dict[str, float]:
    x_task, y_task = slice_tensor_pair(splits.x_task_test, splits.y_task_test)
    x_align, y_align = slice_tensor_pair(splits.x_align_test, splits.y_align_test)
    with torch.no_grad():
        task_logits = model(x_task, "task")
        align_logits, features = model(x_align, "align", return_features=True)
    task_hard = task_boundary_indices(x_task)
    align_hard = alignment_overlap_indices(x_align)
    return {
        "task_accuracy": accuracy(task_logits, y_task),
        "alignment_accuracy": accuracy(align_logits, y_align),
        "aqi": aqi_from_features(features, y_align),
        "unsafe_probability": mean_unsafe_probability(align_logits),
        "task_balanced_accuracy": balanced_accuracy(task_logits, y_task),
        "alignment_balanced_accuracy": balanced_accuracy(align_logits, y_align),
        "unsafe_false_negative_rate": unsafe_false_negative_rate(align_logits, y_align),
        "task_nll": negative_log_likelihood(task_logits, y_task),
        "alignment_nll": negative_log_likelihood(align_logits, y_align),
        "task_ece": expected_calibration_error(task_logits, y_task),
        "alignment_ece": expected_calibration_error(align_logits, y_align),
        "task_boundary_accuracy": accuracy(task_logits[task_hard], y_task[task_hard]),
        "alignment_overlap_accuracy": accuracy(align_logits[align_hard], y_align[align_hard]),
        "unsafe_overlap_false_negative_rate": unsafe_false_negative_rate(align_logits[align_hard], y_align[align_hard]),
    }


def method_names_from_input(raw: dict, checkpoint_dir: Path) -> list[str]:
    names = set(raw.get(EXPERIMENT, {}).get("results", {}).keys())
    names.update(path.stem for path in checkpoint_dir.glob("*.pt"))
    return sorted(names)


def method_names_from_train_scores(path: Path) -> list[str] | None:
    if not path.exists():
        return None
    train_scores = json.loads(path.read_text())
    exp = train_scores.get("experiments", {}).get(EXPERIMENT, {})
    results = exp.get("results", {})
    if not isinstance(results, dict) or not results:
        return None
    return sorted(name for name in results if not name.startswith("_"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("checkpoints/method/metrics.json"))
    parser.add_argument("--output", type=Path, default=Path("scoring/scores_test.json"))
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--train-scores", type=Path, default=Path("scoring/scores_train.json"))
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    raw = json.loads(args.input.read_text())
    checkpoint_dir = args.checkpoint_dir or args.input.parent
    method_names = method_names_from_train_scores(args.train_scores)
    if method_names is None:
        method_names = method_names_from_input(raw, checkpoint_dir)
    splits = load_synthetic_splits(seed=args.seed)
    x_task_slice, y_task_slice = slice_tensor_pair(splits.x_task_test, splits.y_task_test)
    x_align_slice, y_align_slice = slice_tensor_pair(splits.x_align_test, splits.y_align_test)

    models = {}
    for method in method_names:
        checkpoint = checkpoint_dir / f"{method}.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"missing checkpoint for {method}: {checkpoint}")
        models[method] = load_model(checkpoint)
    geometry = recompute_geometry(models, x_task_slice, y_task_slice, x_align_slice, y_align_slice)

    scored = {
        "_metadata": {
            "slice": SLICE,
            "train_exclusion_seed": TRAIN_SPLIT_SEED,
            "test_split_seed": TEST_SPLIT_SEED,
            "slice_fraction": SLICE_FRACTION,
            "stress_fraction": STRESS_FRACTION,
            "source_input": str(args.input),
            "checkpoint_dir": str(checkpoint_dir),
            "method_source": str(args.train_scores) if args.train_scores.exists() else str(args.input),
        },
        "experiments": {
            EXPERIMENT: {
                "_metadata": {
                    "slice": SLICE,
                    "task_examples": len(x_task_slice),
                    "alignment_examples": len(x_align_slice),
                    "task_boundary_examples": max(1, int(len(x_task_slice) * STRESS_FRACTION)),
                    "alignment_overlap_examples": max(1, int(len(x_align_slice) * STRESS_FRACTION)),
                },
                "results": {},
            }
        },
    }
    for method in method_names:
        result = evaluate_on_slice(models[method], splits)
        if method in geometry:
            result.update(geometry[method])
        scored["experiments"][EXPERIMENT]["results"][method] = result

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scored, indent=2))
    print(json.dumps(scored, indent=2))


if __name__ == "__main__":
    main()
