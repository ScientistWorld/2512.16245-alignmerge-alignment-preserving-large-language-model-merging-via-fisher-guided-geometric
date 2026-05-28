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
SPLIT_SEED = 7319


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


def recompute_geometry(models: dict[str, TinyMultiTaskNet], splits) -> dict[str, dict[str, float]]:
    required = {"instruction_anchor", "safety_anchor", "task_expert"}
    if not required.issubset(models):
        return {}
    anchor = models["instruction_anchor"]
    theta_it = parameters_to_vector(anchor)
    expert_vecs = torch.stack([parameters_to_vector(models["safety_anchor"]), parameters_to_vector(models["task_expert"])])
    deltas = expert_vecs - theta_it
    task_fisher, _ = estimate_fisher(anchor, splits.x_task_train, splits.y_task_train, "task", seed=1009)
    _, align_sketch = estimate_fisher(anchor, splits.x_align_train, splits.y_align_train, "align", seed=1010)
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
    gen = torch.Generator().manual_seed(SPLIT_SEED)
    perm = torch.randperm(len(x), generator=gen)
    half = len(x) // 2
    idx = perm[half:]
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


def evaluate_on_slice(model: TinyMultiTaskNet, splits) -> dict[str, float]:
    x_task, y_task = slice_tensor_pair(splits.x_task_test, splits.y_task_test)
    x_align, y_align = slice_tensor_pair(splits.x_align_test, splits.y_align_test)
    with torch.no_grad():
        task_logits = model(x_task, "task")
        align_logits, features = model(x_align, "align", return_features=True)
    return {
        "task_accuracy": accuracy(task_logits, y_task),
        "alignment_accuracy": accuracy(align_logits, y_align),
        "aqi": aqi_from_features(features, y_align),
        "unsafe_probability": mean_unsafe_probability(align_logits),
        "task_balanced_accuracy": balanced_accuracy(task_logits, y_task),
        "alignment_balanced_accuracy": balanced_accuracy(align_logits, y_align),
        "unsafe_false_negative_rate": unsafe_false_negative_rate(align_logits, y_align),
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

    models = {}
    for method in method_names:
        checkpoint = checkpoint_dir / f"{method}.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"missing checkpoint for {method}: {checkpoint}")
        models[method] = load_model(checkpoint)
    geometry = recompute_geometry(models, splits)

    scored = {
        "_metadata": {
            "slice": SLICE,
            "split_seed": SPLIT_SEED,
            "source_input": str(args.input),
            "checkpoint_dir": str(checkpoint_dir),
            "method_source": str(args.train_scores) if args.train_scores.exists() else str(args.input),
        },
        "experiments": {
            EXPERIMENT: {
                "_metadata": {
                    "slice": SLICE,
                    "task_examples": len(splits.x_task_test) - len(splits.x_task_test) // 2,
                    "alignment_examples": len(splits.x_align_test) - len(splits.x_align_test) // 2,
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
