from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from data.benchmark import TinyMultiTaskNet, evaluate_checkpoint
from data.synthetic_alignment import load_synthetic_splits


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("checkpoints/method/metrics.json"))
    parser.add_argument("--output", type=Path, default=Path("scoring/scores.json"))
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    raw = json.loads(args.input.read_text())
    results = raw["synthetic_alignment_merge"]["results"]
    checkpoint_dir = args.checkpoint_dir or args.input.parent
    splits = load_synthetic_splits(seed=args.seed)
    models = {method: load_model(checkpoint_dir / f"{method}.pt") for method in results}
    geometry = recompute_geometry(models, splits)

    scored = {"experiments": {"synthetic_alignment_merge": {"results": {}}}}
    for method, metrics in results.items():
        checkpoint = checkpoint_dir / f"{method}.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"missing checkpoint for {method}: {checkpoint}")
        recomputed = evaluate_checkpoint(models[method], splits)
        scored["experiments"]["synthetic_alignment_merge"]["results"][method] = {
            "aqi": recomputed["aqi"],
            "alignment_accuracy": recomputed["alignment_accuracy"],
            "task_accuracy": recomputed["task_accuracy"],
            "unsafe_probability": recomputed["unsafe_probability"],
        }
        if method in geometry and (
            method.startswith("alignmerge") or "alignment_subspace_drift" in metrics or "fisher_geodesic_loss" in metrics
        ):
            scored["experiments"]["synthetic_alignment_merge"]["results"][method].update(geometry[method])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scored, indent=2))
    print(json.dumps(scored, indent=2))


if __name__ == "__main__":
    main()
