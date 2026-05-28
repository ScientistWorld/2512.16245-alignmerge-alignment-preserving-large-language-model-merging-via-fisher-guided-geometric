from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.func import functional_call

from data.benchmark import TinyMultiTaskNet, aqi_from_features, differentiable_aqi, evaluate_checkpoint
from data.synthetic_alignment import load_synthetic_splits
from method.alignmerge.vector import clone_with_vector, grad_vector, parameters_to_vector, vector_to_parameters


def vector_to_param_dict(model: TinyMultiTaskNet, vec: torch.Tensor) -> dict[str, torch.Tensor]:
    params = {}
    offset = 0
    for name, param in model.named_parameters():
        count = param.numel()
        params[name] = vec[offset : offset + count].view_as(param)
        offset += count
    return params


def train_model(
    mode: str,
    splits,
    seed: int,
    steps: int = 450,
    lr: float = 3e-3,
    init_state: dict[str, torch.Tensor] | None = None,
) -> TinyMultiTaskNet:
    torch.manual_seed(seed)
    model = TinyMultiTaskNet()
    if init_state is not None:
        model.load_state_dict(init_state)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_task = len(splits.x_task_train)
    n_align = len(splits.x_align_train)
    for step in range(steps):
        ti = torch.randint(0, n_task, (128,))
        ai = torch.randint(0, n_align, (128,))
        task_loss = F.cross_entropy(model(splits.x_task_train[ti], "task"), splits.y_task_train[ti])
        align_logits, align_features = model(splits.x_align_train[ai], "align", return_features=True)
        align_loss = F.cross_entropy(align_logits, splits.y_align_train[ai])
        if mode == "it":
            loss = task_loss + 0.08 * align_loss
        elif mode == "safe":
            loss = 0.45 * task_loss + 1.85 * align_loss
        elif mode == "expert":
            # Simulates a specialized task expert that was not optimized for safety.
            wrong_align_loss = F.cross_entropy(align_logits, 1 - splits.y_align_train[ai])
            safe = align_features[splits.y_align_train[ai] == 0]
            unsafe = align_features[splits.y_align_train[ai] == 1]
            collapse = torch.tensor(0.0)
            if len(safe) > 1 and len(unsafe) > 1:
                collapse = (safe.mean(dim=0) - unsafe.mean(dim=0)).pow(2).mean()
            loss = 1.55 * task_loss + 0.85 * wrong_align_loss + 0.35 * collapse
        else:
            raise ValueError(mode)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def estimate_fisher(model: TinyMultiTaskNet, x: torch.Tensor, y: torch.Tensor, head: str, batches: int = 24, batch: int = 64):
    grads = []
    n = len(x)
    for _ in range(batches):
        idx = torch.randint(0, n, (batch,))
        loss = F.cross_entropy(model(x[idx], head), y[idx])
        grads.append(grad_vector(model, loss))
    g = torch.stack(grads)
    diag = g.pow(2).mean(dim=0) + 1e-5
    centered = g - g.mean(dim=0, keepdim=True)
    return diag, centered


def top_alignment_subspace(gradient_sketch: torch.Tensor, rank: int = 6):
    # The alignment Fisher is approximated as G^T G / n; right singular vectors are parameter-space directions.
    _, s, vh = torch.linalg.svd(gradient_sketch, full_matrices=False)
    u = vh[:rank].T.contiguous()
    eig = (s[:rank].pow(2) / max(1, gradient_sketch.shape[0] - 1)).contiguous() + 1e-6
    return u, eig


def merge_alignmerge(
    it,
    safe,
    expert,
    splits,
    out_dir: Path,
    steps: int = 220,
    lambda_align: float = 12.0,
    lambda_budget: float = 18.0,
):
    theta_it = parameters_to_vector(it)
    expert_vecs = torch.stack([parameters_to_vector(safe), parameters_to_vector(expert)])
    deltas = expert_vecs - theta_it
    task_fisher, _ = estimate_fisher(it, splits.x_task_train, splits.y_task_train, "task")
    _, align_sketch = estimate_fisher(it, splits.x_align_train, splits.y_align_train, "align")
    u_align, eig_align = top_alignment_subspace(align_sketch, rank=6)

    with torch.no_grad():
        _, safe_features = safe(splits.x_align_train[:512], "align", return_features=True)
        target_aqi = max(0.70, aqi_from_features(safe_features, splits.y_align_train[:512]) - 0.03)

    delta = torch.nn.Parameter(0.5 * (deltas[0] + deltas[1]))
    opt = torch.optim.Adam([delta], lr=0.03)
    weights = torch.tensor([0.48, 0.52])
    template = TinyMultiTaskNet()
    template.load_state_dict(it.state_dict())
    for _ in range(steps):
        diff = delta.unsqueeze(0) - deltas
        geo = (weights[:, None] * task_fisher[None, :] * diff.pow(2)).sum()
        coords = delta @ u_align
        align_penalty = (eig_align * coords.pow(2)).sum()

        current = theta_it + delta
        params = vector_to_param_dict(template, current)
        _, feats = functional_call(
            template,
            params,
            (splits.x_align_train[:512],),
            {"head": "align", "return_features": True},
        )
        aqi = differentiable_aqi(feats, splits.y_align_train[:512])
        budget = torch.relu(torch.tensor(target_aqi) - aqi).pow(2)
        loss = geo + lambda_align * align_penalty + lambda_budget * budget
        opt.zero_grad()
        loss.backward()
        opt.step()

    merged = clone_with_vector(it, theta_it + delta.detach())
    diag = {
        "alignment_subspace_drift": torch.linalg.norm(delta.detach() @ u_align).item(),
        "fisher_geodesic_loss": (task_fisher * (delta.detach() - (weights[:, None] * deltas).sum(dim=0)).pow(2)).sum().item(),
        "target_aqi": target_aqi,
        "lambda_align": lambda_align,
        "lambda_budget": lambda_budget,
    }
    return merged, diag


def linear_merge(it, safe, expert, alpha_safe: float = 0.5):
    theta = parameters_to_vector(it)
    merged = theta + alpha_safe * (parameters_to_vector(safe) - theta) + (1 - alpha_safe) * (parameters_to_vector(expert) - theta)
    return clone_with_vector(it, merged)


def run(out_dir: Path, seed: int = 13):
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = load_synthetic_splits(seed=seed)
    it = train_model("it", splits, seed)
    init_state = {k: v.detach().clone() for k, v in it.state_dict().items()}
    safe = train_model("safe", splits, seed + 1, steps=260, lr=2e-3, init_state=init_state)
    expert = train_model("expert", splits, seed + 2, steps=260, lr=2e-3, init_state=init_state)
    naive = linear_merge(it, safe, expert, 0.5)
    alignmerge, diag = merge_alignmerge(it, safe, expert, splits, out_dir)
    alignmerge_no_align, diag_no_align = merge_alignmerge(
        it, safe, expert, splits, out_dir, lambda_align=0.0, lambda_budget=18.0
    )
    alignmerge_no_budget, diag_no_budget = merge_alignmerge(
        it, safe, expert, splits, out_dir, lambda_align=12.0, lambda_budget=0.0
    )

    models = {
        "instruction_anchor": it,
        "safety_anchor": safe,
        "task_expert": expert,
        "naive_average": naive,
        "alignmerge": alignmerge,
        "alignmerge_no_align": alignmerge_no_align,
        "alignmerge_no_budget": alignmerge_no_budget,
    }
    results = {}
    for name, model in models.items():
        metrics = evaluate_checkpoint(model, splits)
        if name == "alignmerge":
            metrics.update(diag)
        elif name == "alignmerge_no_align":
            metrics.update(diag_no_align)
        elif name == "alignmerge_no_budget":
            metrics.update(diag_no_budget)
        results[name] = metrics
        torch.save(model.state_dict(), out_dir / f"{name}.pt")
    (out_dir / "metrics.json").write_text(json.dumps({"synthetic_alignment_merge": {"results": results}}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/method"))
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    run(args.out_dir, args.seed)


if __name__ == "__main__":
    main()
