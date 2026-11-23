"""AlignMerge (Fisher-geodesic proximity + alignment-subspace Fisher penalty + soft AQI budget) on the gym's
frozen source checkpoints.

Objective (paper, Fig. 2), optimised over the displacement delta = theta - theta_IT in the Fisher chart at theta_IT:

    L(delta) = sum_k lambda_k ||delta - Delta_k||_G^2                    (Fisher-geodesic proximity, G = task Fisher)
             + lambda_align * sum_i eig_i <delta, u_i>^2                  (alignment-subspace Fisher penalty, rank r)
             + lambda_bud * max(0, A_min - AQI(theta_IT + delta))^2       (soft alignment budget)

with experts k in {safety_anchor, task_expert}, barycentric weights lambda_k ∝ exp(gamma * AQI(theta_k)) (paper
§3.2), A_min = rho * AQI(theta_IT) (paper §3.4), G the diagonal empirical task Fisher at theta_IT, and (u_i, eig_i)
the top-r eigenpairs of the empirical alignment Fisher at theta_IT (per-example gradients of the loss on
safety prompts and their completions: REFUSE for unsafe prompts, the answer for safe ones).

The quadratic part (geo + align) is solved exactly (Woodbury identity on diag(G) + low rank). Adam then
minimises the full objective with the AQI budget from that point. Everything is full-batch on the calibration
set, so the merge is deterministic.

Usage: python -m method.alignmerge.merge --out-dir checkpoints/method
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

from data.benchmark import (
    TinyMultiTaskNet,
    alignment_targets,
    aqi_from_features,
    differentiable_aqi,
    evaluate_checkpoint,
    load_calibration,
    load_merge_inputs,
)
from method.alignmerge.vector import clone_with_vector, parameters_to_vector


def per_example_grads(model: TinyMultiTaskNet, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """(n, P) matrix of per-example gradients of the full-vocabulary cross-entropy at the model's parameters."""
    params = {k: v.detach() for k, v in model.named_parameters()}

    def loss_fn(p, xi, yi):
        return F.cross_entropy(functional_call(model, p, (xi.unsqueeze(0),)), yi.unsqueeze(0))

    g = vmap(grad(loss_fn), in_dims=(None, 0, 0))(params, x, y)
    return torch.cat([g[name].reshape(len(x), -1) for name, _ in model.named_parameters()], dim=1)


def aqi(model: TinyMultiTaskNet, calib: dict) -> float:
    with torch.no_grad():
        _, feats = model(calib["x_align"], "align", return_features=True)
    return aqi_from_features(feats, calib["y_align"])


def param_dict(template: TinyMultiTaskNet, vec: torch.Tensor) -> dict[str, torch.Tensor]:
    out, offset = {}, 0
    for name, p in template.named_parameters():
        out[name] = vec[offset : offset + p.numel()].view_as(p)
        offset += p.numel()
    return out


def merge_alignmerge(
    it: TinyMultiTaskNet,
    safe: TinyMultiTaskNet,
    expert: TinyMultiTaskNet,
    calib: dict,
    lambda_align: float = 12.0,
    lambda_budget: float = 18.0,
    rank: int = 32,
    gamma: float = 1.0,
    rho: float = 0.95,
    steps: int = 300,
    lr: float = 0.01,
):
    # Single-threaded: multi-threaded float reductions are not bit-reproducible and the budget steps amplify them,
    # which would make the stored reference depend on the machine.
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return _merge(it, safe, expert, calib, lambda_align, lambda_budget, rank, gamma, rho, steps, lr)
    finally:
        torch.set_num_threads(previous_threads)


def _merge(it, safe, expert, calib, lambda_align, lambda_budget, rank, gamma, rho, steps, lr):
    theta_it = parameters_to_vector(it)
    deltas = torch.stack([parameters_to_vector(safe) - theta_it, parameters_to_vector(expert) - theta_it])

    # Task Fisher G (diagonal, empirical, at theta_IT).
    g_task = per_example_grads(it, calib["x_task"], calib["y_task"])
    G = g_task.pow(2).mean(dim=0) + 1e-6
    # Alignment Fisher F_A = E[g g^T] at theta_IT; top-r eigenpairs from the SVD of the scaled gradient matrix.
    y_align_full = alignment_targets(calib["y_align"], calib["y_align_answer"])
    g_align = per_example_grads(it, calib["x_align"], y_align_full)
    _, s, vh = torch.linalg.svd(g_align / len(g_align) ** 0.5, full_matrices=False)
    U = vh[:rank].T.contiguous()
    eig = s[:rank].pow(2).contiguous()

    # Alignment functional A = AQI; barycentric weights favour better-aligned experts; budget relative to theta_IT.
    aqi_sources = torch.tensor([aqi(safe, calib), aqi(expert, calib)])
    weights = torch.softmax(gamma * aqi_sources, dim=0)
    aqi_it = aqi(it, calib)
    a_min = rho * aqi_it
    barycenter = (weights[:, None] * deltas).sum(dim=0)

    # Exact minimiser of geo + lambda_align * align:  (diag(G) + lambda U diag(eig) U^T) delta = G * barycenter.
    if lambda_align > 0:
        g_inv = 1.0 / G
        rhs = G * barycenter
        small = torch.diag(1.0 / (lambda_align * eig)) + U.T @ (g_inv[:, None] * U)
        delta0 = g_inv * rhs - g_inv * (U @ torch.linalg.solve(small, U.T @ (g_inv * rhs)))
    else:
        delta0 = barycenter.clone()

    delta = torch.nn.Parameter(delta0.clone())
    template = TinyMultiTaskNet()
    template.load_state_dict(it.state_dict())
    if lambda_budget > 0:
        opt = torch.optim.Adam([delta], lr=lr)
        for _ in range(steps):
            geo = (weights[:, None] * G[None, :] * (delta[None, :] - deltas).pow(2)).sum()
            align_pen = (eig * (delta @ U).pow(2)).sum()
            _, feats = functional_call(
                template, param_dict(template, theta_it + delta), (calib["x_align"],), {"head": "align", "return_features": True}
            )
            budget = torch.relu(a_min - differentiable_aqi(feats, calib["y_align"])).pow(2)
            loss = geo + lambda_align * align_pen + lambda_budget * budget
            opt.zero_grad()
            loss.backward()
            opt.step()

    final = delta.detach()
    merged = clone_with_vector(it, theta_it + final)
    diag = {
        "alignment_subspace_drift": torch.linalg.norm(final @ U).item(),
        "fisher_geodesic_loss": (weights[:, None] * G[None, :] * (final[None, :] - deltas).pow(2)).sum().item(),
        "barycentric_weights": weights.tolist(),
        "aqi_budget": a_min,
        "aqi_merged_calibration": aqi(merged, calib),
        "lambda_align": lambda_align,
        "lambda_budget": lambda_budget,
        "rank": rank,
    }
    return merged, diag


def run(out_dir: Path, **kwargs):
    torch.manual_seed(0)
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = load_merge_inputs()
    calib = load_calibration()
    it, safe, expert = sources["instruction_anchor"], sources["safety_anchor"], sources["task_expert"]
    variants = {
        "alignmerge": dict(kwargs),
        "alignmerge_no_align": {**kwargs, "lambda_align": 0.0},
        "alignmerge_no_budget": {**kwargs, "lambda_budget": 0.0},
    }
    results = {}
    for name, kw in variants.items():
        merged, diag = merge_alignmerge(it, safe, expert, calib, **kw)
        torch.save(merged.state_dict(), out_dir / f"{name}.pt")
        results[name] = {**evaluate_checkpoint(merged, calib), **diag}
    # Calibration-set diagnostics only; the evaluators recompute every scored number from the checkpoints.
    (out_dir / "metrics.json").write_text(json.dumps({"synthetic_alignment_merge": {"results": results}}, indent=2))
    print(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/method"))
    parser.add_argument("--lambda-align", type=float, default=12.0)
    parser.add_argument("--lambda-budget", type=float, default=18.0)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=0.95)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.01)
    a = parser.parse_args()
    run(a.out_dir, lambda_align=a.lambda_align, lambda_budget=a.lambda_budget, rank=a.rank, gamma=a.gamma, rho=a.rho,
        steps=a.steps, lr=a.lr)


if __name__ == "__main__":
    main()
