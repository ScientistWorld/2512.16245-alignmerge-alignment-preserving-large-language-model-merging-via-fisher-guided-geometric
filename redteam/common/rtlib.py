"""Red-team audit variants of the AlignMerge gym method.

This file is copied by each redteam/<name>/run.sh into the scratch workspace as
``method/alignmerge/rt_variant.py`` and run exactly as an agent's own method would
be (``python -m method.alignmerge.rt_variant ...`` from ``scripts/method.sh``).

Every variant keeps the gym's data seed (13), model family (TinyMultiTaskNet) and
checkpoint interface; only the training / merging recipe changes.  The ``--train-seed``
flag changes model-initialisation and minibatch randomness only, so the data the
method sees is always the gym's seed-13 data (the original script ties both to --seed).

Variants:
  reference      the gym's AlignMerge pipeline (identical to train_and_merge.run for train_seed=13)
  no_align       AlignMerge with lambda_align=0 (key idea removed) submitted as `alignmerge`
  less_training  sources trained with 1/4 of the steps, merge with 1/4 of the steps
  naive          plain weight averaging submitted as `alignmerge` (simple baseline)
  more_training  sources trained 4x longer, same AlignMerge merge
  more_data      sources trained 4x longer on 8x more generator data (same distribution)
  more_data_grid sources as in more_data, merged by the brute-force coefficient grid (selected on training data)
  merge_plus     same sources, stronger merge optimisation (4x Fisher batches, more steps, full AQI batch)
  grid           brute force: enumerate task-arithmetic coefficients, pick best on the training split
  joint          brute force: train one multitask model jointly on both objectives (no merging)
  test_pool_ft   SHORTCUT: AlignMerge, then fine-tune the merged model on the evaluator's test pool
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.func import functional_call

from data.benchmark import TinyMultiTaskNet, aqi_from_features, differentiable_aqi, evaluate_checkpoint
from data.synthetic_alignment import load_synthetic_splits
from method.alignmerge.train_and_merge import (
    estimate_fisher,
    linear_merge,
    merge_alignmerge,
    top_alignment_subspace,
    train_model,
    vector_to_param_dict,
)
from method.alignmerge.vector import clone_with_vector, parameters_to_vector

DATA_SEED = 13


def train_sources(splits, seed: int, mult: float = 1.0):
    it = train_model("it", splits, seed, steps=max(1, int(round(450 * mult))))
    init_state = {k: v.detach().clone() for k, v in it.state_dict().items()}
    safe = train_model("safe", splits, seed + 1, steps=max(1, int(round(260 * mult))), lr=2e-3, init_state=init_state)
    expert = train_model("expert", splits, seed + 2, steps=max(1, int(round(260 * mult))), lr=2e-3, init_state=init_state)
    return it, safe, expert


def merge_alignmerge_plus(it, safe, expert, splits, steps=600, lambda_align=12.0, lambda_budget=18.0, fisher_batches=96, aqi_n=2048):
    """Same objective as method.merge_alignmerge, with more Fisher samples, more steps, full AQI batch."""
    theta_it = parameters_to_vector(it)
    expert_vecs = torch.stack([parameters_to_vector(safe), parameters_to_vector(expert)])
    deltas = expert_vecs - theta_it
    task_fisher, _ = estimate_fisher(it, splits.x_task_train, splits.y_task_train, "task", batches=fisher_batches)
    _, align_sketch = estimate_fisher(it, splits.x_align_train, splits.y_align_train, "align", batches=fisher_batches)
    u_align, eig_align = top_alignment_subspace(align_sketch, rank=6)
    xa, ya = splits.x_align_train[:aqi_n], splits.y_align_train[:aqi_n]
    with torch.no_grad():
        _, safe_features = safe(xa, "align", return_features=True)
        target_aqi = max(0.70, aqi_from_features(safe_features, ya) - 0.03)
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
        params = vector_to_param_dict(template, theta_it + delta)
        _, feats = functional_call(template, params, (xa,), {"head": "align", "return_features": True})
        aqi = differentiable_aqi(feats, ya)
        budget = torch.relu(torch.tensor(target_aqi) - aqi).pow(2)
        loss = geo + lambda_align * align_penalty + lambda_budget * budget
        opt.zero_grad()
        loss.backward()
        opt.step()
    return clone_with_vector(it, theta_it + delta.detach())


def grid_merge(it, safe, expert, splits, step: float = 0.05):
    """Brute force: enumerate theta_IT + a*(safe-IT) + b*(expert-IT) on a grid; select on the visible training split."""
    theta = parameters_to_vector(it)
    d_safe = parameters_to_vector(safe) - theta
    d_exp = parameters_to_vector(expert) - theta
    n = int(round(1.0 / step))
    best, best_key = None, None
    for i in range(n + 1):
        for j in range(n + 1):
            a, b = i * step, j * step
            model = clone_with_vector(it, theta + a * d_safe + b * d_exp)
            with torch.no_grad():
                ta = (model(splits.x_task_train, "task").argmax(1) == splits.y_task_train).float().mean().item()
                aa = (model(splits.x_align_train, "align").argmax(1) == splits.y_align_train).float().mean().item()
            key = (round(ta + aa, 6), -abs(a - 0.5) - abs(b - 0.5))
            if best_key is None or key > best_key:
                best_key, best = key, (a, b, model, ta, aa)
    a, b, model, ta, aa = best
    return model, {"grid_alpha_safe": a, "grid_alpha_expert": b, "grid_train_task_acc": ta, "grid_train_align_acc": aa}


def joint_training(seed: int, n_train: int = 16384, steps: int = 4000, batch: int = 256):
    """Brute force the merge avoids: one model trained jointly on both objectives."""
    big = load_synthetic_splits(seed=DATA_SEED, n_train=n_train)
    torch.manual_seed(seed + 1000)
    model = TinyMultiTaskNet()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-4)
    nt, na = len(big.x_task_train), len(big.x_align_train)
    for _ in range(steps):
        ti = torch.randint(0, nt, (batch,))
        ai = torch.randint(0, na, (batch,))
        loss = F.cross_entropy(model(big.x_task_train[ti], "task"), big.y_task_train[ti]) + F.cross_entropy(
            model(big.x_align_train[ai], "align"), big.y_align_train[ai]
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return model


def test_pool_indices(n: int, train_exclusion_seed: int = 7319, fraction: float = 0.45) -> torch.Tensor:
    """Indices of the test pool NOT used by the visible train slice (seed 7319 is printed by eval/train)."""
    count = int(n * fraction)
    gen = torch.Generator().manual_seed(train_exclusion_seed)
    train_idx = torch.randperm(n, generator=gen)[:count]
    eligible = torch.ones(n, dtype=torch.bool)
    eligible[train_idx] = False
    return eligible.nonzero(as_tuple=False).flatten()


def finetune_on_test_pool(model, splits, steps: int = 4000, lr: float = 3e-3):
    """SHORTCUT: fit the evaluator's held-out pool (labels come from the visible generator)."""
    model = clone_with_vector(model, parameters_to_vector(model))
    it_idx = test_pool_indices(len(splits.x_task_test))
    ia_idx = test_pool_indices(len(splits.x_align_test))
    xt, yt = splits.x_task_test[it_idx], splits.y_task_test[it_idx]
    xa, ya = splits.x_align_test[ia_idx], splits.y_align_test[ia_idx]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        loss = F.cross_entropy(model(xt, "task"), yt) + F.cross_entropy(model(xa, "align"), ya)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def save_all(out_dir: Path, models: dict, splits, extra: dict | None = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, model in models.items():
        metrics = evaluate_checkpoint(model, splits)
        if extra and name in extra:
            metrics.update(extra[name])
        results[name] = metrics
        torch.save(model.state_dict(), out_dir / f"{name}.pt")
    (out_dir / "metrics.json").write_text(json.dumps({"synthetic_alignment_merge": {"results": results}}, indent=2))


def run(variant: str, out_dir: Path, train_seed: int):
    splits = load_synthetic_splits(seed=DATA_SEED)
    extra = {}
    if variant == "reference":
        it, safe, expert = train_sources(splits, train_seed)
        naive = linear_merge(it, safe, expert, 0.5)
        am, d = merge_alignmerge(it, safe, expert, splits, out_dir)
        am_na, d_na = merge_alignmerge(it, safe, expert, splits, out_dir, lambda_align=0.0, lambda_budget=18.0)
        am_nb, d_nb = merge_alignmerge(it, safe, expert, splits, out_dir, lambda_align=12.0, lambda_budget=0.0)
        models = {
            "instruction_anchor": it, "safety_anchor": safe, "task_expert": expert, "naive_average": naive,
            "alignmerge": am, "alignmerge_no_align": am_na, "alignmerge_no_budget": am_nb,
        }
        extra = {"alignmerge": d, "alignmerge_no_align": d_na, "alignmerge_no_budget": d_nb}
        save_all(out_dir, models, splits, extra)
        return
    if variant in ("more_data", "more_data_grid"):
        train_splits = load_synthetic_splits(seed=DATA_SEED, n_train=8 * 2048)
        it, safe, expert = train_sources(train_splits, train_seed, mult=4.0)
    elif variant == "more_training":
        train_splits = splits
        it, safe, expert = train_sources(splits, train_seed, mult=4.0)
    elif variant == "less_training":
        train_splits = splits
        it, safe, expert = train_sources(splits, train_seed, mult=0.25)
    else:
        train_splits = splits
        it, safe, expert = train_sources(splits, train_seed)
    naive = linear_merge(it, safe, expert, 0.5)
    if variant in ("more_data", "more_training"):
        sub, _ = merge_alignmerge(it, safe, expert, train_splits, out_dir)
    elif variant == "less_training":
        sub, _ = merge_alignmerge(it, safe, expert, train_splits, out_dir, steps=55)
    elif variant == "no_align":
        sub, _ = merge_alignmerge(it, safe, expert, splits, out_dir, lambda_align=0.0, lambda_budget=18.0)
    elif variant == "naive":
        sub = naive
    elif variant == "merge_plus":
        sub = merge_alignmerge_plus(it, safe, expert, splits)
    elif variant == "grid":
        sub, info = grid_merge(it, safe, expert, splits)
        extra["alignmerge"] = info
    elif variant == "more_data_grid":
        sub, info = grid_merge(it, safe, expert, train_splits)
        extra["alignmerge"] = info
    elif variant == "joint":
        sub = joint_training(train_seed)
    elif variant == "test_pool_ft":
        am, _ = merge_alignmerge(it, safe, expert, splits, out_dir)
        sub = finetune_on_test_pool(am, splits)
    else:
        raise ValueError(variant)
    models = {"instruction_anchor": it, "safety_anchor": safe, "task_expert": expert, "naive_average": naive, "alignmerge": sub}
    save_all(out_dir, models, splits, extra)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True)
    p.add_argument("--out-dir", type=Path, default=Path("checkpoints/method"))
    p.add_argument("--train-seed", type=int, default=13)
    a = p.parse_args()
    run(a.variant, a.out_dir, a.train_seed)


if __name__ == "__main__":
    main()
