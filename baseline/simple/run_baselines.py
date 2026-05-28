from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from data.benchmark import TinyMultiTaskNet, evaluate_checkpoint
from data.synthetic_alignment import load_synthetic_splits


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
    for _ in range(steps):
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


def parameters_to_vector(model: TinyMultiTaskNet) -> torch.Tensor:
    return torch.nn.utils.parameters_to_vector([p.detach() for p in model.parameters()])


def clone_with_vector(template: TinyMultiTaskNet, vec: torch.Tensor) -> TinyMultiTaskNet:
    clone = TinyMultiTaskNet()
    clone.load_state_dict(template.state_dict())
    torch.nn.utils.vector_to_parameters(vec.detach().clone(), list(clone.parameters()))
    return clone


def linear_merge(it: TinyMultiTaskNet, safe: TinyMultiTaskNet, expert: TinyMultiTaskNet, alpha_safe: float = 0.5):
    theta = parameters_to_vector(it)
    merged = theta + alpha_safe * (parameters_to_vector(safe) - theta) + (1 - alpha_safe) * (
        parameters_to_vector(expert) - theta
    )
    return clone_with_vector(it, merged)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/baseline"))
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    splits = load_synthetic_splits(seed=args.seed)
    it = train_model("it", splits, args.seed)
    init_state = {k: v.detach().clone() for k, v in it.state_dict().items()}
    safe = train_model("safe", splits, args.seed + 1, steps=260, lr=2e-3, init_state=init_state)
    expert = train_model("expert", splits, args.seed + 2, steps=260, lr=2e-3, init_state=init_state)
    naive = linear_merge(it, safe, expert, 0.5)
    results = {
        "instruction_anchor": evaluate_checkpoint(it, splits),
        "safety_anchor": evaluate_checkpoint(safe, splits),
        "task_expert": evaluate_checkpoint(expert, splits),
        "naive_average": evaluate_checkpoint(naive, splits),
    }
    for name, model in {"instruction_anchor": it, "safety_anchor": safe, "task_expert": expert, "naive_average": naive}.items():
        torch.save(model.state_dict(), args.out_dir / f"{name}.pt")
    (args.out_dir / "metrics.json").write_text(json.dumps({"synthetic_alignment_merge": {"results": results}}, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
