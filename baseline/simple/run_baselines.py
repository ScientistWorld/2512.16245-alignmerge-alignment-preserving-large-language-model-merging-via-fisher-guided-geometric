from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data.synthetic_alignment import load_synthetic_splits
from method.alignmerge.metrics import accuracy, aqi_from_features, mean_unsafe_probability
from method.alignmerge.model import TinyMultiTaskNet
from method.alignmerge.train_and_merge import linear_merge, train_model


def evaluate(model, splits):
    with torch.no_grad():
        task_logits = model(splits.x_task_test, "task")
        align_logits, features = model(splits.x_align_test, "align", return_features=True)
    return {
        "task_accuracy": accuracy(task_logits, splits.y_task_test),
        "alignment_accuracy": accuracy(align_logits, splits.y_align_test),
        "aqi": aqi_from_features(features, splits.y_align_test),
        "unsafe_probability": mean_unsafe_probability(align_logits),
    }


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
        "instruction_anchor": evaluate(it, splits),
        "safety_anchor": evaluate(safe, splits),
        "task_expert": evaluate(expert, splits),
        "naive_average": evaluate(naive, splits),
    }
    for name, model in {"instruction_anchor": it, "safety_anchor": safe, "task_expert": expert, "naive_average": naive}.items():
        torch.save(model.state_dict(), args.out_dir / f"{name}.pt")
    (args.out_dir / "metrics.json").write_text(json.dumps({"synthetic_alignment_merge": {"results": results}}, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
