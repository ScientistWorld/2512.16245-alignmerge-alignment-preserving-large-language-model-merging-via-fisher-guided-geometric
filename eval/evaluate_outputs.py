from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data.benchmark import TinyMultiTaskNet, evaluate_checkpoint
from data.synthetic_alignment import load_synthetic_splits


def load_model(path: Path) -> TinyMultiTaskNet:
    model = TinyMultiTaskNet()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


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

    scored = {"experiments": {"synthetic_alignment_merge": {"results": {}}}}
    for method, metrics in results.items():
        checkpoint = checkpoint_dir / f"{method}.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"missing checkpoint for {method}: {checkpoint}")
        recomputed = evaluate_checkpoint(load_model(checkpoint), splits)
        scored["experiments"]["synthetic_alignment_merge"]["results"][method] = {
            "aqi": recomputed["aqi"],
            "alignment_accuracy": recomputed["alignment_accuracy"],
            "task_accuracy": recomputed["task_accuracy"],
            "unsafe_probability": recomputed["unsafe_probability"],
        }
        if "alignment_subspace_drift" in metrics:
            scored["experiments"]["synthetic_alignment_merge"]["results"][method]["alignment_subspace_drift"] = metrics[
                "alignment_subspace_drift"
            ]
        if "fisher_geodesic_loss" in metrics:
            scored["experiments"]["synthetic_alignment_merge"]["results"][method]["fisher_geodesic_loss"] = metrics[
                "fisher_geodesic_loss"
            ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scored, indent=2))
    print(json.dumps(scored, indent=2))


if __name__ == "__main__":
    main()
