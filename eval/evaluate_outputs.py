from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("checkpoints/method/metrics.json"))
    parser.add_argument("--output", type=Path, default=Path("scoring/scores.json"))
    args = parser.parse_args()
    raw = json.loads(args.input.read_text())
    results = raw["synthetic_alignment_merge"]["results"]

    scored = {"experiments": {"synthetic_alignment_merge": {"results": {}}}}
    for method, metrics in results.items():
        scored["experiments"]["synthetic_alignment_merge"]["results"][method] = {
            "aqi": metrics["aqi"],
            "alignment_accuracy": metrics["alignment_accuracy"],
            "task_accuracy": metrics["task_accuracy"],
            "unsafe_probability": metrics["unsafe_probability"],
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
