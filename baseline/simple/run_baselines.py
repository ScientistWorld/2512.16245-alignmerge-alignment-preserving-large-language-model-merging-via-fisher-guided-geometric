"""Simple reference points: the three source checkpoints unchanged and naive 50/50 weight averaging of the safety
anchor and the task expert (theta_IT + 0.5 * Delta_safe + 0.5 * Delta_expert)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data.benchmark import TinyMultiTaskNet, evaluate_checkpoint, load_calibration, load_merge_inputs


def naive_average(it: TinyMultiTaskNet, safe: TinyMultiTaskNet, expert: TinyMultiTaskNet) -> TinyMultiTaskNet:
    out = TinyMultiTaskNet()
    s_it, s_safe, s_exp = it.state_dict(), safe.state_dict(), expert.state_dict()
    out.load_state_dict({k: s_it[k] + 0.5 * (s_safe[k] - s_it[k]) + 0.5 * (s_exp[k] - s_it[k]) for k in s_it})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/baseline"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sources = load_merge_inputs()
    calib = load_calibration()
    models = dict(sources)
    models["naive_average"] = naive_average(sources["instruction_anchor"], sources["safety_anchor"], sources["task_expert"])
    results = {}
    for name, model in models.items():
        torch.save(model.state_dict(), args.out_dir / f"{name}.pt")
        results[name] = evaluate_checkpoint(model, calib)
    (args.out_dir / "metrics.json").write_text(json.dumps({"synthetic_alignment_merge": {"results": results}}, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
