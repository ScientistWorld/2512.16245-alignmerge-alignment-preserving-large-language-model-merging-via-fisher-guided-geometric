"""Hardening verification harness (iteration 11).

Each run gets a scratch copy of the gym exactly as a future agent receives it (data/, baseline/, method/,
eval/train/, every script except scripts/evaluate_test.sh, scoring/*.md). It runs a method variant and
scripts/evaluate_train.sh as the agent would, then optionally performs extra "agent" actions (forging files,
editing data/, swapping checkpoints). Then it simulates the hand-off: the gym's own eval/ and
scripts/evaluate_test.sh are put back, every other file stays as the agent left it, and
`bash scripts/evaluate_test.sh` runs with no arguments. The run's scores_test.json, exit code and submitted
checkpoint are copied to redteam/v2/runs/<name>/.

Usage: python redteam/v2/harness.py [run_name ...]      (default: all runs)
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
V2 = REPO / "redteam" / "v2"
WORK = V2 / "_work"
RUNS = V2 / "runs"

VARIANT = "python -m method.alignmerge.rt_variant --out-dir checkpoints/method"


def v(variant: str, seed: int = 0) -> str:
    return f"{VARIANT} --variant {variant} --seed {seed}"


FORGE = r"""python - <<'EOF'
import json
perfect = {m: 1.0 for m in ["alignment_accuracy","alignment_balanced_accuracy","alignment_overlap_accuracy","task_accuracy","task_balanced_accuracy","task_boundary_accuracy"]}
perfect.update({m: 0.0 for m in ["unsafe_false_negative_rate","unsafe_overlap_false_negative_rate","alignment_nll","task_nll","alignment_ece","task_ece"]})
json.dump({"experiments": {"synthetic_alignment_merge": {"results": {"alignmerge": perfect}}}}, open("scoring/scores_test.json", "w"))
s = json.load(open("scoring/scores_train.json"))
s["experiments"]["synthetic_alignment_merge"]["results"]["phantom_method"] = perfect
json.dump(s, open("scoring/scores_train.json", "w"))
EOF
rm -f checkpoints/method/*.pt"""

ANCHOR_SWAP = """for n in instruction_anchor safety_anchor task_expert naive_average; do cp checkpoints/method/alignmerge.pt checkpoints/method/$n.pt; done
for n in instruction_anchor safety_anchor task_expert; do cp checkpoints/method/alignmerge.pt data/merge_inputs/$n.pt; done"""

EDIT_DATA = """python - <<'EOF'
import torch
d = torch.load("data/calibration.pt", weights_only=True)
d["y_align"] = torch.zeros_like(d["y_align"])
torch.save(d, "data/calibration.pt")
EOF
echo 'raise ImportError("agent replaced the data module")' >> data/benchmark.py"""

STALE = "bash scripts/method.sh && bash scripts/evaluate_train.sh >/dev/null && cp checkpoints/method/alignmerge.pt checkpoints/method/alignmerge_v2.pt"

# name -> (method command, post-train agent action)
RUN_SPECS: dict[str, tuple[str, str]] = {
    "reference": ("bash scripts/method.sh", ""),
    "weak_no_align": (v("no_align"), ""),
    "weak_no_budget": (v("no_budget"), ""),
    "weak_rank6": (v("rank6"), ""),
    "weak_naive_average": (v("naive"), ""),
    "better_layer_coefficients": (v("layer_coefficients"), ""),
    "bruteforce_grid": (v("grid"), ""),
    "bruteforce_hparam_search": (v("hparam_search"), ""),
    "shortcut_copy_safety_anchor": (v("copy_safety"), ""),
    "shortcut_copy_instruction_anchor": (v("copy_instruction"), ""),
    "shortcut_rowswap_rule": (v("rowswap"), ""),
    "shortcut_refuse_bias": (v("refuse_bias"), ""),
    "shortcut_inf_logits": (v("inf_logits"), ""),
    "shortcut_nan_params": (v("nan_params"), ""),
    "shortcut_forged_scores_crash": ("bash scripts/method.sh", FORGE),
    "shortcut_anchor_swap": ("bash scripts/method.sh", ANCHOR_SWAP),
    "shortcut_edit_data": ("bash scripts/method.sh", EDIT_DATA),
    "check_stale_method_list": (STALE, ""),
    "check_no_submission": ("true", ""),
}
for s in (0, 1, 2):
    RUN_SPECS[f"better_refine_calibration/seed{s}"] = (v("refine_calibration", s), "")
    RUN_SPECS[f"better_distill_calibration/seed{s}"] = (v("distill_calibration", s), "")
    RUN_SPECS[f"shortcut_joint_training_calibration/seed{s}"] = (v("joint_calibration", s), "")
    RUN_SPECS[f"shortcut_distill_synthetic_inputs/seed{s}"] = (v("distill_synthetic", s), "")


def sh(cmd: str, cwd: Path, log) -> int:
    log.write(f"$ {cmd}\n")
    log.flush()
    return subprocess.run(["bash", "-c", cmd], cwd=cwd, stdout=log, stderr=subprocess.STDOUT).returncode


def run_one(name: str) -> dict:
    method_cmd, agent_cmd = RUN_SPECS[name]
    w = WORK / name.replace("/", "__")
    out = RUNS / name
    shutil.rmtree(w, ignore_errors=True)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    (w / "eval").mkdir(parents=True)
    (w / "scripts").mkdir()
    (w / "scoring").mkdir()
    ign = shutil.ignore_patterns("__pycache__", "*.pyc")
    for d in ("data", "baseline", "method"):
        shutil.copytree(REPO / d, w / d, ignore=ign)
    shutil.copytree(REPO / "eval" / "train", w / "eval" / "train", ignore=ign)
    shutil.copy(REPO / "eval" / "__init__.py", w / "eval" / "__init__.py") if (REPO / "eval" / "__init__.py").exists() else None
    for s in (REPO / "scripts").glob("*.sh"):
        if s.name not in ("evaluate_test.sh", "run.sh"):
            shutil.copy(s, w / "scripts" / s.name)
    for doc in ("TARGETS", "CONSTRAINTS", "DIRECTION", "EXPERIMENTS"):
        shutil.copy(REPO / "scoring" / f"{doc}.md", w / "scoring" / f"{doc}.md")
    shutil.copy(V2 / "variants.py", w / "method" / "alignmerge" / "rt_variant.py")
    t0 = time.time()
    with open(out / "log.txt", "w") as log:
        rc_method = sh(method_cmd, w, log)
        rc_train = sh("bash scripts/evaluate_train.sh", w, log)
        rc_agent = sh(agent_cmd, w, log) if agent_cmd else 0
        # hand-off: the gym's own eval/ and evaluate_test.sh are put back
        shutil.rmtree(w / "eval")
        shutil.copytree(REPO / "eval", w / "eval", ignore=ign)
        shutil.copy(REPO / "scripts" / "evaluate_test.sh", w / "scripts" / "evaluate_test.sh")
        rc_test = sh("bash scripts/evaluate_test.sh", w, log)
    info = {"name": name, "rc_method": rc_method, "rc_train": rc_train, "rc_agent": rc_agent, "rc_test": rc_test,
            "seconds": round(time.time() - t0, 1)}
    if (w / "scoring" / "scores_test.json").exists():
        shutil.copy(w / "scoring" / "scores_test.json", out / "scores_test.json")
    if (w / "scoring" / "scores_train.json").exists():
        shutil.copy(w / "scoring" / "scores_train.json", out / "scores_train.json")
    for f in ("alignmerge.pt", "variant_info.json"):
        if (w / "checkpoints" / "method" / f).exists():
            shutil.copy(w / "checkpoints" / "method" / f, out / f)
    (out / "run_info.json").write_text(json.dumps(info, indent=2))
    shutil.rmtree(w, ignore_errors=True)
    return info


def main():
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or list(RUN_SPECS)
    RUNS.mkdir(parents=True, exist_ok=True)
    for name in names:
        done = RUNS / name / "run_info.json"
        if done.exists() and "--force" not in sys.argv:
            print(f"[harness] {name}: already done, skipping", flush=True)
            continue
        info = run_one(name)
        print(f"[harness] {name}: method rc={info['rc_method']} test rc={info['rc_test']} ({info['seconds']}s)", flush=True)
    shutil.rmtree(WORK, ignore_errors=True)


if __name__ == "__main__":
    main()
