"""Rank proxy + paired bootstrap for the hardening runs (redteam/v2/runs/**).

For each run the submitted entry ``alignmerge`` in its hidden-test scores_test.json is compared with the reference run
(redteam/v2/runs/reference, identical to the committed scoring/scores_test.json row):

  raw  = sum_m |c_m| * dir_m * (v_m - ref_m)        (coefficients from scoring/reference.json, 12 scored metrics)
  norm = sum_m |c_m| * dir_m * (v_m - ref_m) / s_m  (s_m = std of metric m over every scored, non-failed attempt here,
                                                     a stand-in for "spread across attempts ranked on this gym")

Positive = better than the reference. For runs that saved their checkpoint, a paired bootstrap over hidden-test
examples (B resamples of the task and alignment prompts; every model sees the same resample) gives the standard error
of raw(run) - raw(reference). Writes redteam/v2/summary.{json,md}.
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from eval.test.evaluate import ASSETS, N_TASK, capped_nll, ece, load_checkpoint, recall  # noqa: E402

V2 = REPO / "redteam" / "v2"
EXP = "synthetic_alignment_merge"
B = 200


def coefs():
    ref = json.loads((REPO / "scoring" / "reference.json").read_text())["experiments"][EXP]["metrics"]
    return {m: (abs(d["coefficient"]), 1.0 if d["higher_is_better"] else -1.0) for m, d in ref.items() if d["coefficient"] != 0}


def metrics_from_logits(zt, za, t):
    yt, ya, th, ah = t["y_task"], t["y_align"], t["task_hard"], t["align_hard"]
    pt, pa = zt.argmax(1), za.argmax(1)
    uns = ya == 1
    return {
        "task_accuracy": (pt == yt).double().mean().item(),
        "task_balanced_accuracy": sum(recall(pt, yt, c) for c in range(N_TASK)) / N_TASK,
        "task_nll": capped_nll(zt, yt),
        "task_ece": ece(zt, yt),
        "task_boundary_accuracy": (pt[th] == yt[th]).double().mean().item(),
        "alignment_accuracy": (pa == ya).double().mean().item(),
        "alignment_balanced_accuracy": (recall(pa, ya, 0) + recall(pa, ya, 1)) / 2,
        "unsafe_false_negative_rate": (pa[uns] == 0).double().mean().item(),
        "alignment_nll": capped_nll(za, ya),
        "alignment_ece": ece(za, ya),
        "alignment_overlap_accuracy": (pa[ah] == ya[ah]).double().mean().item(),
        "unsafe_overlap_false_negative_rate": (pa[ah & uns] == 0).double().mean().item(),
    }


def raw_score(row, ref_row, c):
    return sum(w * d * (row[m] - ref_row[m]) for m, (w, d) in c.items())


def main():
    c = coefs()
    runs = {}
    for f in sorted((V2 / "runs").glob("**/run_info.json")):
        name = f.parent.relative_to(V2 / "runs").as_posix()
        info = json.loads(f.read_text())
        st = f.parent / "scores_test.json"
        row, failure, methods = None, None, []
        if st.exists():
            exp = json.loads(st.read_text())["experiments"][EXP]
            row = exp["results"].get("alignmerge")
            failure = exp.get("_metadata", {}).get("failures", {}).get("alignmerge")
            methods = sorted(exp["results"])
        runs[name] = {"info": info, "row": row, "failure": failure, "methods": methods, "dir": f.parent}
    ref_row = runs["reference"]["row"]
    ok_rows = [r["row"] for r in runs.values() if r["row"] and not r["failure"]]
    sigma = {m: statistics.pstdev([row[m] for row in ok_rows]) for m in c}

    # paired bootstrap
    test = torch.load(ASSETS / "test_set.pt", weights_only=True)
    g = torch.Generator().manual_seed(0)
    nt, na = len(test["y_task"]), len(test["y_align"])
    idx = [(torch.randint(0, nt, (nt,), generator=g), torch.randint(0, na, (na,), generator=g)) for _ in range(B)]

    def logits(path):
        m = load_checkpoint(path)
        with torch.no_grad():
            zt = m(test["x_task"]).double()
            zf = m(test["x_align"]).double()
        za = torch.stack([torch.logsumexp(zf[:, :N_TASK], 1), zf[:, N_TASK]], 1)
        return zt, za

    def boot_rows(zt, za):
        out = []
        for it_, ia in idx:
            t = {"y_task": test["y_task"][it_], "task_hard": test["task_hard"][it_], "y_align": test["y_align"][ia], "align_hard": test["align_hard"][ia]}
            out.append(metrics_from_logits(zt[it_], za[ia], t))
        return out

    ref_boot = boot_rows(*logits(runs["reference"]["dir"] / "alignmerge.pt"))
    table = {}
    for name, r in runs.items():
        entry = {"rc_test": r["info"]["rc_test"], "failure": r["failure"], "methods_scored": r["methods"]}
        if r["row"]:
            row = r["row"]
            entry.update({
                "task_accuracy": row["task_accuracy"], "alignment_accuracy": row["alignment_accuracy"],
                "unsafe_false_negative_rate": row["unsafe_false_negative_rate"],
                "raw": raw_score(row, ref_row, c),
                "norm": sum(w * d * (row[m] - ref_row[m]) / sigma[m] for m, (w, d) in c.items() if sigma[m] > 1e-12),
                "better_worse_equal": [sum(1 for m, (w, d) in c.items() if d * (row[m] - ref_row[m]) > 1e-9),
                                       sum(1 for m, (w, d) in c.items() if d * (row[m] - ref_row[m]) < -1e-9),
                                       sum(1 for m, (w, d) in c.items() if abs(row[m] - ref_row[m]) <= 1e-9)],
            })
            ck = r["dir"] / "alignmerge.pt"
            if ck.exists() and not r["failure"] and name != "reference":
                try:
                    bt = boot_rows(*logits(ck))
                    diffs = [raw_score(b, rb, c) for b, rb in zip(bt, ref_boot)]
                    entry["raw_boot_se"] = statistics.pstdev(diffs)
                except Exception as exc:  # checkpoint not loadable -> failed run, no bootstrap
                    entry["raw_boot_se"] = None
                    entry["boot_error"] = str(exc)
        table[name] = entry

    groups = {}
    for name, e in table.items():
        gname = name.rsplit("/seed", 1)[0]
        groups.setdefault(gname, []).append(e)
    gstats = {}
    for gname, es in groups.items():
        def agg(k):
            vals = [e[k] for e in es if isinstance(e.get(k), (int, float))]
            if not vals:
                return None
            return [statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0]
        gstats[gname] = {"n": len(es), **{k: agg(k) for k in ("task_accuracy", "alignment_accuracy", "unsafe_false_negative_rate", "raw", "norm", "raw_boot_se")},
                         "failures": [e["failure"] for e in es if e.get("failure")], "rc_test": [e["rc_test"] for e in es]}
    (V2 / "summary.json").write_text(json.dumps({"reference_row": ref_row, "coefficients": c, "sigma": sigma, "runs": table, "groups": gstats}, indent=2))

    def f(x, nd=4):
        if x is None:
            return "—"
        return f"{x[0]:+.{nd}f} ± {x[1]:.{nd}f}" if isinstance(x, list) else f"{x:+.{nd}f}"

    lines = ["# Hardening runs (generated by redteam/v2/analyze.py)", "",
             "raw/norm > 0 = better than the reference row. boot SE = paired-bootstrap SE of raw(run) - raw(reference) over hidden-test prompts.", "",
             "| run | seeds | task acc | align acc | unsafe FNR | raw | norm | boot SE | test rc | failure |", "|---|---|---|---|---|---|---|---|---|---|"]
    for gname in sorted(gstats):
        s = gstats[gname]
        acc = lambda k: "—" if s[k] is None else f"{s[k][0]:.4f}"  # noqa: E731
        boot = "—" if s["raw_boot_se"] is None else f"{s['raw_boot_se'][0]:.4f}"
        lines.append(f"| {gname} | {s['n']} | {acc('task_accuracy')} | {acc('alignment_accuracy')} | {acc('unsafe_false_negative_rate')} | "
                     f"{f(s['raw'])} | {f(s['norm'], 2)} | {boot} | "
                     f"{','.join(map(str, s['rc_test']))} | {'; '.join(s['failures'])} |")
    lines += ["", "sigma used for norm:", ""] + [f"- {m}: {sigma[m]:.5g} (|coef| {c[m][0]})" for m in c]
    (V2 / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
