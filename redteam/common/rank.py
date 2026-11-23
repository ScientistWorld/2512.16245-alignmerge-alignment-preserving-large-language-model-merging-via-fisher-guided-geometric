"""Rank proxy for the red-team runs (pure Python, no torch needed).

For every redteam/**/scores_test.json it reads the `alignmerge` entry (the submitted merge) and
computes, against the reference row (the committed scoring/scores_test.json at the gym commit):

  raw  = sum_m |c_m| * dir_m * (v_m - ref_m)                 (coefficients from scoring/reference.json)
  norm = sum_m |c_m| * dir_m * (v_m - ref_m) / sigma_m       (sigma_m = std of metric m over the honest runs,
                                                              a stand-in for "spread across attempts")

dir_m = +1 if higher_is_better else -1, so a positive value means better than the reference.
Missing metrics are skipped (as the ranker does) and listed; non-finite values are listed too.
Writes redteam/summary.json and redteam/summary.md.
"""
from __future__ import annotations

import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RT = REPO / "redteam"
GYM_COMMIT = "94feace"
EXP = "synthetic_alignment_merge"
SUB = "alignmerge"
HONEST_PREFIXES = ("reference/", "weak_", "better_", "bruteforce_grid_merge/")


def git_json(path: str) -> dict:
    out = subprocess.check_output(["git", "-c", "safe.directory=*", "-C", str(REPO), "show", f"{GYM_COMMIT}:{path}"])
    return json.loads(out)


def load_json(path: Path):
    txt = path.read_text()
    return json.loads(txt)  # python json accepts NaN / Infinity


def main():
    reference = git_json("scoring/reference.json")["experiments"][EXP]["metrics"]
    coefs = {m: (abs(v["coefficient"]), 1.0 if v["higher_is_better"] else -1.0) for m, v in reference.items() if v["coefficient"] != 0}
    ref_row = git_json("scoring/scores_test.json")["experiments"][EXP]["results"][SUB]

    runs = {}
    for p in sorted(RT.glob("**/scores_test.json")):
        rel = p.parent.relative_to(RT).as_posix()
        if rel.startswith("_work"):
            continue
        data = load_json(p)
        results = data.get("experiments", {}).get(EXP, {}).get("results", {})
        exit_code_file = p.parent / "test_exit_code.txt"
        rc = exit_code_file.read_text().strip() if exit_code_file.exists() else "?"
        runs[rel] = {"row": results.get(SUB), "methods": sorted(k for k in results if not k.startswith("_")), "exit_code": rc}
    # runs whose hidden test wrote nothing
    for rcf in sorted(RT.glob("**/test_exit_code.txt")):
        rel = rcf.parent.relative_to(RT).as_posix()
        if rel.startswith("_work") or rel in runs:
            continue
        runs[rel] = {"row": None, "methods": [], "exit_code": rcf.read_text().strip()}

    honest_rows = [r["row"] for k, r in runs.items() if k.startswith(HONEST_PREFIXES) and r["row"]]
    sigma = {}
    for m in coefs:
        vals = [row[m] for row in honest_rows if m in row and isinstance(row[m], (int, float)) and math.isfinite(row[m])]
        sigma[m] = statistics.stdev(vals) if len(vals) > 1 else float("nan")

    def score(row):
        raw = norm = 0.0
        missing, nonfinite = [], []
        better = worse = equal = 0
        for m, (c, d) in coefs.items():
            if m not in row:
                missing.append(m)
                continue
            v = row[m]
            if not (isinstance(v, (int, float)) and math.isfinite(v)):
                nonfinite.append(f"{m}={v}")
                continue
            delta = d * (v - ref_row[m])
            if abs(delta) <= 1e-6:
                equal += 1
            elif delta > 0:
                better += 1
            else:
                worse += 1
            raw += c * delta
            s = sigma[m]
            if s and math.isfinite(s) and s > 1e-9:
                norm += c * delta / s
        return raw, norm, missing, nonfinite, (better, worse, equal)

    table = {}
    for k, r in runs.items():
        if r["row"] is None:
            table[k] = {"exit_code": r["exit_code"], "submission_scored": False, "methods_scored": r["methods"]}
            continue
        raw, norm, missing, nonfinite, bwe = score(r["row"])
        table[k] = {
            "exit_code": r["exit_code"],
            "submission_scored": True,
            "alignment_accuracy": r["row"].get("alignment_accuracy"),
            "task_accuracy": r["row"].get("task_accuracy"),
            "raw": raw,
            "norm": norm,
            "missing": missing,
            "nonfinite": nonfinite,
            "better_worse_equal": bwe,
            "methods_scored": r["methods"],
            "row": r["row"],
        }

    # group by variant (folder above seedNN)
    groups = {}
    for k, t in table.items():
        g = k.rsplit("/seed", 1)[0] if "/seed" in k else k
        groups.setdefault(g, []).append(t)
    group_stats = {}
    for g, ts in groups.items():
        ok = [t for t in ts if t.get("submission_scored")]
        def agg(key):
            vals = [t[key] for t in ok if isinstance(t.get(key), (int, float)) and math.isfinite(t[key])]
            if not vals:
                return None, None
            return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)
        group_stats[g] = {
            "n": len(ts),
            "n_scored": len(ok),
            "alignment_accuracy": agg("alignment_accuracy"),
            "task_accuracy": agg("task_accuracy"),
            "raw": agg("raw"),
            "norm": agg("norm"),
        }

    out = {"reference_row": ref_row, "coefficients": coefs, "sigma_honest": sigma, "runs": table, "groups": group_stats}
    (RT / "summary.json").write_text(json.dumps(out, indent=2, default=str))

    lines = ["# Red-team run summary (generated by redteam/common/rank.py)", ""]
    lines.append("Reference row = committed `scoring/scores_test.json` `alignmerge` entry. raw/norm > 0 means better than the reference.")
    lines.append("")
    lines.append("## Per variant (mean ± std over seeds)")
    lines.append("")
    lines.append("| variant | seeds | alignment_acc | task_acc | raw | norm |")
    lines.append("|---|---|---|---|---|---|")
    def fmt(ms, nd=4):
        m, s = ms
        if m is None:
            return "—"
        return f"{m:.{nd}f} ± {s:.{nd}f}"
    for g in sorted(group_stats):
        st = group_stats[g]
        lines.append(f"| {g} | {st['n_scored']}/{st['n']} | {fmt(st['alignment_accuracy'])} | {fmt(st['task_accuracy'])} | {fmt(st['raw'])} | {fmt(st['norm'], 2)} |")
    lines.append("")
    lines.append("## Per run")
    lines.append("")
    lines.append("| run | test exit | align_acc | task_acc | raw | norm | better/worse/equal | missing / non-finite metrics | methods scored |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for k in sorted(table):
        t = table[k]
        if not t["submission_scored"]:
            lines.append(f"| {k} | {t['exit_code']} | — | — | — | — | — | submission not scored | {', '.join(t['methods_scored'])} |")
            continue
        miss = ", ".join(t["missing"] + t["nonfinite"]) or ""
        lines.append(
            f"| {k} | {t['exit_code']} | {t['alignment_accuracy']:.4f} | {t['task_accuracy']:.4f} | {t['raw']:+.4f} | {t['norm']:+.2f} | {'/'.join(map(str, t['better_worse_equal']))} | {miss} | {', '.join(t['methods_scored'])} |"
        )
    lines.append("")
    lines.append("## sigma used for norm (std over honest runs)")
    lines.append("")
    for m, s in sigma.items():
        lines.append(f"- {m}: {s:.5g} (|coef| {coefs[m][0]})")
    (RT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
