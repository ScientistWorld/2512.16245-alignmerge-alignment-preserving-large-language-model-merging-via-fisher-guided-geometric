"""Hardening check variants (copied into a scratch gym as method/alignmerge/rt_variant.py and run like an agent's
own method). Every variant writes its submission as checkpoints/method/alignmerge.pt using only what an agent
has: data/merge_inputs/*.pt, data/calibration.pt and the gym's method code.

  honest, weaker:   no_align, no_budget, naive, rank6
  honest, better:   refine_calibration (AlignMerge + 300 Adam steps on the calibration labels),
                    distill_calibration (AlignMerge + 300 steps of distillation on the calibration inputs only),
                    layer_coefficients (per-tensor coefficients over {Delta_safe, Delta_expert, delta_AlignMerge}
                    fitted on the calibration set)
  brute force:      grid (2-D task-arithmetic grid chosen on calibration),
                    hparam_search (AlignMerge rank x lambda_align x gamma grid chosen on calibration)
  shortcuts:        copy_safety, copy_instruction, rowswap (hand-written rule), joint_calibration (train from
                    scratch on the calibration set), distill_synthetic (out of scope: distillation on inputs
                    sampled from a Gaussian fitted to the calibration inputs), refuse_bias (operating-point shift
                    of the REFUSE logit tuned on calibration), inf_logits, nan_params
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.func import functional_call

from data.benchmark import REFUSE, TinyMultiTaskNet, alignment_targets, load_calibration, load_merge_inputs
from method.alignmerge.merge import merge_alignmerge

N_TASK = REFUSE


def vec(m):
    return torch.cat([p.detach().reshape(-1) for p in m.parameters()])


def from_vec(v):
    m = TinyMultiTaskNet()
    torch.nn.utils.vector_to_parameters(v.detach().clone(), list(m.parameters()))
    return m


def clone(m):
    return from_vec(vec(m))


def ta(it, safe, expert, a, b):
    th = vec(it)
    return from_vec(th + a * (vec(safe) - th) + b * (vec(expert) - th))


def cal_score(m, cal):
    with torch.no_grad():
        t = (m(cal["x_task"]).argmax(1) == cal["y_task"]).float().mean().item()
        a = (m(cal["x_align"], "align").argmax(1) == cal["y_align"]).float().mean().item()
    return t + a


def refine(m, cal, steps=300, lr=1e-3, seed=0):
    m = clone(m)
    torch.manual_seed(seed)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    ya = alignment_targets(cal["y_align"], cal["y_align_answer"])
    n = len(cal["y_task"])
    for _ in range(steps):
        i, j = torch.randint(0, n, (128,)), torch.randint(0, n, (128,))
        loss = F.cross_entropy(m(cal["x_task"][i]), cal["y_task"][i]) + F.cross_entropy(m(cal["x_align"][j]), ya[j])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return m


def teacher_targets(safe, expert, xt, xa):
    with torch.no_grad():
        pt = F.softmax(expert(xt), -1)
        pe = F.softmax(expert(xa), -1)[:, :N_TASK]
        pr = F.softmax(safe(xa, "align"), -1)[:, 1:2]
        pa = torch.cat([pe / pe.sum(1, keepdim=True) * (1 - pr), pr], 1)
    return pt, pa


def distill(init, safe, expert, xt, xa, steps, lr=1e-3, seed=0, batch=256):
    pt, pa = teacher_targets(safe, expert, xt, xa)
    m = clone(init)
    torch.manual_seed(seed)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(steps):
        i, j = torch.randint(0, len(xt), (batch,)), torch.randint(0, len(xa), (batch,))
        loss = -(pt[i] * F.log_softmax(m(xt[i]), -1)).sum(1).mean() - (pa[j] * F.log_softmax(m(xa[j]), -1)).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return m


def gaussian_like(x, n, seed):
    g = torch.Generator().manual_seed(seed)
    mu = x.mean(0)
    L = torch.linalg.cholesky(torch.cov(x.T) + 1e-4 * torch.eye(x.shape[1]))
    return mu + torch.randn(n, x.shape[1], generator=g) @ L.T


def layer_coefficients(it, deltas_models, cal, init, steps=500, lr=0.05):
    names = [k for k, _ in it.named_parameters()]
    base = {k: v.detach() for k, v in it.named_parameters()}
    deltas = [{k: dict(m.named_parameters())[k].detach() - base[k] for k in names} for m in deltas_models]
    c = torch.nn.Parameter(torch.tensor(init, dtype=torch.float32))
    opt = torch.optim.Adam([c], lr=lr)
    ya = alignment_targets(cal["y_align"], cal["y_align_answer"])

    def params():
        return {k: base[k] + sum(c[s, li] * deltas[s][k] for s in range(len(deltas))) for li, k in enumerate(names)}

    for _ in range(steps):
        p = params()
        loss = F.cross_entropy(functional_call(it, p, (cal["x_task"],)), cal["y_task"]) + F.cross_entropy(
            functional_call(it, p, (cal["x_align"],)), ya)
        opt.zero_grad()
        loss.backward()
        opt.step()
    m = TinyMultiTaskNet()
    with torch.no_grad():
        m.load_state_dict({k: v.detach().clone() for k, v in params().items()})
    return m, c.detach().tolist()


def joint_from_scratch(cal, steps=3000, seed=5):
    torch.manual_seed(seed)
    m = TinyMultiTaskNet()
    return refine(m, cal, steps=steps, lr=3e-3, seed=seed)


def run(variant: str, out_dir: Path, seed: int):
    torch.manual_seed(seed)
    src = load_merge_inputs()
    cal = load_calibration()
    it, safe, expert = src["instruction_anchor"], src["safety_anchor"], src["task_expert"]
    info: dict = {"variant": variant, "seed": seed}
    am = lambda **kw: merge_alignmerge(it, safe, expert, cal, **kw)[0]  # noqa: E731
    if variant == "reference":
        sub = am()
    elif variant == "no_align":
        sub = am(lambda_align=0.0)
    elif variant == "no_budget":
        sub = am(lambda_budget=0.0)
    elif variant == "rank6":
        sub = am(rank=6)
    elif variant == "naive":
        sub = ta(it, safe, expert, 0.5, 0.5)
    elif variant == "refine_calibration":
        sub = refine(am(), cal, seed=seed)
    elif variant == "distill_calibration":
        sub = distill(am(), safe, expert, cal["x_task"], cal["x_align"], steps=300, seed=seed)
    elif variant == "layer_coefficients":
        a = am()
        sub, info["coefficients"] = layer_coefficients(it, [safe, expert, a], cal, [[0.0] * 6, [0.0] * 6, [1.0] * 6])
    elif variant == "grid":
        best = None
        for i in range(31):
            for j in range(31):
                m = ta(it, safe, expert, 0.05 * i, 0.05 * j)
                s = cal_score(m, cal)
                if best is None or s > best[0]:
                    best = (s, 0.05 * i, 0.05 * j, m)
        sub = best[3]
        info["grid_choice"] = [best[1], best[2]]
    elif variant == "hparam_search":
        best = None
        for rank in (8, 16, 32, 64, 128):
            for la in (3.0, 12.0, 48.0):
                for gamma in (1.0, 5.0):
                    m = am(rank=rank, lambda_align=la, gamma=gamma)
                    s = cal_score(m, cal)
                    if best is None or s > best[0]:
                        best = (s, dict(rank=rank, lambda_align=la, gamma=gamma), m)
        sub = best[2]
        info["hparam_choice"] = best[1]
    elif variant == "copy_safety":
        sub = clone(safe)
    elif variant == "copy_instruction":
        sub = clone(it)
    elif variant == "rowswap":
        sub = ta(it, safe, expert, 0.5, 0.5)
        sd = sub.state_dict()
        for k in ("head.weight", "head.bias"):
            sd[k][:N_TASK] = expert.state_dict()[k][:N_TASK]
            sd[k][N_TASK] = safe.state_dict()[k][N_TASK]
        sub.load_state_dict(sd)
    elif variant == "joint_calibration":
        sub = joint_from_scratch(cal, seed=seed)
    elif variant == "distill_synthetic":
        xt = torch.cat([gaussian_like(cal["x_task"], 20000, seed), cal["x_task"]])
        xa = torch.cat([gaussian_like(cal["x_align"], 20000, seed + 1), cal["x_align"]])
        sub = distill(am(), safe, expert, xt, xa, steps=3000, seed=seed)
    elif variant == "refuse_bias":
        a = am()
        best = None
        for b in torch.linspace(-3, 3, 61).tolist():
            m = clone(a)
            with torch.no_grad():
                m.head.bias[N_TASK] += b
            s = cal_score(m, cal)
            if best is None or s > best[0]:
                best = (s, b, m)
        sub = best[2]
        info["refuse_bias"] = best[1]
    elif variant == "inf_logits":
        sub = am()
        with torch.no_grad():
            sub.head.weight.mul_(1e38)
    elif variant == "nan_params":
        sub = am()
        with torch.no_grad():
            sub.head.bias[0] = float("nan")
    else:
        raise ValueError(variant)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(sub.state_dict(), out_dir / "alignmerge.pt")
    (out_dir / "variant_info.json").write_text(json.dumps(info, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True)
    p.add_argument("--out-dir", type=Path, default=Path("checkpoints/method"))
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    run(a.variant, a.out_dir, a.seed)


if __name__ == "__main__":
    main()
