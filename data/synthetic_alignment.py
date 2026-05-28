import math
from dataclasses import dataclass

import torch


@dataclass
class SyntheticSplits:
    x_task_train: torch.Tensor
    y_task_train: torch.Tensor
    x_align_train: torch.Tensor
    y_align_train: torch.Tensor
    x_task_test: torch.Tensor
    y_task_test: torch.Tensor
    x_align_test: torch.Tensor
    y_align_test: torch.Tensor


def _make_task(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 2, generator=gen)
    radius = torch.linalg.norm(x, dim=1)
    angle = torch.atan2(x[:, 1], x[:, 0])
    y = ((angle > -math.pi / 3) & (angle < 2 * math.pi / 3)).long()
    y ^= (radius > 1.45).long()
    flip = torch.rand(n, generator=gen) < 0.04
    y[flip] = 1 - y[flip]
    return x, y


def _make_alignment(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    half = n // 2
    safe = torch.randn(half, 2, generator=gen) * torch.tensor([0.55, 0.7]) + torch.tensor([-1.0, -0.35])
    unsafe = torch.randn(n - half, 2, generator=gen) * torch.tensor([0.55, 0.7]) + torch.tensor([1.0, 0.35])
    # Curved overlap makes the problem non-trivial but learnable.
    safe[:, 1] += 0.35 * torch.sin(2.0 * safe[:, 0])
    unsafe[:, 1] += 0.35 * torch.sin(2.0 * unsafe[:, 0])
    x = torch.cat([safe, unsafe], dim=0)
    y = torch.cat([torch.zeros(half), torch.ones(n - half)]).long()
    perm = torch.randperm(n, generator=gen)
    return x[perm], y[perm]


def load_synthetic_splits(seed: int = 13, n_train: int = 2048, n_test: int = 1024) -> SyntheticSplits:
    x_task_train, y_task_train = _make_task(n_train, seed)
    x_align_train, y_align_train = _make_alignment(n_train, seed + 1)
    x_task_test, y_task_test = _make_task(n_test, seed + 2)
    x_align_test, y_align_test = _make_alignment(n_test, seed + 3)
    return SyntheticSplits(
        x_task_train=x_task_train,
        y_task_train=y_task_train,
        x_align_train=x_align_train,
        y_align_train=y_align_train,
        x_task_test=x_task_test,
        y_task_test=y_task_test,
        x_align_test=x_align_test,
        y_align_test=y_align_test,
    )
