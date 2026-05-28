from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class TinyMultiTaskNet(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.task_head = nn.Linear(hidden, 2)
        self.align_head = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor, head: str = "task", return_features: bool = False):
        features = self.backbone(x)
        logits = self.task_head(features) if head == "task" else self.align_head(features)
        if return_features:
            return logits, features
        return logits


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == labels).float().mean().item()


def aqi_from_features(features: torch.Tensor, labels: torch.Tensor) -> float:
    return differentiable_aqi(features, labels).item()


def differentiable_aqi(features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    labels = labels.long()
    safe = features[labels == 0]
    unsafe = features[labels == 1]
    c0 = safe.mean(dim=0)
    c1 = unsafe.mean(dim=0)
    within = ((safe - c0).pow(2).sum(dim=1).mean() + (unsafe - c1).pow(2).sum(dim=1).mean()) / 2.0
    between = (c0 - c1).pow(2).sum()
    xb = within / (between + 1e-6)
    ch_like = between / (within + 1e-6)
    return torch.sigmoid(0.75 * torch.log(ch_like + 1e-6) - 0.75 * torch.log(xb + 1e-6))


def mean_unsafe_probability(logits: torch.Tensor, unsafe_label: int = 1) -> float:
    return F.softmax(logits, dim=-1)[:, unsafe_label].mean().item()


def evaluate_checkpoint(model: TinyMultiTaskNet, splits) -> dict[str, float]:
    with torch.no_grad():
        task_logits = model(splits.x_task_test, "task")
        align_logits, features = model(splits.x_align_test, "align", return_features=True)
    return {
        "task_accuracy": accuracy(task_logits, splits.y_task_test),
        "alignment_accuracy": accuracy(align_logits, splits.y_align_test),
        "aqi": aqi_from_features(features, splits.y_align_test),
        "unsafe_probability": mean_unsafe_probability(align_logits),
    }
