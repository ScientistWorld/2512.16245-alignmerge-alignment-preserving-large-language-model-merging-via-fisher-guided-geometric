"""Visible model family, loaders and simple metrics for the AlignMerge merge benchmark.

Model family: a 16-d input MLP with one output head of C+1 logits, the C=4 task answers plus a REFUSE token
(index 4), like an LM whose answer and refusal share one output layer.

- ``model(x)`` / ``model(x, "task")`` returns the C+1 logits.
- ``model(x, "align")`` returns two logits ``[answer, refuse]``, where ``answer = logsumexp`` over the task
  logits. Argmax = 1 means the model refuses the prompt.

The three checkpoints to merge live in ``data/merge_inputs/``. The labelled calibration set is
``data/calibration.pt``. It is the only data a merge method may use; it is also the visible train-evaluation slice.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

DATA_DIR = Path(__file__).resolve().parent
MERGE_INPUTS = DATA_DIR / "merge_inputs"
CALIBRATION = DATA_DIR / "calibration.pt"
SOURCE_NAMES = ("instruction_anchor", "safety_anchor", "task_expert")
N_TASK = 4
REFUSE = N_TASK


class TinyMultiTaskNet(nn.Module):
    def __init__(self, d_in: int = 16, hidden: int = 48, n_task: int = N_TASK):
        super().__init__()
        self.n_task = n_task
        self.backbone = nn.Sequential(nn.Linear(d_in, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.head = nn.Linear(hidden, n_task + 1)

    def forward(self, x: torch.Tensor, head: str = "task", return_features: bool = False):
        features = self.backbone(x)
        logits = self.head(features)
        if head == "align":
            logits = torch.stack([torch.logsumexp(logits[:, : self.n_task], dim=1), logits[:, self.n_task]], dim=1)
        if return_features:
            return logits, features
        return logits


def load_model(path: Path | str) -> TinyMultiTaskNet:
    model = TinyMultiTaskNet()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def load_merge_inputs() -> dict[str, TinyMultiTaskNet]:
    return {name: load_model(MERGE_INPUTS / f"{name}.pt") for name in SOURCE_NAMES}


def load_calibration() -> dict[str, torch.Tensor]:
    """Keys: x_task, y_task, task_hard, x_align, y_align (1 = unsafe), y_align_answer, align_hard."""
    return torch.load(CALIBRATION, map_location="cpu", weights_only=True)


def alignment_targets(y_align: torch.Tensor, y_align_answer: torch.Tensor) -> torch.Tensor:
    """Full-vocabulary targets on alignment prompts: REFUSE for unsafe prompts, the task answer for safe ones."""
    target = y_align_answer.clone()
    target[y_align == 1] = REFUSE
    return target


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == labels).float().mean().item()


def differentiable_aqi(features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Simplified Alignment Quality Index: sigmoid of a Calinski-Harabasz / Xie-Beni style between/within ratio of
    safe vs unsafe prompt features."""
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


def aqi_from_features(features: torch.Tensor, labels: torch.Tensor) -> float:
    return differentiable_aqi(features, labels).item()


def evaluate_checkpoint(model: TinyMultiTaskNet, calib: dict[str, torch.Tensor]) -> dict[str, float]:
    """Quick diagnostics on the calibration set (the official numbers come from eval/)."""
    with torch.no_grad():
        task_logits = model(calib["x_task"], "task")
        align_logits, features = model(calib["x_align"], "align", return_features=True)
    return {
        "task_accuracy": accuracy(task_logits, calib["y_task"]),
        "alignment_accuracy": accuracy(align_logits, calib["y_align"]),
        "aqi": aqi_from_features(features, calib["y_align"]),
        "unsafe_probability": F.softmax(align_logits, dim=-1)[:, 1].mean().item(),
    }
