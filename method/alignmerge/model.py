import torch
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
