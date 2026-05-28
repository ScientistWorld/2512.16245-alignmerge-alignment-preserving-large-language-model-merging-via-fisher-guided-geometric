from __future__ import annotations

import torch
from torch import nn


def named_trainable(model: nn.Module):
    return [(name, p) for name, p in model.named_parameters() if p.requires_grad]


def parameters_to_vector(model: nn.Module) -> torch.Tensor:
    return torch.nn.utils.parameters_to_vector([p.detach() for _, p in named_trainable(model)])


def vector_to_parameters(vec: torch.Tensor, model: nn.Module) -> None:
    torch.nn.utils.vector_to_parameters(vec, [p for _, p in named_trainable(model)])


def clone_with_vector(template: nn.Module, vec: torch.Tensor) -> nn.Module:
    clone = type(template)()
    clone.load_state_dict(template.state_dict())
    vector_to_parameters(vec.detach().clone(), clone)
    return clone


def grad_vector(model: nn.Module, loss: torch.Tensor) -> torch.Tensor:
    params = [p for _, p in named_trainable(model)]
    grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False, allow_unused=True)
    flat = []
    for param, grad in zip(params, grads):
        if grad is None:
            flat.append(torch.zeros_like(param).flatten())
        else:
            flat.append(grad.detach().flatten())
    return torch.cat(flat)
