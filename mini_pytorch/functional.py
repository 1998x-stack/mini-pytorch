"""Functional API analogous to a tiny ``torch.nn.functional``."""
from __future__ import annotations

from .tensor import Tensor


def relu(x: Tensor) -> Tensor:
    return x.relu()


def sigmoid(x: Tensor) -> Tensor:
    return x.sigmoid()


def mse_loss(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    diff = input - target
    loss = diff * diff
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError("reduction must be one of: 'mean', 'sum', 'none'")
