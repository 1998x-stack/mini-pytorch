"""Loss modules."""
from __future__ import annotations

import numpy as np

from .module import Module
from ..autograd import Context, Function
from ..functional import mse_loss
from ..tensor import Tensor


class _CrossEntropy(Function):
    """Numerically stable softmax cross-entropy for class-index targets."""

    @staticmethod
    def forward(ctx: Context, logits: np.ndarray, target: np.ndarray, *, reduction: str):
        if logits.ndim != 2:
            raise ValueError("CrossEntropyLoss expects logits with shape (batch, classes)")
        if logits.shape[1] <= 0:
            raise ValueError("CrossEntropyLoss requires at least one class")

        raw_labels = np.asarray(target).reshape(-1)
        if raw_labels.shape[0] != logits.shape[0]:
            raise ValueError("target length must equal logits batch size")
        if not np.all(np.isfinite(raw_labels)):
            raise ValueError("target must contain finite integer class indices")

        labels = raw_labels.astype(np.int64)
        if not np.all(raw_labels == labels):
            raise ValueError("target must contain integer class indices")
        if np.any(labels < 0) or np.any(labels >= logits.shape[1]):
            raise ValueError("target contains an out-of-range class index")

        # log-softmax via log-sum-exp keeps the forward mathematically aligned
        # with the standard softmax-cross-entropy gradient, even for extreme
        # logits. Adding an epsilon inside log(prob + eps) would change the
        # represented objective while the backward below remained unchanged.
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_shifted = np.exp(shifted)
        normalizer = exp_shifted.sum(axis=1, keepdims=True)
        probs = exp_shifted / normalizer
        log_normalizer = np.log(normalizer.reshape(-1))
        rows = np.arange(logits.shape[0])
        losses = log_normalizer - shifted[rows, labels]

        ctx.save_for_backward(probs, labels)
        ctx.reduction = reduction
        ctx.batch_size = logits.shape[0]

        if reduction == "none":
            return losses
        if reduction == "sum":
            return np.asarray(losses.sum())
        return np.asarray(losses.mean())

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        probs, labels = ctx.saved_tensors
        grad = probs.copy()
        grad[np.arange(ctx.batch_size), labels] -= 1.0

        if ctx.reduction == "mean":
            grad /= ctx.batch_size
            grad *= np.asarray(grad_output)
        elif ctx.reduction == "sum":
            grad *= np.asarray(grad_output)
        else:
            grad *= np.asarray(grad_output).reshape(-1, 1)
        return grad, None


class MSELoss(Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("invalid reduction")
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return mse_loss(input, target, reduction=self.reduction)

    def extra_repr(self) -> str:
        return f"reduction='{self.reduction}'"


class CrossEntropyLoss(Module):
    """Softmax + negative log-likelihood for integer class targets."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("invalid reduction")
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return _CrossEntropy.apply(input, target, reduction=self.reduction)

    def extra_repr(self) -> str:
        return f"reduction='{self.reduction}'"
