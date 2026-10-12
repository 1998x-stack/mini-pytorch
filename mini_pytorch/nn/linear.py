"""Linear layer."""
from __future__ import annotations

import math

from .module import Module, Parameter
from ..runtime.rng import uniform
from ..tensor import Tensor


class Linear(Module):
    """Applies ``y = x A^T + b`` with PyTorch-compatible weight layout."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        self.in_features = int(in_features)
        self.out_features = int(out_features)

        bound = 1.0 / math.sqrt(in_features)
        self.weight = Parameter(
            uniform(-bound, bound, size=(out_features, in_features))
        )
        if bias:
            self.bias = Parameter(uniform(-bound, bound, size=(out_features,)))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2:
            raise ValueError("Linear currently expects input shape (batch, in_features)")
        if x.shape[1] != self.in_features:
            raise ValueError(
                f"expected {self.in_features} input features, got {x.shape[1]}"
            )
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )
