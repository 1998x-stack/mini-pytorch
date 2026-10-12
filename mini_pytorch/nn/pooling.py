"""Pooling modules."""
from __future__ import annotations

from .module import Module
from .. import functional as F
from ..tensor import Tensor


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, input: Tensor) -> Tensor:
        return F.max_pool2d(
            input,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )

    def extra_repr(self) -> str:
        return (
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}"
        )
