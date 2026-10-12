"""Convolution modules."""
from __future__ import annotations

import math
import numpy as np

from .module import Module, Parameter
from .. import functional as F
from ..runtime.rng import uniform
from ..tensor import Tensor


def _pair(value, *, name: str) -> tuple[int, int]:
    if isinstance(value, (int, np.integer)):
        return (int(value), int(value))
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise TypeError(f"{name} must be an int or pair of ints")


class Conv2d(Module):
    """A small PyTorch-like 2-D convolution module.

    The module owns trainable parameters; numerical execution lives in the
    functional API and Dispatcher/backend layers.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        *,
        implementation: str = "reference",
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive")
        if int(groups) != 1:
            raise NotImplementedError("mini_pytorch v0.6 Conv2d supports groups=1 only")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = _pair(kernel_size, name="kernel_size")
        self.stride = _pair(stride, name="stride")
        self.padding = _pair(padding, name="padding")
        self.dilation = _pair(dilation, name="dilation")
        self.groups = int(groups)
        self.implementation = str(implementation)
        if self.implementation not in {"reference", "im2col"}:
            raise ValueError("implementation must be 'reference' or 'im2col'")

        k_h, k_w = self.kernel_size
        bound = 1.0 / math.sqrt(self.in_channels * k_h * k_w)
        self.weight = Parameter(
            uniform(
                -bound,
                bound,
                size=(self.out_channels, self.in_channels, k_h, k_w),
            )
        )
        if bias:
            self.bias = Parameter(
                uniform(-bound, bound, size=(self.out_channels,))
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, input: Tensor) -> Tensor:
        return F.conv2d(
            input,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            implementation=self.implementation,
        )

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, "
            f"bias={self.bias is not None}, implementation={self.implementation!r}"
        )
