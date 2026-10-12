"""Autograd definitions for Dispatcher-backed neural-network operators."""
from __future__ import annotations

from .autograd import Context
from . import kernels


class Conv2dOp:
    """Backward definition for ``aten::conv2d.default``.

    Forward numerical work is owned by the CPU kernel.  This class stores only
    the information needed by the Autograd dispatch wrapper.
    """

    @staticmethod
    def setup_context(
        ctx: Context,
        input,
        weight,
        bias,
        *,
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        groups: int = 1,
        implementation: str = "reference",
    ) -> None:
        ctx.save_for_backward(input, weight)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = int(groups)
        ctx.implementation = implementation

    @staticmethod
    def backward(ctx: Context, grad_output):
        input, weight = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = kernels.conv2d_backward(
            input,
            weight,
            grad_output,
            stride=ctx.stride,
            padding=ctx.padding,
            dilation=ctx.dilation,
            groups=ctx.groups,
        )
        return grad_input, grad_weight, grad_bias


class MaxPool2dOp:
    """Backward definition for ``aten::max_pool2d.default``."""

    @staticmethod
    def setup_context(
        ctx: Context,
        input,
        *,
        kernel_size,
        stride=None,
        padding=0,
    ) -> None:
        ctx.save_for_backward(input)
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding

    @staticmethod
    def backward(ctx: Context, grad_output):
        (input,) = ctx.saved_tensors
        return (
            kernels.max_pool2d_backward(
                input,
                grad_output,
                kernel_size=ctx.kernel_size,
                stride=ctx.stride,
                padding=ctx.padding,
            ),
        )
