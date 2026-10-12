"""Functional API analogous to a tiny ``torch.nn.functional``."""
from __future__ import annotations

import numpy as np

from .dispatcher import call
from .tensor import Tensor


def _pair(value, *, name: str) -> tuple[int, int]:
    if isinstance(value, (int, np.integer)):
        return (int(value), int(value))
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise TypeError(f"{name} must be an int or pair of ints")


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


def conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride=1,
    padding=0,
    dilation=1,
    groups: int = 1,
    *,
    implementation: str = "reference",
) -> Tensor:
    if not isinstance(input, Tensor) or not isinstance(weight, Tensor):
        raise TypeError("conv2d input and weight must be Tensor objects")
    if weight.ndim != 4:
        raise ValueError("conv2d weight must have shape (out_channels, in_channels, kH, kW)")
    if bias is None:
        bias = Tensor(
            np.zeros(weight.shape[0], dtype=weight.dtype),
            dtype=weight.dtype,
            device=weight.device,
        )
    if not isinstance(bias, Tensor):
        raise TypeError("conv2d bias must be a Tensor or None")

    return call(
        "aten::conv2d.default",
        input,
        weight,
        bias,
        stride=_pair(stride, name="stride"),
        padding=_pair(padding, name="padding"),
        dilation=_pair(dilation, name="dilation"),
        groups=int(groups),
        implementation=str(implementation),
    )


def max_pool2d(input: Tensor, kernel_size, stride=None, padding=0) -> Tensor:
    if not isinstance(input, Tensor):
        raise TypeError("max_pool2d input must be a Tensor")
    kernel_size = _pair(kernel_size, name="kernel_size")
    stride = None if stride is None else _pair(stride, name="stride")
    padding = _pair(padding, name="padding")
    return call(
        "aten::max_pool2d.default",
        input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )


def dropout(
    input: Tensor,
    p: float = 0.5,
    training: bool = True,
    *,
    seed: int | None = None,
) -> Tensor:
    p = float(p)
    if not 0.0 <= p < 1.0:
        raise ValueError("dropout probability p must satisfy 0 <= p < 1")
    if not training or p == 0.0:
        return input
    rng = np.random.default_rng(seed)
    mask = (rng.random(input.shape) >= p).astype(input.data.dtype)
    mask /= 1.0 - p
    return input * Tensor(mask, dtype=input.dtype, device=input.device)


def batch_norm(
    input: Tensor,
    running_mean: Tensor | None,
    running_var: Tensor | None,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    *,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tensor:
    if input.ndim not in (2, 3, 4):
        raise ValueError("batch_norm expects 2-D, 3-D, or 4-D input")
    channels = input.shape[1]
    for label, tensor in (
        ("weight", weight),
        ("bias", bias),
        ("running_mean", running_mean),
        ("running_var", running_var),
    ):
        if tensor is not None and tensor.device != input.device:
            raise ValueError(f"batch_norm {label} device {tensor.device} != input device {input.device}")
    if weight is not None and weight.shape != (channels,):
        raise ValueError(f"batch_norm weight shape must be ({channels},)")
    if bias is not None and bias.shape != (channels,):
        raise ValueError(f"batch_norm bias shape must be ({channels},)")
    if running_mean is not None and running_mean.shape != (channels,):
        raise ValueError(f"running_mean shape must be ({channels},)")
    if running_var is not None and running_var.shape != (channels,):
        raise ValueError(f"running_var shape must be ({channels},)")
    if eps <= 0:
        raise ValueError("batch_norm eps must be positive")
    if not 0.0 <= momentum <= 1.0:
        raise ValueError("batch_norm momentum must be in [0, 1]")

    reduce_axes = (0,) + tuple(range(2, input.ndim))
    broadcast_shape = (1, channels) + (1,) * (input.ndim - 2)
    use_batch_stats = bool(training or running_mean is None or running_var is None)

    if use_batch_stats:
        mean = input.mean(axis=reduce_axes, keepdims=True)
        centered = input - mean
        var = (centered * centered).mean(axis=reduce_axes, keepdims=True)

        if training and running_mean is not None and running_var is not None:
            observed_mean = input.data.mean(axis=reduce_axes)
            count = int(np.prod([input.shape[axis] for axis in reduce_axes]))
            observed_var = input.data.var(axis=reduce_axes, ddof=1 if count > 1 else 0)
            running_mean.data[...] = (1.0 - momentum) * running_mean.data + momentum * observed_mean
            running_var.data[...] = (1.0 - momentum) * running_var.data + momentum * observed_var
            running_mean._mark_dirty()
            running_var._mark_dirty()
    else:
        mean = running_mean.reshape(broadcast_shape)
        var = running_var.reshape(broadcast_shape)
        centered = input - mean

    normalized = centered * ((var + eps) ** -0.5)
    if weight is not None:
        normalized = normalized * weight.reshape(broadcast_shape)
    if bias is not None:
        normalized = normalized + bias.reshape(broadcast_shape)
    return normalized
