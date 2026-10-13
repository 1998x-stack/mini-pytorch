"""Pure NumPy helper kernels used by the educational v0.13 AOT backward graph.

These helpers are backend numerical mechanisms, not differentiation policy.
The compiler decides *when* they appear in a backward graph; Dispatcher still
selects the concrete CPU/EDU backend implementation at execution time.
"""
from __future__ import annotations

import numpy as np


def unbroadcast(grad: np.ndarray, shape) -> np.ndarray:
    """Reduce a broadcasted gradient back to ``shape``."""

    target = tuple(int(x) for x in shape)
    out = np.asarray(grad)
    while out.ndim > len(target):
        out = out.sum(axis=0)
    if out.ndim != len(target):
        raise ValueError(f"cannot unbroadcast rank {out.ndim} gradient to shape {target}")
    for axis, (actual, wanted) in enumerate(zip(out.shape, target)):
        if wanted == 1 and actual != 1:
            out = out.sum(axis=axis, keepdims=True)
        elif wanted != actual:
            raise ValueError(
                f"cannot unbroadcast gradient shape {np.asarray(grad).shape} to {target}"
            )
    return np.asarray(out).reshape(target)


def sum_backward(grad: np.ndarray, input_shape, *, axis=None, keepdims: bool = False) -> np.ndarray:
    """Expand a reduction gradient to the original input shape."""

    shape = tuple(int(x) for x in input_shape)
    out = np.asarray(grad)
    if axis is None:
        return np.array(np.broadcast_to(out, shape), copy=True)

    if isinstance(axis, (int, np.integer)):
        axes = (int(axis),)
    else:
        axes = tuple(int(x) for x in axis)
    ndim = len(shape)
    normalized = tuple(sorted({a + ndim if a < 0 else a for a in axes}))
    if any(a < 0 or a >= ndim for a in normalized):
        raise ValueError(f"sum backward axis {axis!r} is invalid for rank {ndim}")

    if not keepdims:
        for a in normalized:
            out = np.expand_dims(out, axis=a)
    return np.array(np.broadcast_to(out, shape), copy=True)


def transpose2d(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim != 2:
        raise ValueError(f"aot transpose2d expects rank-2 input, got shape {value.shape}")
    return np.array(value.T, copy=True)


def relu_backward(grad: np.ndarray, forward_input: np.ndarray) -> np.ndarray:
    grad = np.asarray(grad)
    forward_input = np.asarray(forward_input)
    if grad.shape != forward_input.shape:
        raise ValueError(
            f"relu backward shape mismatch: grad {grad.shape}, input {forward_input.shape}"
        )
    return grad * (forward_input > 0)
