"""Pure NumPy backend kernels.

The backend layer intentionally knows nothing about Autograd, Module, Optimizer,
or DataLoader.  v0.6 adds readable CNN kernels while keeping the v0.5
Dispatcher boundary intact.
"""
from __future__ import annotations

import numpy as np


def _pair(value, *, name: str) -> tuple[int, int]:
    if isinstance(value, (int, np.integer)):
        result = (int(value), int(value))
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        result = (int(value[0]), int(value[1]))
    else:
        raise TypeError(f"{name} must be an int or pair of ints")
    return result


def add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a + b


def mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a * b


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("mini_pytorch MatMul currently supports 2-D tensors only")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"matmul shape mismatch: {a.shape} @ {b.shape}")
    return a @ b


def relu(a: np.ndarray) -> np.ndarray:
    return np.maximum(a, 0)


def sum(a: np.ndarray, *, axis=None, keepdims: bool = False) -> np.ndarray:
    return np.asarray(a.sum(axis=axis, keepdims=keepdims))


# ---------------------------------------------------------------------------
# Conv2d
# ---------------------------------------------------------------------------


def _validate_conv2d(
    input: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None,
    *,
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    groups: int = 1,
) -> tuple[int, int, int, int]:
    if input.ndim != 4:
        raise ValueError(f"conv2d expects NCHW input, got shape {input.shape}")
    if weight.ndim != 4:
        raise ValueError(f"conv2d expects OIHW weight, got shape {weight.shape}")
    if int(groups) != 1:
        raise NotImplementedError("mini_pytorch v0.6 Conv2d supports groups=1 only")

    n, c_in, h_in, w_in = input.shape
    c_out, weight_c_in, k_h, k_w = weight.shape
    if c_in != weight_c_in:
        raise ValueError(
            f"conv2d channel mismatch: input has {c_in} channels, "
            f"weight expects {weight_c_in}"
        )
    if bias is not None and np.asarray(bias).shape != (c_out,):
        raise ValueError(f"conv2d bias shape must be ({c_out},), got {np.asarray(bias).shape}")

    s_h, s_w = _pair(stride, name="stride")
    p_h, p_w = _pair(padding, name="padding")
    d_h, d_w = _pair(dilation, name="dilation")
    if min(s_h, s_w) <= 0:
        raise ValueError("conv2d stride values must be positive")
    if min(d_h, d_w) <= 0:
        raise ValueError("conv2d dilation values must be positive")
    if min(p_h, p_w) < 0:
        raise ValueError("conv2d padding values must be non-negative")
    if min(k_h, k_w) <= 0:
        raise ValueError("conv2d kernel dimensions must be positive")

    h_out = (h_in + 2 * p_h - d_h * (k_h - 1) - 1) // s_h + 1
    w_out = (w_in + 2 * p_w - d_w * (k_w - 1) - 1) // s_w + 1
    if h_out <= 0 or w_out <= 0:
        raise ValueError(
            "conv2d calculated a non-positive output size; check input, kernel, "
            "stride, padding and dilation"
        )
    return n, c_out, h_out, w_out


def conv2d_reference(
    input: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
    *,
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    groups: int = 1,
) -> np.ndarray:
    """Readable NCHW/OIHW cross-correlation reference implementation."""

    n, c_out, h_out, w_out = _validate_conv2d(
        input,
        weight,
        bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    _, c_in, h_in, w_in = input.shape
    _, _, k_h, k_w = weight.shape
    s_h, s_w = _pair(stride, name="stride")
    p_h, p_w = _pair(padding, name="padding")
    d_h, d_w = _pair(dilation, name="dilation")

    padded = np.pad(input, ((0, 0), (0, 0), (p_h, p_h), (p_w, p_w)))
    dtype = np.result_type(input.dtype, weight.dtype, None if bias is None else bias.dtype)
    out = np.empty((n, c_out, h_out, w_out), dtype=dtype)

    for batch in range(n):
        for out_channel in range(c_out):
            for out_h in range(h_out):
                h0 = out_h * s_h
                for out_w in range(w_out):
                    w0 = out_w * s_w
                    total = 0.0 if bias is None else bias[out_channel]
                    for in_channel in range(c_in):
                        for kernel_h in range(k_h):
                            in_h = h0 + kernel_h * d_h
                            for kernel_w in range(k_w):
                                in_w = w0 + kernel_w * d_w
                                total += (
                                    padded[batch, in_channel, in_h, in_w]
                                    * weight[out_channel, in_channel, kernel_h, kernel_w]
                                )
                    out[batch, out_channel, out_h, out_w] = total
    return out


def im2col2d(
    input: np.ndarray,
    kernel_size,
    *,
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
) -> tuple[np.ndarray, int, int]:
    """Lower NCHW convolution windows into rows for matrix multiplication."""

    if input.ndim != 4:
        raise ValueError("im2col2d expects NCHW input")
    k_h, k_w = _pair(kernel_size, name="kernel_size")
    _, c_in, _, _ = input.shape
    dummy_weight = np.empty((1, c_in, k_h, k_w), dtype=input.dtype)
    n, _, h_out, w_out = _validate_conv2d(
        input,
        dummy_weight,
        None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=1,
    )
    s_h, s_w = _pair(stride, name="stride")
    p_h, p_w = _pair(padding, name="padding")
    d_h, d_w = _pair(dilation, name="dilation")
    padded = np.pad(input, ((0, 0), (0, 0), (p_h, p_h), (p_w, p_w)))

    columns = np.empty((n * h_out * w_out, c_in * k_h * k_w), dtype=input.dtype)
    row = 0
    for batch in range(n):
        for out_h in range(h_out):
            h0 = out_h * s_h
            for out_w in range(w_out):
                w0 = out_w * s_w
                cursor = 0
                for in_channel in range(c_in):
                    for kernel_h in range(k_h):
                        in_h = h0 + kernel_h * d_h
                        for kernel_w in range(k_w):
                            in_w = w0 + kernel_w * d_w
                            columns[row, cursor] = padded[batch, in_channel, in_h, in_w]
                            cursor += 1
                row += 1
    return columns, h_out, w_out


def conv2d_im2col(
    input: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
    *,
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    groups: int = 1,
) -> np.ndarray:
    """Educational optimized path: im2col followed by GEMM."""

    n, c_out, h_out, w_out = _validate_conv2d(
        input,
        weight,
        bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    columns, check_h, check_w = im2col2d(
        input,
        weight.shape[2:],
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    assert (check_h, check_w) == (h_out, w_out)
    matrix = columns @ weight.reshape(c_out, -1).T
    if bias is not None:
        matrix = matrix + bias.reshape(1, c_out)
    return matrix.reshape(n, h_out, w_out, c_out).transpose(0, 3, 1, 2)


def conv2d_backward(
    input: np.ndarray,
    weight: np.ndarray,
    grad_output: np.ndarray,
    *,
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    groups: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference backward for input, weight and bias gradients."""

    n, c_out, h_out, w_out = _validate_conv2d(
        input,
        weight,
        None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    if grad_output.shape != (n, c_out, h_out, w_out):
        raise ValueError(
            f"conv2d grad_output shape mismatch: expected {(n, c_out, h_out, w_out)}, "
            f"got {grad_output.shape}"
        )

    _, c_in, h_in, w_in = input.shape
    _, _, k_h, k_w = weight.shape
    s_h, s_w = _pair(stride, name="stride")
    p_h, p_w = _pair(padding, name="padding")
    d_h, d_w = _pair(dilation, name="dilation")
    padded = np.pad(input, ((0, 0), (0, 0), (p_h, p_h), (p_w, p_w)))
    dtype = np.result_type(input.dtype, weight.dtype, grad_output.dtype)
    grad_padded = np.zeros_like(padded, dtype=dtype)
    grad_weight = np.zeros_like(weight, dtype=dtype)
    grad_bias = grad_output.sum(axis=(0, 2, 3))

    for batch in range(n):
        for out_channel in range(c_out):
            for out_h in range(h_out):
                h0 = out_h * s_h
                for out_w in range(w_out):
                    w0 = out_w * s_w
                    grad = grad_output[batch, out_channel, out_h, out_w]
                    for in_channel in range(c_in):
                        for kernel_h in range(k_h):
                            in_h = h0 + kernel_h * d_h
                            for kernel_w in range(k_w):
                                in_w = w0 + kernel_w * d_w
                                grad_weight[out_channel, in_channel, kernel_h, kernel_w] += (
                                    grad * padded[batch, in_channel, in_h, in_w]
                                )
                                grad_padded[batch, in_channel, in_h, in_w] += (
                                    grad * weight[out_channel, in_channel, kernel_h, kernel_w]
                                )

    if p_h or p_w:
        grad_input = grad_padded[:, :, p_h : p_h + h_in, p_w : p_w + w_in]
    else:
        grad_input = grad_padded
    return grad_input, grad_weight, grad_bias


# ---------------------------------------------------------------------------
# MaxPool2d
# ---------------------------------------------------------------------------


def _validate_max_pool2d(input: np.ndarray, *, kernel_size, stride=None, padding=0):
    if input.ndim != 4:
        raise ValueError(f"max_pool2d expects NCHW input, got shape {input.shape}")
    k_h, k_w = _pair(kernel_size, name="kernel_size")
    s_h, s_w = _pair(kernel_size if stride is None else stride, name="stride")
    p_h, p_w = _pair(padding, name="padding")
    if min(k_h, k_w, s_h, s_w) <= 0:
        raise ValueError("max_pool2d kernel_size and stride must be positive")
    if min(p_h, p_w) < 0:
        raise ValueError("max_pool2d padding must be non-negative")
    n, channels, h_in, w_in = input.shape
    h_out = (h_in + 2 * p_h - k_h) // s_h + 1
    w_out = (w_in + 2 * p_w - k_w) // s_w + 1
    if h_out <= 0 or w_out <= 0:
        raise ValueError("max_pool2d calculated a non-positive output size")
    return (n, channels, h_out, w_out), (k_h, k_w), (s_h, s_w), (p_h, p_w)


def max_pool2d(
    input: np.ndarray,
    *,
    kernel_size,
    stride=None,
    padding=0,
) -> np.ndarray:
    shape, (k_h, k_w), (s_h, s_w), (p_h, p_w) = _validate_max_pool2d(
        input, kernel_size=kernel_size, stride=stride, padding=padding
    )
    n, channels, h_out, w_out = shape
    padded = np.pad(
        input,
        ((0, 0), (0, 0), (p_h, p_h), (p_w, p_w)),
        mode="constant",
        constant_values=-np.inf,
    )
    out = np.empty(shape, dtype=input.dtype)
    for batch in range(n):
        for channel in range(channels):
            for out_h in range(h_out):
                h0 = out_h * s_h
                for out_w in range(w_out):
                    w0 = out_w * s_w
                    window = padded[batch, channel, h0 : h0 + k_h, w0 : w0 + k_w]
                    out[batch, channel, out_h, out_w] = np.max(window)
    return out


def max_pool2d_backward(
    input: np.ndarray,
    grad_output: np.ndarray,
    *,
    kernel_size,
    stride=None,
    padding=0,
) -> np.ndarray:
    shape, (k_h, k_w), (s_h, s_w), (p_h, p_w) = _validate_max_pool2d(
        input, kernel_size=kernel_size, stride=stride, padding=padding
    )
    if grad_output.shape != shape:
        raise ValueError(f"max_pool2d grad_output shape mismatch: expected {shape}, got {grad_output.shape}")

    n, channels, h_out, w_out = shape
    _, _, h_in, w_in = input.shape
    padded = np.pad(
        input,
        ((0, 0), (0, 0), (p_h, p_h), (p_w, p_w)),
        mode="constant",
        constant_values=-np.inf,
    )
    grad_padded = np.zeros_like(padded, dtype=np.result_type(input.dtype, grad_output.dtype))

    for batch in range(n):
        for channel in range(channels):
            for out_h in range(h_out):
                h0 = out_h * s_h
                for out_w in range(w_out):
                    w0 = out_w * s_w
                    window = padded[batch, channel, h0 : h0 + k_h, w0 : w0 + k_w]
                    flat_index = int(np.argmax(window))
                    win_h, win_w = np.unravel_index(flat_index, window.shape)
                    grad_padded[batch, channel, h0 + win_h, w0 + win_w] += grad_output[
                        batch, channel, out_h, out_w
                    ]

    if p_h or p_w:
        return grad_padded[:, :, p_h : p_h + h_in, p_w : p_w + w_in]
    return grad_padded
