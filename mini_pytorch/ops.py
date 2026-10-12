"""Differentiable tensor operations used by :mod:`mini_pytorch.tensor`."""
from __future__ import annotations

import numpy as np

from .autograd import Context, Function


def unbroadcast(grad: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """Sum a broadcasted gradient back to the original operand shape.

    NumPy broadcasting can expand dimensions implicitly during forward. During
    backward, the inverse operation is summation over those expanded axes.
    """
    grad = np.asarray(grad)
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(target_shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(target_shape)


class Add(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ctx.a_shape, ctx.b_shape = a.shape, b.shape
        return a + b

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        return (
            unbroadcast(grad_output, ctx.a_shape),
            unbroadcast(grad_output, ctx.b_shape),
        )


class Sub(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ctx.a_shape, ctx.b_shape = a.shape, b.shape
        return a - b

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        return (
            unbroadcast(grad_output, ctx.a_shape),
            unbroadcast(-grad_output, ctx.b_shape),
        )


class Mul(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ctx.save_for_backward(a, b)
        ctx.a_shape, ctx.b_shape = a.shape, b.shape
        return a * b

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        a, b = ctx.saved_tensors
        return (
            unbroadcast(grad_output * b, ctx.a_shape),
            unbroadcast(grad_output * a, ctx.b_shape),
        )


class Div(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ctx.save_for_backward(a, b)
        ctx.a_shape, ctx.b_shape = a.shape, b.shape
        return a / b

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        a, b = ctx.saved_tensors
        return (
            unbroadcast(grad_output / b, ctx.a_shape),
            unbroadcast(-grad_output * a / (b ** 2), ctx.b_shape),
        )


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray) -> np.ndarray:
        return -a

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        return (-grad_output,)


class Pow(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, *, exponent: float) -> np.ndarray:
        ctx.save_for_backward(a)
        ctx.exponent = exponent
        return a ** exponent

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        (a,) = ctx.saved_tensors
        p = ctx.exponent
        return (grad_output * p * (a ** (p - 1)),)


class MatMul(Function):
    """2-D matrix multiplication.

    Keeping v1 restricted to matrices makes its derivative easy to inspect.
    Batched matmul can later be added without changing the public Tensor API.
    """

    @staticmethod
    def forward(ctx: Context, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError("mini_pytorch MatMul currently supports 2-D tensors only")
        if a.shape[1] != b.shape[0]:
            raise ValueError(f"matmul shape mismatch: {a.shape} @ {b.shape}")
        ctx.save_for_backward(a, b)
        return a @ b

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        a, b = ctx.saved_tensors
        return grad_output @ b.T, a.T @ grad_output


class Sum(Function):
    @staticmethod
    def forward(
        ctx: Context,
        a: np.ndarray,
        *,
        axis=None,
        keepdims: bool = False,
    ) -> np.ndarray:
        ctx.input_shape = a.shape
        ctx.axis = axis
        ctx.keepdims = keepdims
        return a.sum(axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        grad = np.asarray(grad_output)
        if ctx.axis is None:
            return (np.ones(ctx.input_shape, dtype=grad.dtype) * grad,)

        axes = ctx.axis if isinstance(ctx.axis, tuple) else (ctx.axis,)
        axes = tuple(ax if ax >= 0 else ax + len(ctx.input_shape) for ax in axes)
        if not ctx.keepdims:
            for ax in sorted(axes):
                grad = np.expand_dims(grad, axis=ax)
        return (np.broadcast_to(grad, ctx.input_shape),)


class Mean(Function):
    @staticmethod
    def forward(
        ctx: Context,
        a: np.ndarray,
        *,
        axis=None,
        keepdims: bool = False,
    ) -> np.ndarray:
        ctx.input_shape = a.shape
        ctx.axis = axis
        ctx.keepdims = keepdims
        if axis is None:
            ctx.count = a.size
        else:
            axes = axis if isinstance(axis, tuple) else (axis,)
            axes = tuple(ax if ax >= 0 else ax + a.ndim for ax in axes)
            ctx.count = int(np.prod([a.shape[ax] for ax in axes]))
        return a.mean(axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        grad = np.asarray(grad_output)
        if ctx.axis is None:
            return (np.ones(ctx.input_shape, dtype=grad.dtype) * grad / ctx.count,)

        axes = ctx.axis if isinstance(ctx.axis, tuple) else (ctx.axis,)
        axes = tuple(ax if ax >= 0 else ax + len(ctx.input_shape) for ax in axes)
        if not ctx.keepdims:
            for ax in sorted(axes):
                grad = np.expand_dims(grad, axis=ax)
        return (np.broadcast_to(grad, ctx.input_shape) / ctx.count,)


class Reshape(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, *, shape: tuple[int, ...]) -> np.ndarray:
        ctx.input_shape = a.shape
        return a.reshape(shape)

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        return (grad_output.reshape(ctx.input_shape),)


class Transpose2D(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray) -> np.ndarray:
        if a.ndim != 2:
            raise ValueError(".T currently supports 2-D tensors only")
        return a.T

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        return (grad_output.T,)


class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray) -> np.ndarray:
        mask = a > 0
        ctx.save_for_backward(mask)
        return np.maximum(a, 0)

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        (mask,) = ctx.saved_tensors
        return (grad_output * mask,)


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray) -> np.ndarray:
        out = np.exp(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        (out,) = ctx.saved_tensors
        return (grad_output * out,)


class Log(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray) -> np.ndarray:
        ctx.save_for_backward(a)
        return np.log(a)

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        (a,) = ctx.saved_tensors
        return (grad_output / a,)
