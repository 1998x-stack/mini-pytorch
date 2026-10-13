"""Differentiable tensor operations for mini-pytorch v0.5."""
from __future__ import annotations

import numpy as np

from .autograd import Context, Function
from . import kernels


def unbroadcast(grad: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    grad = np.asarray(grad)
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(target_shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(target_shape)


def _numpy_root(array: np.ndarray) -> np.ndarray:
    """Return the deepest ndarray base for structural view checks.

    NumPy reports ``shares_memory=False`` for zero-element arrays because no
    bytes overlap. Empty tensors still have meaningful view/layout ownership, so
    ``View`` uses a common ndarray root as the zero-size aliasing proof.
    """

    current = array
    seen: set[int] = set()
    while isinstance(current.base, np.ndarray):
        identity = id(current)
        if identity in seen:
            break
        seen.add(identity)
        current = current.base
    return current


def _is_numpy_view_of(out: np.ndarray, source: np.ndarray) -> bool:
    if np.shares_memory(out, source):
        return True
    return out.size == 0 and source.size == 0 and _numpy_root(out) is _numpy_root(source)


class Add(Function):
    @staticmethod
    def setup_context(ctx: Context, a, b):
        ctx.a_shape, ctx.b_shape = a.shape, b.shape

    @staticmethod
    def forward(ctx: Context, a, b):
        Add.setup_context(ctx, a, b)
        return kernels.add(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output):
        return unbroadcast(grad_output, ctx.a_shape), unbroadcast(grad_output, ctx.b_shape)


class Sub(Function):
    @staticmethod
    def forward(ctx: Context, a, b):
        ctx.a_shape, ctx.b_shape = a.shape, b.shape
        return a - b

    @staticmethod
    def backward(ctx: Context, grad_output):
        return unbroadcast(grad_output, ctx.a_shape), unbroadcast(-grad_output, ctx.b_shape)


class Mul(Function):
    @staticmethod
    def setup_context(ctx: Context, a, b):
        ctx.save_for_backward(a, b)
        ctx.a_shape, ctx.b_shape = a.shape, b.shape

    @staticmethod
    def forward(ctx: Context, a, b):
        Mul.setup_context(ctx, a, b)
        return kernels.mul(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output):
        a, b = ctx.saved_tensors
        return (
            unbroadcast(grad_output * b, ctx.a_shape),
            unbroadcast(grad_output * a, ctx.b_shape),
        )


class Div(Function):
    @staticmethod
    def forward(ctx: Context, a, b):
        ctx.save_for_backward(a, b)
        ctx.a_shape, ctx.b_shape = a.shape, b.shape
        return a / b

    @staticmethod
    def backward(ctx: Context, grad_output):
        a, b = ctx.saved_tensors
        return (
            unbroadcast(grad_output / b, ctx.a_shape),
            unbroadcast(-grad_output * a / (b ** 2), ctx.b_shape),
        )


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, a):
        return -a

    @staticmethod
    def backward(ctx: Context, grad_output):
        return (-grad_output,)


class Pow(Function):
    @staticmethod
    def forward(ctx: Context, a, *, exponent: float):
        ctx.save_for_backward(a)
        ctx.exponent = exponent
        return a ** exponent

    @staticmethod
    def backward(ctx: Context, grad_output):
        (a,) = ctx.saved_tensors
        if ctx.exponent == 0:
            # x**0 is constant on the represented real-valued domain. Avoid
            # evaluating 0 * x**-1, which produces NaN at x=0 even though the
            # derivative should be exactly zero.
            return (np.zeros_like(a, dtype=np.result_type(a, grad_output)),)
        return (grad_output * ctx.exponent * (a ** (ctx.exponent - 1)),)


class MatMul(Function):
    @staticmethod
    def setup_context(ctx: Context, a, b):
        ctx.save_for_backward(a, b)

    @staticmethod
    def forward(ctx: Context, a, b):
        MatMul.setup_context(ctx, a, b)
        return kernels.matmul(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output):
        a, b = ctx.saved_tensors
        return grad_output @ b.T, a.T @ grad_output


class Sum(Function):
    @staticmethod
    def setup_context(ctx: Context, a, *, axis=None, keepdims: bool = False):
        ctx.input_shape, ctx.axis, ctx.keepdims = a.shape, axis, keepdims

    @staticmethod
    def forward(ctx: Context, a, *, axis=None, keepdims: bool = False):
        Sum.setup_context(ctx, a, axis=axis, keepdims=keepdims)
        return kernels.sum(a, axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(ctx: Context, grad_output):
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
    def forward(ctx: Context, a, *, axis=None, keepdims: bool = False):
        ctx.input_shape, ctx.axis, ctx.keepdims = a.shape, axis, keepdims
        if axis is None:
            ctx.count = a.size
        else:
            axes = axis if isinstance(axis, tuple) else (axis,)
            axes = tuple(ax if ax >= 0 else ax + a.ndim for ax in axes)
            ctx.count = int(np.prod([a.shape[ax] for ax in axes]))
        return a.mean(axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(ctx: Context, grad_output):
        grad = np.asarray(grad_output)
        if ctx.axis is None:
            return (np.ones(ctx.input_shape, dtype=grad.dtype) * grad / ctx.count,)
        axes = ctx.axis if isinstance(ctx.axis, tuple) else (ctx.axis,)
        axes = tuple(ax if ax >= 0 else ax + len(ctx.input_shape) for ax in axes)
        if not ctx.keepdims:
            for ax in sorted(axes):
                grad = np.expand_dims(grad, axis=ax)
        return (np.broadcast_to(grad, ctx.input_shape) / ctx.count,)


class View(Function):
    @staticmethod
    def forward(ctx: Context, a, *, shape: tuple[int, ...]):
        ctx.input_shape = a.shape
        out = a.reshape(shape)
        if not _is_numpy_view_of(out, a):
            raise RuntimeError(
                "view size is not compatible with the tensor's size and stride; "
                "use reshape() to allow a copy"
            )
        return out

    @staticmethod
    def backward(ctx: Context, grad_output):
        return (grad_output.reshape(ctx.input_shape),)


class Reshape(Function):
    @staticmethod
    def forward(ctx: Context, a, *, shape: tuple[int, ...]):
        ctx.input_shape = a.shape
        return a.reshape(shape)

    @staticmethod
    def backward(ctx: Context, grad_output):
        return (grad_output.reshape(ctx.input_shape),)


class Transpose(Function):
    @staticmethod
    def forward(ctx: Context, a, *, dim0: int, dim1: int):
        ndim = a.ndim
        if ndim == 0:
            raise IndexError("transpose expects at least one dimension")
        for dim in (dim0, dim1):
            if dim < -ndim or dim >= ndim:
                raise IndexError(
                    f"dimension out of range (expected to be in range of [{-ndim}, {ndim - 1}], got {dim})"
                )
        dim0 = dim0 % ndim
        dim1 = dim1 % ndim
        ctx.dim0, ctx.dim1 = dim0, dim1
        return np.swapaxes(a, dim0, dim1)

    @staticmethod
    def backward(ctx: Context, grad_output):
        return (np.swapaxes(grad_output, ctx.dim0, ctx.dim1),)


class Permute(Function):
    @staticmethod
    def forward(ctx: Context, a, *, dims: tuple[int, ...]):
        if len(dims) != a.ndim or set(dims) != set(range(a.ndim)):
            raise ValueError(f"invalid permutation {dims} for tensor with {a.ndim} dims")
        ctx.inverse = tuple(np.argsort(dims).tolist())
        return np.transpose(a, axes=dims)

    @staticmethod
    def backward(ctx: Context, grad_output):
        return (np.transpose(grad_output, axes=ctx.inverse),)


class Slice(Function):
    @staticmethod
    def forward(ctx: Context, a, *, index):
        if isinstance(index, (list, np.ndarray)):
            raise TypeError("v0.4 supports basic indexing views, not advanced indexing")
        if isinstance(index, tuple) and any(isinstance(x, (list, np.ndarray)) for x in index):
            raise TypeError("v0.4 supports basic indexing views, not advanced indexing")
        ctx.input_shape = a.shape
        ctx.index = index
        return a[index]

    @staticmethod
    def backward(ctx: Context, grad_output):
        grad = np.zeros(ctx.input_shape, dtype=np.asarray(grad_output).dtype)
        grad[ctx.index] = grad_output
        return (grad,)


class Contiguous(Function):
    @staticmethod
    def forward(ctx: Context, a):
        return np.ascontiguousarray(a).copy()

    @staticmethod
    def backward(ctx: Context, grad_output):
        return (np.asarray(grad_output),)


class ReLU(Function):
    @staticmethod
    def setup_context(ctx: Context, a):
        mask = a > 0
        ctx.save_for_backward(mask)

    @staticmethod
    def forward(ctx: Context, a):
        ReLU.setup_context(ctx, a)
        return kernels.relu(a)

    @staticmethod
    def backward(ctx: Context, grad_output):
        (mask,) = ctx.saved_tensors
        return (grad_output * mask,)


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, a):
        out = np.exp(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output):
        (out,) = ctx.saved_tensors
        return (grad_output * out,)


class Log(Function):
    @staticmethod
    def forward(ctx: Context, a):
        ctx.save_for_backward(a)
        return np.log(a)

    @staticmethod
    def backward(ctx: Context, grad_output):
        (a,) = ctx.saved_tensors
        return (grad_output / a,)
