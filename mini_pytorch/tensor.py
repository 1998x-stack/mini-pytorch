"""Tensor object: data + autograd metadata + operator syntax."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .autograd import GradFn


_DEFAULT_DTYPE = np.float64


def _as_array(data: Any, dtype=None) -> np.ndarray:
    if isinstance(data, np.ndarray):
        if dtype is not None:
            return data.astype(dtype, copy=False)
        if np.issubdtype(data.dtype, np.floating):
            return data
        return data.astype(_DEFAULT_DTYPE)
    return np.asarray(data, dtype=dtype or _DEFAULT_DTYPE)


class Tensor:
    """A tiny differentiable n-dimensional array.

    The implementation deliberately exposes core concepts rather than hiding
    them behind a C++ backend. Gradients are stored as NumPy arrays to keep the
    engine small and auditable.
    """

    __array_priority__ = 1000

    def __init__(
        self,
        data: Any,
        *,
        requires_grad: bool = False,
        dtype=None,
        name: Optional[str] = None,
    ) -> None:
        self.data = _as_array(data, dtype=dtype)
        self.requires_grad = bool(requires_grad)
        self.grad: Optional[np.ndarray] = None
        self._grad_fn: Optional[GradFn] = None
        self.name = name

    # ---------- metadata ----------
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def grad_fn(self) -> Optional[str]:
        return None if self._grad_fn is None else self._grad_fn.name

    @property
    def is_leaf(self) -> bool:
        return self.requires_grad and self._grad_fn is None

    def numel(self) -> int:
        return self.data.size

    # ---------- graph / grad ----------
    def backward(self, grad: Optional[Any] = None) -> None:
        """Reverse-mode automatic differentiation from this Tensor.

        If ``grad`` is omitted, the tensor must contain exactly one element.
        This matches the important PyTorch rule that non-scalar outputs require
        an explicit vector-Jacobian product seed.
        """
        if not self.requires_grad:
            raise RuntimeError("cannot call backward() on a tensor that does not require grad")

        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("grad must be specified for non-scalar outputs")
            seed = np.ones_like(self.data)
        else:
            seed = _as_array(grad, dtype=self.data.dtype)
            if seed.shape != self.data.shape:
                raise ValueError(f"gradient shape {seed.shape} != tensor shape {self.data.shape}")

        topo: list[Tensor] = []
        visited: set[int] = set()

        def build(node: Tensor) -> None:
            node_id = id(node)
            if node_id in visited:
                return
            visited.add(node_id)
            if node._grad_fn is not None:
                for parent in node._grad_fn.parents:
                    build(parent)
            topo.append(node)

        build(self)

        # Intermediate grads are local to this backward traversal. Leaf grads
        # accumulate across backward calls, like PyTorch, unless zeroed.
        local_grads: dict[int, np.ndarray] = {id(self): seed}

        for node in reversed(topo):
            node_grad = local_grads.get(id(node))
            if node_grad is None:
                continue

            if node.requires_grad:
                if node.is_leaf:
                    node.grad = node_grad.copy() if node.grad is None else node.grad + node_grad
                else:
                    # Retaining non-leaf grads is omitted in this MVP; backward
                    # still propagates through them using local_grads.
                    node.grad = None

            if node._grad_fn is None:
                continue

            parent_grads = node._grad_fn.function.backward(node._grad_fn.ctx, node_grad)
            if not isinstance(parent_grads, tuple):
                parent_grads = (parent_grads,)
            if len(parent_grads) != len(node._grad_fn.parents):
                raise RuntimeError(
                    f"{node._grad_fn.function.__name__}.backward returned "
                    f"{len(parent_grads)} grads for {len(node._grad_fn.parents)} parents"
                )

            for parent, parent_grad in zip(node._grad_fn.parents, parent_grads):
                if parent_grad is None or not parent.requires_grad:
                    continue
                parent_grad = np.asarray(parent_grad, dtype=parent.data.dtype)
                key = id(parent)
                local_grads[key] = (
                    parent_grad if key not in local_grads else local_grads[key] + parent_grad
                )

    def zero_grad(self) -> None:
        self.grad = None

    def detach(self) -> "Tensor":
        """Return a graph-free tensor sharing the same NumPy storage."""
        return Tensor(self.data, requires_grad=False, dtype=self.data.dtype)

    def numpy(self) -> np.ndarray:
        return self.data

    def item(self):
        if self.data.size != 1:
            raise ValueError("only one-element tensors can be converted to Python scalars")
        return self.data.item()

    # ---------- type coercion ----------
    @staticmethod
    def _ensure_tensor(value: Any) -> "Tensor":
        return value if isinstance(value, Tensor) else Tensor(value)

    # ---------- arithmetic ----------
    def __add__(self, other):
        from .ops import Add
        return Add.apply(self, self._ensure_tensor(other))

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        from .ops import Sub
        return Sub.apply(self, self._ensure_tensor(other))

    def __rsub__(self, other):
        return self._ensure_tensor(other) - self

    def __mul__(self, other):
        from .ops import Mul
        return Mul.apply(self, self._ensure_tensor(other))

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        from .ops import Div
        return Div.apply(self, self._ensure_tensor(other))

    def __rtruediv__(self, other):
        return self._ensure_tensor(other) / self

    def __neg__(self):
        from .ops import Neg
        return Neg.apply(self)

    def __pow__(self, exponent: float):
        from .ops import Pow
        return Pow.apply(self, exponent=float(exponent))

    def __matmul__(self, other):
        from .ops import MatMul
        return MatMul.apply(self, self._ensure_tensor(other))

    # ---------- shape / reductions ----------
    def sum(self, axis=None, keepdims: bool = False):
        from .ops import Sum
        return Sum.apply(self, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims: bool = False):
        from .ops import Mean
        return Mean.apply(self, axis=axis, keepdims=keepdims)

    def reshape(self, *shape: int):
        from .ops import Reshape
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return Reshape.apply(self, shape=tuple(shape))

    @property
    def T(self):
        from .ops import Transpose2D
        return Transpose2D.apply(self)

    # ---------- nonlinearities ----------
    def relu(self):
        from .ops import ReLU
        return ReLU.apply(self)

    def exp(self):
        from .ops import Exp
        return Exp.apply(self)

    def log(self):
        from .ops import Log
        return Log.apply(self)

    def sigmoid(self):
        return 1.0 / (1.0 + (-self).exp())

    def __repr__(self) -> str:
        meta = f", requires_grad={self.requires_grad}"
        if self.grad_fn is not None:
            meta += f", grad_fn=<{self.grad_fn}>"
        return f"Tensor({self.data!r}{meta})"
