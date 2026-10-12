"""Tensor object: Storage-backed layout metadata + autograd + operator syntax."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .autograd import GradFn, is_grad_enabled
from .storage import Storage, contiguous_strides


_DEFAULT_DTYPE = np.float64


def _as_array(data: Any, dtype=None) -> np.ndarray:
    if isinstance(data, np.ndarray):
        if dtype is not None:
            return data.astype(dtype, copy=False)
        if np.issubdtype(data.dtype, np.floating):
            return data
        return data.astype(_DEFAULT_DTYPE)
    return np.asarray(data, dtype=dtype or _DEFAULT_DTYPE)


def _normalize_shape(shape: tuple[int, ...], numel: int) -> tuple[int, ...]:
    dims = [int(dim) for dim in shape]
    if dims.count(-1) > 1:
        raise ValueError("only one dimension can be inferred")
    if any(dim < -1 for dim in dims):
        raise ValueError("shape dimensions must be >= -1")
    if -1 in dims:
        known = 1
        for dim in dims:
            if dim != -1:
                known *= dim
        if known == 0 or numel % known != 0:
            raise ValueError(f"cannot infer shape {shape} for {numel} elements")
        dims[dims.index(-1)] = numel // known
    total = 1
    for dim in dims:
        total *= dim
    if total != numel:
        raise ValueError(f"shape {tuple(dims)} is invalid for {numel} elements")
    return tuple(dims)


class Tensor:
    """A tiny differentiable Tensor with explicit Storage/layout semantics.

    v0.4 makes the PyTorch-like mental model visible:
    Storage + shape + stride + storage_offset describe a Tensor view.
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
        array = _as_array(data, dtype=dtype)
        # np.ascontiguousarray() promotes a 0-D ndarray to shape (1,), which
        # would silently change scalar Tensor semantics. np.array(..., order="C")
        # preserves the original dimensionality while still materializing a
        # canonical contiguous allocation for Storage.
        contiguous = np.array(array, copy=True, order="C", subok=False)
        self._storage = Storage(contiguous, copy=True)
        self._shape = tuple(contiguous.shape)
        self._strides = contiguous_strides(self._shape)
        self._storage_offset = 0
        self.data = self._make_numpy_view()
        self.requires_grad = bool(requires_grad)
        self.grad: Optional[np.ndarray] = None
        self._grad_fn: Optional[GradFn] = None
        self.name = name

    @classmethod
    def _from_storage(
        cls,
        storage: Storage,
        *,
        shape: tuple[int, ...],
        strides: tuple[int, ...],
        storage_offset: int = 0,
        requires_grad: bool = False,
        name: Optional[str] = None,
    ) -> "Tensor":
        obj = cls.__new__(cls)
        obj._storage = storage
        obj._shape = tuple(int(x) for x in shape)
        obj._strides = tuple(int(x) for x in strides)
        obj._storage_offset = int(storage_offset)
        obj.data = obj._make_numpy_view()
        obj.requires_grad = bool(requires_grad)
        obj.grad = None
        obj._grad_fn = None
        obj.name = name
        return obj

    @classmethod
    def _from_operation_result(
        cls,
        array: np.ndarray,
        *,
        inputs: tuple["Tensor", ...],
        requires_grad: bool,
    ) -> "Tensor":
        """Preserve Storage when NumPy produced a genuine view of an input."""
        array = np.asarray(array)
        for source in inputs:
            if array.dtype != source.data.dtype:
                continue
            if not np.shares_memory(array, source.data):
                continue
            itemsize = int(array.dtype.itemsize)
            base_ptr = int(source._storage._array.__array_interface__["data"][0])
            out_ptr = int(array.__array_interface__["data"][0])
            byte_delta = out_ptr - base_ptr
            if byte_delta % itemsize != 0:
                continue
            offset = byte_delta // itemsize
            strides = tuple(int(s // itemsize) for s in array.strides)
            return cls._from_storage(
                source._storage,
                shape=tuple(array.shape),
                strides=strides,
                storage_offset=offset,
                requires_grad=requires_grad,
            )
        return cls(array, requires_grad=requires_grad, dtype=array.dtype)

    def _make_numpy_view(self) -> np.ndarray:
        if any(dim == 0 for dim in self._shape):
            return np.empty(self._shape, dtype=self._storage.dtype)
        itemsize = int(self._storage.dtype.itemsize)
        return np.ndarray(
            shape=self._shape,
            dtype=self._storage.dtype,
            buffer=self._storage._array,
            offset=self._storage_offset * itemsize,
            strides=tuple(stride * itemsize for stride in self._strides),
        )

    # ---------- memory/layout metadata ----------
    @property
    def storage(self) -> Storage:
        return self._storage

    @property
    def storage_offset(self) -> int:
        return self._storage_offset

    @property
    def _version(self) -> int:
        return self._storage.version

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def size(self) -> int:
        return int(np.prod(self._shape, dtype=int)) if self._shape else 1

    @property
    def dtype(self):
        return self._storage.dtype

    def stride(self, dim: Optional[int] = None):
        if dim is None:
            return self._strides
        return self._strides[dim]

    def numel(self) -> int:
        return self.size

    def is_contiguous(self) -> bool:
        expected = 1
        for size, stride in zip(reversed(self._shape), reversed(self._strides)):
            if size == 1:
                continue
            if stride != expected:
                return False
            expected *= size
        return True

    # ---------- autograd metadata ----------
    @property
    def grad_fn(self) -> Optional[str]:
        return None if self._grad_fn is None else self._grad_fn.name

    @property
    def is_leaf(self) -> bool:
        return self.requires_grad and self._grad_fn is None

    def backward(self, grad: Optional[Any] = None) -> None:
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
        local_grads: dict[int, np.ndarray] = {id(self): seed}

        for node in reversed(topo):
            node_grad = local_grads.get(id(node))
            if node_grad is None:
                continue
            if node.requires_grad:
                if node.is_leaf:
                    node.grad = node_grad.copy() if node.grad is None else node.grad + node_grad
                else:
                    node.grad = None
            if node._grad_fn is None:
                continue

            node._grad_fn.ctx.check_saved_versions()
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
                local_grads[key] = parent_grad if key not in local_grads else local_grads[key] + parent_grad

    def zero_grad(self) -> None:
        self.grad = None

    def detach(self) -> "Tensor":
        return Tensor._from_storage(
            self._storage,
            shape=self._shape,
            strides=self._strides,
            storage_offset=self._storage_offset,
            requires_grad=False,
            name=self.name,
        )

    def numpy(self) -> np.ndarray:
        return self.data

    def item(self):
        if self.data.size != 1:
            raise ValueError("only one-element tensors can be converted to Python scalars")
        return self.data.item()

    # ---------- tracked inplace operations ----------
    def _check_inplace_allowed(self) -> None:
        if self.requires_grad and self.is_leaf and is_grad_enabled():
            raise RuntimeError(
                "a leaf Tensor that requires grad cannot be modified inplace while grad is enabled; "
                "use no_grad() for parameter-style updates"
            )

    def _mark_dirty(self) -> None:
        self._storage.bump_version()

    def add_(self, other) -> "Tensor":
        self._check_inplace_allowed()
        value = other.data if isinstance(other, Tensor) else other
        result = self.data + value
        if result.shape != self.shape:
            raise ValueError("inplace add cannot change tensor shape")
        self.data[...] = result
        self._mark_dirty()
        return self

    def mul_(self, other) -> "Tensor":
        self._check_inplace_allowed()
        value = other.data if isinstance(other, Tensor) else other
        result = self.data * value
        if result.shape != self.shape:
            raise ValueError("inplace mul cannot change tensor shape")
        self.data[...] = result
        self._mark_dirty()
        return self

    def copy_(self, other) -> "Tensor":
        self._check_inplace_allowed()
        value = other.data if isinstance(other, Tensor) else np.asarray(other)
        if value.shape != self.shape:
            raise ValueError(f"copy_ shape mismatch: {value.shape} != {self.shape}")
        self.data[...] = value
        self._mark_dirty()
        return self

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

    # ---------- reductions/layout ----------
    def sum(self, axis=None, keepdims: bool = False):
        from .ops import Sum
        return Sum.apply(self, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims: bool = False):
        from .ops import Mean
        return Mean.apply(self, axis=axis, keepdims=keepdims)

    def view(self, *shape: int):
        from .ops import View
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        target = _normalize_shape(tuple(shape), self.numel())
        return View.apply(self, shape=target)

    def reshape(self, *shape: int):
        from .ops import Reshape
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        target = _normalize_shape(tuple(shape), self.numel())
        return Reshape.apply(self, shape=target)

    def transpose(self, dim0: int, dim1: int):
        from .ops import Transpose
        return Transpose.apply(self, dim0=int(dim0), dim1=int(dim1))

    def permute(self, *dims: int):
        from .ops import Permute
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = tuple(dims[0])
        return Permute.apply(self, dims=tuple(int(d) for d in dims))

    def contiguous(self):
        from .ops import Contiguous
        return self if self.is_contiguous() else Contiguous.apply(self)

    @property
    def T(self):
        if self.ndim != 2:
            raise ValueError(".T currently supports 2-D tensors only")
        return self.transpose(0, 1)

    def __getitem__(self, index):
        from .ops import Slice
        return Slice.apply(self, index=index)

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
        meta += f", stride={self._strides}, offset={self._storage_offset}"
        if self.grad_fn is not None:
            meta += f", grad_fn=<{self.grad_fn}>"
        return f"Tensor({self.data!r}{meta})"
