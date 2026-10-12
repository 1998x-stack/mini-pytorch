"""Pure NumPy reference kernels used by the v0.5 dispatcher.

These functions intentionally know nothing about Autograd, Module or Optimizer.
They express backend numerical behavior only.
"""
from __future__ import annotations

import numpy as np


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
