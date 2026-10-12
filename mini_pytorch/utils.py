"""Small educational utilities."""
from __future__ import annotations

import numpy as np

from .tensor import Tensor


def gradcheck(fn, inputs, eps: float = 1e-6, atol: float = 1e-5, rtol: float = 1e-4):
    """Finite-difference check for a scalar-valued function.

    Args:
        fn: callable receiving Tensors and returning one scalar Tensor.
        inputs: iterable of NumPy arrays or array-like inputs.
    Returns:
        ``(ok, details)`` where details contains analytical/numerical gradients.
    """
    tensors = [Tensor(np.array(x, dtype=float), requires_grad=True) for x in inputs]
    out = fn(*tensors)
    if out.size != 1:
        raise ValueError("gradcheck requires a scalar output")
    out.backward()
    analytical = [t.grad.copy() for t in tensors]

    numerical = []
    base = [np.array(x, dtype=float, copy=True) for x in inputs]
    for input_index, arr in enumerate(base):
        grad = np.zeros_like(arr)
        it = np.nditer(arr, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            idx = it.multi_index
            plus = [x.copy() for x in base]
            minus = [x.copy() for x in base]
            plus[input_index][idx] += eps
            minus[input_index][idx] -= eps
            f_plus = fn(*[Tensor(x) for x in plus]).item()
            f_minus = fn(*[Tensor(x) for x in minus]).item()
            grad[idx] = (f_plus - f_minus) / (2 * eps)
            it.iternext()
        numerical.append(grad)

    checks = [np.allclose(a, n, atol=atol, rtol=rtol) for a, n in zip(analytical, numerical)]
    return all(checks), {"analytical": analytical, "numerical": numerical, "checks": checks}
