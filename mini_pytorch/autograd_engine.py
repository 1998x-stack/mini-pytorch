"""Backward execution engine for mini-pytorch v0.8.

The Engine owns transient gradient flow and graph-lifetime policy. Tensor owns
persistent user-visible state (`.grad`, hooks and retain-grad intent), while
GradFn/Context own operation-specific backward state.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class AutogradEngine:
    """Single-threaded educational reverse-mode execution engine."""

    def _build_topology(self, output: "Tensor") -> list["Tensor"]:
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

        build(output)
        return topo

    def _seed_gradient(self, output: "Tensor", grad: Any | None) -> np.ndarray:
        from .tensor import Tensor

        if grad is None:
            if output.data.size != 1:
                raise RuntimeError("grad must be specified for non-scalar outputs")
            return np.ones_like(output.data)

        if isinstance(grad, Tensor):
            if grad.device != output.device:
                raise ValueError(
                    f"gradient device mismatch: {grad.device} vs output {output.device}"
                )
            seed = np.asarray(grad.data, dtype=output.data.dtype)
        else:
            seed = np.asarray(grad, dtype=output.data.dtype)

        if seed.shape != output.data.shape:
            raise ValueError(
                f"gradient shape {seed.shape} != tensor shape {output.data.shape}"
            )
        return seed

    @staticmethod
    def _accumulate_persistent_grad(node: "Tensor", grad: np.ndarray) -> None:
        if not node.requires_grad:
            return
        if node.is_leaf or node._retain_grad:
            node.grad = grad.copy() if node.grad is None else node.grad + grad
        elif not node.is_leaf:
            node.grad = None

    @staticmethod
    def _validate_graph_is_available(topo: list["Tensor"]) -> None:
        """Fail before observable hook/grad side effects on released graphs."""

        for node in topo:
            if node._grad_fn is not None:
                node._grad_fn.ctx.ensure_available()

    def backward(
        self,
        output: "Tensor",
        grad: Any | None = None,
        *,
        retain_graph: bool = False,
        create_graph: bool = False,
    ) -> None:
        """Execute reverse-mode autodiff from ``output``.

        ``create_graph`` is intentionally explicit but unsupported in v0.8:
        current backward formulas return NumPy arrays, so they cannot build a
        differentiable graph for second derivatives yet.
        """

        if create_graph:
            raise NotImplementedError(
                "create_graph=True is not supported in v0.8 because backward "
                "formulas currently produce NumPy gradients rather than Tensor "
                "operations"
            )
        if not output.requires_grad:
            raise RuntimeError(
                "cannot call backward() on a tensor that does not require grad"
            )

        seed = self._seed_gradient(output, grad)
        topo = self._build_topology(output)
        self._validate_graph_is_available(topo)

        local_grads: dict[int, np.ndarray] = {id(output): seed}

        for node in reversed(topo):
            node_grad = local_grads.get(id(node))
            if node_grad is None:
                continue

            node_grad = node._run_backward_hooks(node_grad)
            self._accumulate_persistent_grad(node, node_grad)

            grad_fn = node._grad_fn
            if grad_fn is None:
                continue

            grad_fn.ctx.check_saved_versions()
            parent_grads = grad_fn.function.backward(grad_fn.ctx, node_grad)
            if not isinstance(parent_grads, tuple):
                parent_grads = (parent_grads,)
            if len(parent_grads) != len(grad_fn.parents):
                raise RuntimeError(
                    f"{grad_fn.function.__name__}.backward returned "
                    f"{len(parent_grads)} grads for {len(grad_fn.parents)} parents"
                )

            for parent, parent_grad in zip(grad_fn.parents, parent_grads):
                if parent_grad is None or not parent.requires_grad:
                    continue
                parent_grad = np.asarray(parent_grad, dtype=parent.data.dtype)
                if parent_grad.shape != parent.data.shape:
                    raise RuntimeError(
                        f"{grad_fn.function.__name__}.backward produced gradient "
                        f"shape {parent_grad.shape} for parent shape {parent.data.shape}"
                    )
                key = id(parent)
                local_grads[key] = (
                    parent_grad
                    if key not in local_grads
                    else local_grads[key] + parent_grad
                )

            if not retain_graph:
                grad_fn.release()


engine = AutogradEngine()


try:  # pragma: no cover
    from .tensor import Tensor
except Exception:  # pragma: no cover
    Tensor = Any
