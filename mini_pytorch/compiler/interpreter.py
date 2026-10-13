"""Reference interpreter for mini-pytorch Graph IR."""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .capture import is_capturing
from .ir import Graph, ValueRef


class GraphInterpreter:
    """Replay Graph nodes through the normal mini-pytorch Dispatcher."""

    def __init__(self, graph: Graph) -> None:
        if not isinstance(graph, Graph):
            raise TypeError("GraphInterpreter expects a Graph")
        graph.validate()
        self.graph = graph

    def _execute(self, *inputs: "Tensor") -> tuple[Any, dict[int, Any]]:
        from ..dispatcher import dispatcher
        from ..tensor import Tensor

        if is_capturing():
            raise RuntimeError("GraphInterpreter cannot run inside an active graph capture")
        if len(inputs) != len(self.graph.inputs):
            raise ValueError(
                f"graph expects {len(self.graph.inputs)} inputs, got {len(inputs)}"
            )
        if not all(isinstance(value, Tensor) for value in inputs):
            raise TypeError("GraphInterpreter runtime inputs must be Tensor objects")

        env: dict[int, Any] = {}
        for value_id, tensor in zip(self.graph.inputs, inputs):
            env[value_id] = tensor

        for value_id, value in self.graph.values.items():
            if value.kind != "constant":
                continue
            payload = value.constant
            assert payload is not None
            env[value_id] = Tensor(
                np.array(payload.data, copy=True),
                requires_grad=payload.requires_grad,
                dtype=payload.data.dtype,
                device=payload.device,
            )

        for node in self.graph.nodes:
            args = self._resolve(node.args, env)
            kwargs = self._resolve(node.kwargs, env)
            result = dispatcher.dispatch(node.operator, *args, **kwargs)
            self._bind(node.outputs, result, env)

        return self._resolve(self.graph.outputs, env), env

    def run(self, *inputs: "Tensor"):
        output, _ = self._execute(*inputs)
        return output

    __call__ = run

    def run_with_values(self, *inputs: "Tensor", value_ids: Iterable[int]):
        """Replay and also expose selected internal Values.

        v0.13 uses this as the explicit forward/backward ABI for saved forward
        values. It is intentionally an interpreter/debug surface rather than a
        hidden Python closure over intermediates.
        """

        output, env = self._execute(*inputs)
        selected = []
        for value_id in value_ids:
            value_id = int(value_id)
            try:
                selected.append(env[value_id])
            except KeyError as exc:
                raise ValueError(f"requested graph value %v{value_id} is unavailable") from exc
        return output, tuple(selected)

    def _resolve(self, spec: Any, env: dict[int, Any]) -> Any:
        if isinstance(spec, ValueRef):
            try:
                return env[spec.value_id]
            except KeyError as exc:
                raise RuntimeError(
                    f"graph value %v{spec.value_id} is not available during replay"
                ) from exc
        if isinstance(spec, tuple):
            return tuple(self._resolve(x, env) for x in spec)
        if isinstance(spec, list):
            return [self._resolve(x, env) for x in spec]
        if isinstance(spec, dict):
            return {
                self._resolve(key, env): self._resolve(value, env)
                for key, value in spec.items()
            }
        return spec

    def _bind(self, spec: Any, result: Any, env: dict[int, Any]) -> None:
        if isinstance(spec, ValueRef):
            env[spec.value_id] = result
            return
        if isinstance(spec, tuple):
            if not isinstance(result, tuple) or len(spec) != len(result):
                raise RuntimeError("graph replay output structure mismatch (tuple)")
            for child, value in zip(spec, result):
                self._bind(child, value, env)
            return
        if isinstance(spec, list):
            if not isinstance(result, list) or len(spec) != len(result):
                raise RuntimeError("graph replay output structure mismatch (list)")
            for child, value in zip(spec, result):
                self._bind(child, value, env)
            return
        raise RuntimeError(f"unsupported graph output binding spec: {spec!r}")


try:  # pragma: no cover
    from ..tensor import Tensor
except Exception:  # pragma: no cover
    Tensor = Any
