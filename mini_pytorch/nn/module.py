"""Neural-network module hierarchy and automatic parameter registration."""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterator, Optional

import numpy as np

from ..tensor import Tensor


class Parameter(Tensor):
    """A Tensor that is automatically registered when assigned to a Module."""

    def __init__(self, data, *, requires_grad: bool = True, name: Optional[str] = None):
        if isinstance(data, Tensor):
            data = data.data
        super().__init__(data, requires_grad=requires_grad, name=name)


class Module:
    """Base class for all neural-network modules.

    Assigning a ``Parameter`` or another ``Module`` to an attribute registers it,
    which enables recursive ``parameters()``, ``state_dict()``, and train/eval.
    """

    def __init__(self) -> None:
        object.__setattr__(self, "_parameters", OrderedDict())
        object.__setattr__(self, "_modules", OrderedDict())
        object.__setattr__(self, "_buffers", OrderedDict())
        object.__setattr__(self, "training", True)

    def __setattr__(self, name, value) -> None:
        if name.startswith("_") or "_parameters" not in self.__dict__:
            object.__setattr__(self, name, value)
            return

        # Remove any previous registration under this name before reassigning.
        self._parameters.pop(name, None)
        self._modules.pop(name, None)
        self._buffers.pop(name, None)

        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        object.__setattr__(self, name, value)

    def register_parameter(self, name: str, parameter: Optional[Parameter]) -> None:
        if not name or "." in name:
            raise KeyError("parameter name must be non-empty and cannot contain '.'")
        if parameter is not None and not isinstance(parameter, Parameter):
            raise TypeError("parameter must be a Parameter or None")
        setattr(self, name, parameter)
        if parameter is None:
            self._parameters[name] = None

    def register_buffer(self, name: str, tensor: Tensor) -> None:
        if not name or "." in name:
            raise KeyError("buffer name must be non-empty and cannot contain '.'")
        if not isinstance(tensor, Tensor):
            raise TypeError("buffer must be a Tensor")
        self._parameters.pop(name, None)
        self._modules.pop(name, None)
        self._buffers[name] = tensor
        object.__setattr__(self, name, tensor)

    def add_module(self, name: str, module: "Module") -> None:
        if not name or "." in name:
            raise KeyError("module name must be non-empty and cannot contain '.'")
        if not isinstance(module, Module):
            raise TypeError("module must be a Module")
        self._parameters.pop(name, None)
        self._buffers.pop(name, None)
        self._modules[name] = module
        object.__setattr__(self, name, module)

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
        for name, param in self._parameters.items():
            if param is not None:
                full_name = f"{prefix}.{name}" if prefix else name
                yield full_name, param
        for child_name, module in self._modules.items():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            yield from module.named_parameters(child_prefix)

    def parameters(self) -> Iterator[Parameter]:
        for _, parameter in self.named_parameters():
            yield parameter

    def named_buffers(self, prefix: str = "") -> Iterator[tuple[str, Tensor]]:
        for name, buffer in self._buffers.items():
            full_name = f"{prefix}.{name}" if prefix else name
            yield full_name, buffer
        for child_name, module in self._modules.items():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            yield from module.named_buffers(child_prefix)

    def children(self) -> Iterator["Module"]:
        yield from self._modules.values()

    def modules(self) -> Iterator["Module"]:
        yield self
        for module in self._modules.values():
            yield from module.modules()

    def train(self, mode: bool = True) -> "Module":
        self.training = bool(mode)
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self) -> "Module":
        return self.train(False)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter in self.parameters():
            if set_to_none:
                parameter.grad = None
            else:
                parameter.grad = np.zeros_like(parameter.data)

    def state_dict(self) -> Dict[str, np.ndarray]:
        state: Dict[str, np.ndarray] = {}
        for name, parameter in self.named_parameters():
            state[name] = parameter.data.copy()
        for name, buffer in self.named_buffers():
            state[name] = buffer.data.copy()
        return state

    def load_state_dict(self, state: Dict[str, np.ndarray], strict: bool = True) -> None:
        destinations = {name: p for name, p in self.named_parameters()}
        destinations.update({name: b for name, b in self.named_buffers()})

        missing = [name for name in destinations if name not in state]
        unexpected = [name for name in state if name not in destinations]
        if strict and (missing or unexpected):
            raise KeyError(f"missing keys={missing}, unexpected keys={unexpected}")

        for name, target in destinations.items():
            if name not in state:
                continue
            value = np.asarray(state[name], dtype=target.data.dtype)
            if value.shape != target.data.shape:
                raise ValueError(
                    f"shape mismatch for {name}: expected {target.data.shape}, got {value.shape}"
                )
            target.data[...] = value

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def extra_repr(self) -> str:
        return ""

    def __repr__(self) -> str:
        lines = []
        extra = self.extra_repr()
        if extra:
            lines.append(extra)
        for name, module in self._modules.items():
            module_str = repr(module).replace("\n", "\n  ")
            lines.append(f"({name}): {module_str}")
        if not lines:
            return f"{self.__class__.__name__}()"
        body = "\n  ".join(lines)
        return f"{self.__class__.__name__}(\n  {body}\n)"
