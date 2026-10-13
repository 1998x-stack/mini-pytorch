"""Runtime signatures used to specialize captured graphs in v0.12."""
from __future__ import annotations

from dataclasses import dataclass

from ..autograd import is_grad_enabled


@dataclass(frozen=True)
class TensorSignature:
    """Immutable metadata relevant to one graph-input specialization."""

    shape: tuple[int, ...]
    dtype: str
    device: str
    requires_grad: bool

    @classmethod
    def from_tensor(cls, tensor: "Tensor") -> "TensorSignature":
        from ..tensor import Tensor

        if not isinstance(tensor, Tensor):
            raise TypeError("TensorSignature.from_tensor expects a Tensor")
        return cls(
            shape=tuple(int(x) for x in tensor.shape),
            dtype=str(tensor.dtype),
            device=str(tensor.device),
            requires_grad=bool(tensor.requires_grad),
        )

    def format(self) -> str:
        return (
            f"Tensor(shape={self.shape}, dtype={self.dtype}, device={self.device}, "
            f"requires_grad={self.requires_grad})"
        )


@dataclass(frozen=True)
class InputSignature:
    """Specialization signature for one compiled-function runtime call."""

    tensors: tuple[TensorSignature, ...]
    grad_enabled: bool

    @classmethod
    def from_inputs(
        cls,
        *inputs: "Tensor",
        grad_enabled: bool | None = None,
    ) -> "InputSignature":
        from ..tensor import Tensor

        if not inputs:
            raise ValueError("compiled graph calls require at least one Tensor input")
        if not all(isinstance(value, Tensor) for value in inputs):
            raise TypeError("compiled graph inputs must all be Tensor objects")
        if len({id(value) for value in inputs}) != len(inputs):
            raise ValueError("the same Tensor object cannot occupy multiple graph input slots")
        enabled = is_grad_enabled() if grad_enabled is None else bool(grad_enabled)
        return cls(
            tensors=tuple(TensorSignature.from_tensor(value) for value in inputs),
            grad_enabled=enabled,
        )

    def format(self) -> str:
        lines = [f"grad_enabled={self.grad_enabled}"]
        lines.extend(f"input[{i}]={sig.format()}" for i, sig in enumerate(self.tensors))
        return "\n".join(lines)


try:  # pragma: no cover
    from ..tensor import Tensor
except Exception:  # pragma: no cover
    Tensor = object
