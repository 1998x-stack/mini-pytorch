"""Dataset abstractions for mini-pytorch."""
from __future__ import annotations

from typing import Any

from ..tensor import Tensor


class Dataset:
    """Map-style dataset interface, mirroring the core PyTorch protocol."""

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class TensorDataset(Dataset):
    """Dataset that indexes multiple tensors/arrays along their first axis."""

    def __init__(self, *tensors) -> None:
        if not tensors:
            raise ValueError("TensorDataset requires at least one tensor")
        lengths = [len(t.data) if isinstance(t, Tensor) else len(t) for t in tensors]
        if len(set(lengths)) != 1:
            raise ValueError("all tensors must have the same first-dimension length")
        self.tensors = tensors
        self._length = lengths[0]

    def __getitem__(self, index: int):
        items = []
        for tensor in self.tensors:
            if isinstance(tensor, Tensor):
                items.append(tensor.data[index])
            else:
                items.append(tensor[index])
        return tuple(items)

    def __len__(self) -> int:
        return self._length
