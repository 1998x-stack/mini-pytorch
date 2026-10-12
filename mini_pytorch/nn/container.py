"""Container and shape modules."""
from __future__ import annotations

import math

from .module import Module


class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        for index, module in enumerate(modules):
            self.add_module(str(index), module)

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x

    def __len__(self) -> int:
        return len(self._modules)

    def __getitem__(self, index: int) -> Module:
        return list(self._modules.values())[index]


class Flatten(Module):
    """Flatten a contiguous range of dimensions using Tensor.reshape."""

    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = int(start_dim)
        self.end_dim = int(end_dim)

    def forward(self, x):
        ndim = x.ndim
        start = self.start_dim if self.start_dim >= 0 else ndim + self.start_dim
        end = self.end_dim if self.end_dim >= 0 else ndim + self.end_dim
        if not 0 <= start < ndim or not 0 <= end < ndim or start > end:
            raise ValueError(
                f"invalid flatten dimensions start_dim={self.start_dim}, end_dim={self.end_dim} "
                f"for input with {ndim} dims"
            )
        flattened = math.prod(x.shape[start : end + 1])
        shape = x.shape[:start] + (flattened,) + x.shape[end + 1 :]
        return x.reshape(shape)

    def extra_repr(self) -> str:
        return f"start_dim={self.start_dim}, end_dim={self.end_dim}"
