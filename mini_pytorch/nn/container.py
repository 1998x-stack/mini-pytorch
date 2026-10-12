"""Container modules."""
from __future__ import annotations

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
