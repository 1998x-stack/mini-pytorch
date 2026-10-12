"""Dropout module."""
from __future__ import annotations

import numpy as np

from .module import Module
from .. import functional as F
from ..tensor import Tensor


class Dropout(Module):
    """Inverted dropout with deterministic seeded educational behavior."""

    def __init__(self, p: float = 0.5, *, seed: int | None = None) -> None:
        super().__init__()
        p = float(p)
        if not 0.0 <= p < 1.0:
            raise ValueError("dropout probability p must satisfy 0 <= p < 1")
        self.p = p
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def forward(self, input: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return input
        step_seed = int(self._rng.integers(0, np.iinfo(np.int64).max))
        return F.dropout(input, p=self.p, training=True, seed=step_seed)

    def extra_repr(self) -> str:
        return f"p={self.p}"
