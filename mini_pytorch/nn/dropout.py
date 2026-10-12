"""Dropout module."""
from __future__ import annotations

import numpy as np

from .module import Module
from .. import functional as F
from ..runtime.rng import integers as framework_integers
from ..tensor import Tensor


class Dropout(Module):
    """Inverted dropout with framework-owned or explicitly private RNG state.

    ``seed=None`` uses mini-pytorch's framework RNG so checkpoint restoration can
    resume future masks exactly. Passing an explicit seed keeps the historical
    component-local deterministic stream used by earlier versions.
    """

    def __init__(self, p: float = 0.5, *, seed: int | None = None) -> None:
        super().__init__()
        p = float(p)
        if not 0.0 <= p < 1.0:
            raise ValueError("dropout probability p must satisfy 0 <= p < 1")
        self.p = p
        self.seed = seed
        self._rng = None if seed is None else np.random.default_rng(seed)

    def _next_seed(self) -> int:
        high = np.iinfo(np.int64).max
        if self._rng is None:
            return int(framework_integers(0, high))
        return int(self._rng.integers(0, high))

    def forward(self, input: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return input
        return F.dropout(input, p=self.p, training=True, seed=self._next_seed())

    def extra_repr(self) -> str:
        return f"p={self.p}"
