"""Framework-owned random-number stream for reproducible training.

A seed selects the start of a stream. A checkpoint must capture the current
bit-generator state so future draws continue from the same position.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np


_GLOBAL_RNG = np.random.default_rng()


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be non-negative")
    return seed


def manual_seed(seed: int) -> int:
    """Reset the framework RNG stream and return the normalized seed."""
    global _GLOBAL_RNG
    seed = _validate_seed(seed)
    _GLOBAL_RNG = np.random.default_rng(seed)
    return seed


def get_rng_state() -> dict[str, Any]:
    """Return a detached, serializable snapshot of the current RNG position."""
    return deepcopy(_GLOBAL_RNG.bit_generator.state)


def set_rng_state(state: dict[str, Any]) -> None:
    """Restore a state previously produced by :func:`get_rng_state`."""
    global _GLOBAL_RNG
    if not isinstance(state, dict):
        raise TypeError("rng state must be a dictionary")
    bit_generator = state.get("bit_generator")
    if bit_generator != "PCG64":
        raise ValueError(
            f"unsupported bit generator {bit_generator!r}; mini-pytorch v0.9 uses PCG64"
        )
    generator = np.random.Generator(np.random.PCG64())
    try:
        generator.bit_generator.state = deepcopy(state)
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError("invalid RNG state") from exc
    _GLOBAL_RNG = generator


def random(size=None) -> np.ndarray | float:
    return _GLOBAL_RNG.random(size=size)


def uniform(low=0.0, high=1.0, size=None):
    return _GLOBAL_RNG.uniform(low, high, size=size)


def integers(low, high=None, size=None, *, endpoint: bool = False):
    return _GLOBAL_RNG.integers(low, high=high, size=size, endpoint=endpoint)


def permutation(x):
    return _GLOBAL_RNG.permutation(x)
