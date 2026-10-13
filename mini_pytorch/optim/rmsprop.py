"""RMSprop optimizer."""
from __future__ import annotations

import numpy as np

from .optimizer import Optimizer


class RMSprop(Optimizer):
    def __init__(self, params, lr: float = 1e-2, alpha: float = 0.99, eps: float = 1e-8, weight_decay: float = 0.0, momentum: float = 0.0, centered: bool = False) -> None:
        if lr < 0:
            raise ValueError("lr must be non-negative")
        if not 0 <= alpha < 1:
            raise ValueError("alpha must be in [0, 1)")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if weight_decay < 0 or momentum < 0:
            raise ValueError("weight_decay and momentum must be non-negative")
        super().__init__(params, defaults={"lr": lr, "alpha": alpha, "eps": eps, "weight_decay": weight_decay, "momentum": momentum, "centered": centered})

    def _validate_param_group(self, group) -> None:
        if group["lr"] < 0:
            raise ValueError("lr must be non-negative")
        if not 0 <= group["alpha"] < 1:
            raise ValueError("alpha must be in [0, 1)")
        if group["eps"] <= 0:
            raise ValueError("eps must be positive")
        if group["weight_decay"] < 0 or group["momentum"] < 0:
            raise ValueError("weight_decay and momentum must be non-negative")
        if not isinstance(group["centered"], (bool, np.bool_)):
            raise TypeError("centered must be bool")

    def _validate_param_state(self, parameter, values, *, group, index) -> None:
        expected = {"square_avg"}
        if group["momentum"] > 0:
            expected.add("momentum_buffer")
        if group["centered"]:
            expected.add("grad_avg")
        self._require_state_keys(values, expected, index=index)

    def step(self) -> None:
        for group in self.param_groups:
            lr, alpha, eps = group["lr"], group["alpha"], group["eps"]
            weight_decay, momentum, centered = group["weight_decay"], group["momentum"], group["centered"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = np.asarray(p.grad)
                if weight_decay != 0:
                    grad = grad + weight_decay * p.data
                state = self.state.setdefault(id(p), {})
                if not state:
                    state["square_avg"] = np.zeros_like(p.data)
                    if momentum > 0:
                        state["momentum_buffer"] = np.zeros_like(p.data)
                    if centered:
                        state["grad_avg"] = np.zeros_like(p.data)
                square_avg = state["square_avg"]
                square_avg *= alpha
                square_avg += (1.0 - alpha) * grad * grad
                if centered:
                    grad_avg = state["grad_avg"]
                    grad_avg *= alpha
                    grad_avg += (1.0 - alpha) * grad
                    avg = np.sqrt(np.maximum(square_avg - grad_avg * grad_avg, 0.0)) + eps
                else:
                    avg = np.sqrt(square_avg) + eps
                if momentum > 0:
                    buf = state["momentum_buffer"]
                    buf *= momentum
                    buf += grad / avg
                    p.data -= lr * buf
                else:
                    p.data -= lr * grad / avg
                p._mark_dirty()
