"""Stochastic gradient descent."""
from __future__ import annotations

import numpy as np

from .optimizer import Optimizer


class SGD(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-2,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0:
            raise ValueError("lr must be non-negative")
        if momentum < 0:
            raise ValueError("momentum must be non-negative")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        super().__init__(
            params,
            defaults={"lr": lr, "momentum": momentum, "weight_decay": weight_decay},
        )

    def step(self) -> None:
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if weight_decay != 0:
                    grad = grad + weight_decay * p.data

                if momentum != 0:
                    state = self.state.setdefault(id(p), {})
                    velocity = state.get("momentum_buffer")
                    if velocity is None:
                        velocity = np.zeros_like(p.data)
                    velocity = momentum * velocity + grad
                    state["momentum_buffer"] = velocity
                    grad = velocity

                p.data -= lr * grad
