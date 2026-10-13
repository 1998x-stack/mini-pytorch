"""Adamax optimizer (the infinity-norm variant of Adam)."""
from __future__ import annotations

import numpy as np

from .optimizer import Optimizer


class Adamax(Optimizer):
    def __init__(self, params, lr: float = 2e-3, betas=(0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.0) -> None:
        beta1, beta2 = betas
        if lr < 0:
            raise ValueError("lr must be non-negative")
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("betas must be in [0, 1)")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        super().__init__(params, defaults={"lr": lr, "betas": (beta1, beta2), "eps": eps, "weight_decay": weight_decay})

    def _validate_param_group(self, group) -> None:
        beta1, beta2 = group["betas"]
        if group["lr"] < 0:
            raise ValueError("lr must be non-negative")
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("betas must be in [0, 1)")
        if group["eps"] <= 0:
            raise ValueError("eps must be positive")
        if group["weight_decay"] < 0:
            raise ValueError("weight_decay must be non-negative")

    def step(self) -> None:
        for group in self.param_groups:
            lr, eps, weight_decay = group["lr"], group["eps"], group["weight_decay"]
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = np.asarray(p.grad)
                if weight_decay != 0:
                    grad = grad + weight_decay * p.data
                state = self.state.setdefault(id(p), {})
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = np.zeros_like(p.data)
                    state["exp_inf"] = np.zeros_like(p.data)
                state["step"] += 1
                t = state["step"]
                m, u = state["exp_avg"], state["exp_inf"]
                m *= beta1
                m += (1.0 - beta1) * grad
                np.maximum(beta2 * u, np.abs(grad) + eps, out=u)
                p.data -= (lr / (1.0 - beta1**t)) * m / u
                p._mark_dirty()
