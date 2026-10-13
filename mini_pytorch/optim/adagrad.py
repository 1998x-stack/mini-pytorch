"""Adagrad optimizer."""
from __future__ import annotations

import numpy as np

from .optimizer import Optimizer


class Adagrad(Optimizer):
    def __init__(self, params, lr: float = 1e-2, lr_decay: float = 0.0, weight_decay: float = 0.0, initial_accumulator_value: float = 0.0, eps: float = 1e-10) -> None:
        if lr < 0 or lr_decay < 0 or weight_decay < 0:
            raise ValueError("lr, lr_decay, and weight_decay must be non-negative")
        if initial_accumulator_value < 0:
            raise ValueError("initial_accumulator_value must be non-negative")
        if eps <= 0:
            raise ValueError("eps must be positive")
        super().__init__(params, defaults={"lr": lr, "lr_decay": lr_decay, "weight_decay": weight_decay, "initial_accumulator_value": initial_accumulator_value, "eps": eps})

    def _validate_param_group(self, group) -> None:
        if group["lr"] < 0 or group["lr_decay"] < 0 or group["weight_decay"] < 0:
            raise ValueError("lr, lr_decay, and weight_decay must be non-negative")
        if group["initial_accumulator_value"] < 0:
            raise ValueError("initial_accumulator_value must be non-negative")
        if group["eps"] <= 0:
            raise ValueError("eps must be positive")

    def step(self) -> None:
        for group in self.param_groups:
            lr, lr_decay = group["lr"], group["lr_decay"]
            weight_decay, initial, eps = group["weight_decay"], group["initial_accumulator_value"], group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = np.asarray(p.grad)
                if weight_decay != 0:
                    grad = grad + weight_decay * p.data
                state = self.state.setdefault(id(p), {})
                if not state:
                    state["step"] = 0
                    state["sum"] = np.full_like(p.data, initial)
                state["step"] += 1
                step = state["step"]
                clr = lr / (1.0 + (step - 1.0) * lr_decay)
                state["sum"] += grad * grad
                p.data -= clr * grad / (np.sqrt(state["sum"]) + eps)
                p._mark_dirty()
