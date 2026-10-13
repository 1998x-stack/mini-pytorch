"""Stochastic gradient descent with momentum and Nesterov acceleration."""
from __future__ import annotations

import numpy as np

from .optimizer import Optimizer


class SGD(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-2,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        if lr < 0:
            raise ValueError("lr must be non-negative")
        if momentum < 0:
            raise ValueError("momentum must be non-negative")
        if dampening < 0:
            raise ValueError("dampening must be non-negative")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov requires momentum > 0 and dampening == 0")

        super().__init__(
            params,
            defaults={
                "lr": lr,
                "momentum": momentum,
                "dampening": dampening,
                "weight_decay": weight_decay,
                "nesterov": nesterov,
            },
        )

    def _validate_param_group(self, group) -> None:
        if group["lr"] < 0:
            raise ValueError("lr must be non-negative")
        if group["momentum"] < 0:
            raise ValueError("momentum must be non-negative")
        if group["dampening"] < 0:
            raise ValueError("dampening must be non-negative")
        if group["weight_decay"] < 0:
            raise ValueError("weight_decay must be non-negative")
        if group["nesterov"] and (group["momentum"] <= 0 or group["dampening"] != 0):
            raise ValueError("Nesterov requires momentum > 0 and dampening == 0")

    def _validate_param_state(self, parameter, values, *, group, index) -> None:
        expected = {"momentum_buffer"} if group["momentum"] != 0 else set()
        self._require_state_keys(values, expected, index=index)

    def step(self) -> None:
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = np.asarray(p.grad)

                if weight_decay != 0:
                    grad = grad + weight_decay * p.data

                if momentum != 0:
                    state = self.state.setdefault(id(p), {})
                    buf = state.get("momentum_buffer")
                    if buf is None:
                        buf = grad.copy()
                    else:
                        buf = momentum * buf + (1.0 - dampening) * grad
                    state["momentum_buffer"] = buf
                    grad = grad + momentum * buf if nesterov else buf

                p.data -= lr * grad
                p._mark_dirty()
