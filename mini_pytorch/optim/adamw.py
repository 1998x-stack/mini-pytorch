"""AdamW: Adam with decoupled weight decay."""
from __future__ import annotations

import numpy as np

from .optimizer import Optimizer


class AdamW(Optimizer):
    """AdamW optimizer with decoupled weight decay."""

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8, weight_decay: float = 1e-2) -> None:
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

    def step(self) -> None:
        for group in self.param_groups:
            lr, eps, weight_decay = group["lr"], group["eps"], group["weight_decay"]
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = np.asarray(p.grad)
                if weight_decay != 0:
                    p.data *= 1.0 - lr * weight_decay
                state = self.state.setdefault(id(p), {})
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = np.zeros_like(p.data)
                    state["exp_avg_sq"] = np.zeros_like(p.data)
                state["step"] += 1
                t = state["step"]
                m, v = state["exp_avg"], state["exp_avg_sq"]
                m *= beta1
                m += (1.0 - beta1) * grad
                v *= beta2
                v += (1.0 - beta2) * grad * grad
                bias_correction1 = 1.0 - beta1**t
                bias_correction2 = 1.0 - beta2**t
                m_hat = m / bias_correction1
                v_hat = v / bias_correction2
                p.data -= lr * m_hat / (np.sqrt(v_hat) + eps)
                p._mark_dirty()
