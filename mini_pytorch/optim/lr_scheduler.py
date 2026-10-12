"""Learning-rate schedulers with a small PyTorch-like API."""
from __future__ import annotations

import bisect
import math
from copy import deepcopy
from typing import Iterable, List, Sequence

from .optimizer import Optimizer


class LRScheduler:
    """Base class for epoch-based learning-rate schedulers."""

    def __init__(self, optimizer: Optimizer, last_epoch: int = -1) -> None:
        if not isinstance(optimizer, Optimizer):
            raise TypeError("optimizer must be an Optimizer")
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault("initial_lr", group["lr"])
        elif any("initial_lr" not in group for group in optimizer.param_groups):
            raise KeyError("initial_lr is required when resuming a scheduler")
        self.base_lrs = [float(group["initial_lr"]) for group in optimizer.param_groups]
        self.last_epoch = int(last_epoch)
        self._step_count = 0
        self._last_lr = [float(group["lr"]) for group in optimizer.param_groups]
        self._initial_step()

    def _initial_step(self) -> None:
        self._step_count = 0
        self.step()

    def get_lr(self) -> List[float]:
        raise NotImplementedError

    def step(self, epoch: int | None = None) -> None:
        self.last_epoch = self.last_epoch + 1 if epoch is None else int(epoch)
        values = list(self.get_lr())
        if len(values) != len(self.optimizer.param_groups):
            raise ValueError("scheduler returned the wrong number of learning rates")
        for group, lr in zip(self.optimizer.param_groups, values):
            group["lr"] = float(lr)
        self._last_lr = [float(v) for v in values]
        self._step_count += 1

    def get_last_lr(self) -> List[float]:
        return list(self._last_lr)

    def state_dict(self) -> dict:
        return {"base_lrs": list(self.base_lrs), "last_epoch": self.last_epoch, "_step_count": self._step_count, "_last_lr": list(self._last_lr)}

    def load_state_dict(self, state_dict: dict) -> None:
        self.base_lrs = list(state_dict["base_lrs"])
        self.last_epoch = int(state_dict["last_epoch"])
        self._step_count = int(state_dict.get("_step_count", 0))
        self._last_lr = list(state_dict.get("_last_lr", self._last_lr))
        for group, lr in zip(self.optimizer.param_groups, self._last_lr):
            group["lr"] = float(lr)


class LambdaLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, lr_lambda, last_epoch: int = -1) -> None:
        if callable(lr_lambda):
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        else:
            self.lr_lambdas = list(lr_lambda)
            if len(self.lr_lambdas) != len(optimizer.param_groups):
                raise ValueError("one lr_lambda is required per parameter group")
            if not all(callable(fn) for fn in self.lr_lambdas):
                raise TypeError("all lr_lambda values must be callable")
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        return [base_lr * fn(self.last_epoch) for base_lr, fn in zip(self.base_lrs, self.lr_lambdas)]


class StepLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1, last_epoch: int = -1) -> None:
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        self.step_size, self.gamma = int(step_size), float(gamma)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        factor = self.gamma ** (self.last_epoch // self.step_size)
        return [base_lr * factor for base_lr in self.base_lrs]


class MultiStepLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, milestones: Iterable[int], gamma: float = 0.1, last_epoch: int = -1) -> None:
        self.milestones = sorted(int(m) for m in milestones)
        if any(m < 0 for m in self.milestones):
            raise ValueError("milestones must be non-negative")
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        self.gamma = float(gamma)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        return [base_lr * self.gamma ** bisect.bisect_right(self.milestones, self.last_epoch) for base_lr in self.base_lrs]


class ExponentialLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, gamma: float, last_epoch: int = -1) -> None:
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        self.gamma = float(gamma)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        return [base_lr * self.gamma**self.last_epoch for base_lr in self.base_lrs]


class CosineAnnealingLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0.0, last_epoch: int = -1) -> None:
        if T_max <= 0:
            raise ValueError("T_max must be positive")
        if eta_min < 0:
            raise ValueError("eta_min must be non-negative")
        self.T_max, self.eta_min = int(T_max), float(eta_min)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        cosine = (1.0 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2.0
        return [self.eta_min + (base_lr - self.eta_min) * cosine for base_lr in self.base_lrs]


class LinearLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, start_factor: float = 1.0 / 3.0, end_factor: float = 1.0, total_iters: int = 5, last_epoch: int = -1) -> None:
        if not 0 <= start_factor <= 1 or not 0 <= end_factor <= 1:
            raise ValueError("start_factor and end_factor must be in [0, 1]")
        if total_iters <= 0:
            raise ValueError("total_iters must be positive")
        self.start_factor, self.end_factor, self.total_iters = float(start_factor), float(end_factor), int(total_iters)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        progress = min(max(self.last_epoch, 0), self.total_iters) / self.total_iters
        factor = self.start_factor + progress * (self.end_factor - self.start_factor)
        return [base_lr * factor for base_lr in self.base_lrs]


class ReduceLROnPlateau:
    """Metric-driven scheduler. Call ``step(metric)`` after validation."""

    def __init__(self, optimizer: Optimizer, mode: str = "min", factor: float = 0.1, patience: int = 10, threshold: float = 1e-4, threshold_mode: str = "rel", cooldown: int = 0, min_lr: float | Sequence[float] = 0.0, eps: float = 1e-8) -> None:
        if not isinstance(optimizer, Optimizer):
            raise TypeError("optimizer must be an Optimizer")
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        if not 0 < factor < 1:
            raise ValueError("factor must be in (0, 1)")
        if patience < 0 or cooldown < 0:
            raise ValueError("patience and cooldown must be non-negative")
        if threshold_mode not in {"rel", "abs"}:
            raise ValueError("threshold_mode must be 'rel' or 'abs'")
        self.optimizer, self.mode, self.factor = optimizer, mode, float(factor)
        self.patience, self.threshold, self.threshold_mode = int(patience), float(threshold), threshold_mode
        self.cooldown, self.cooldown_counter, self.eps = int(cooldown), 0, float(eps)
        self.last_epoch, self.num_bad_epochs = 0, 0
        self.best = math.inf if mode == "min" else -math.inf
        if isinstance(min_lr, (list, tuple)):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("one min_lr is required per parameter group")
            self.min_lrs = [float(v) for v in min_lr]
        else:
            self.min_lrs = [float(min_lr)] * len(optimizer.param_groups)
        self._last_lr = optimizer.learning_rates

    def _is_better(self, current: float, best: float) -> bool:
        if self.mode == "min" and self.threshold_mode == "rel":
            return current < best * (1.0 - self.threshold)
        if self.mode == "min" and self.threshold_mode == "abs":
            return current < best - self.threshold
        if self.mode == "max" and self.threshold_mode == "rel":
            return current > best * (1.0 + self.threshold)
        return current > best + self.threshold

    def step(self, metric: float) -> None:
        current = float(metric)
        self.last_epoch += 1
        if self._is_better(current, self.best):
            self.best, self.num_bad_epochs = current, 0
        else:
            self.num_bad_epochs += 1
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0
        if self.num_bad_epochs > self.patience:
            for i, group in enumerate(self.optimizer.param_groups):
                old_lr = float(group["lr"])
                new_lr = max(old_lr * self.factor, self.min_lrs[i])
                if old_lr - new_lr > self.eps:
                    group["lr"] = new_lr
            self.cooldown_counter, self.num_bad_epochs = self.cooldown, 0
        self._last_lr = self.optimizer.learning_rates

    def get_last_lr(self) -> List[float]:
        return list(self._last_lr)

    def state_dict(self) -> dict:
        return {"mode": self.mode, "factor": self.factor, "patience": self.patience, "threshold": self.threshold, "threshold_mode": self.threshold_mode, "cooldown": self.cooldown, "cooldown_counter": self.cooldown_counter, "min_lrs": list(self.min_lrs), "eps": self.eps, "last_epoch": self.last_epoch, "num_bad_epochs": self.num_bad_epochs, "best": self.best, "_last_lr": list(self._last_lr)}

    def load_state_dict(self, state_dict: dict) -> None:
        for key, value in deepcopy(state_dict).items():
            setattr(self, key, value)
        for group, lr in zip(self.optimizer.param_groups, self._last_lr):
            group["lr"] = float(lr)
