"""Batch-normalization modules with explicit Parameter/Buffer separation."""
from __future__ import annotations

import numpy as np

from .module import Module, Parameter
from .. import functional as F
from ..tensor import Tensor


class _BatchNorm(Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        if num_features <= 0:
            raise ValueError("num_features must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if momentum is not None and not 0.0 <= float(momentum) <= 1.0:
            raise ValueError("momentum must be in [0, 1] or None")

        self.num_features = int(num_features)
        self.eps = float(eps)
        self.momentum = None if momentum is None else float(momentum)
        self.affine = bool(affine)
        self.track_running_stats = bool(track_running_stats)

        if self.affine:
            self.weight = Parameter(np.ones(self.num_features))
            self.bias = Parameter(np.zeros(self.num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if self.track_running_stats:
            self.register_buffer("running_mean", Tensor(np.zeros(self.num_features)))
            self.register_buffer("running_var", Tensor(np.ones(self.num_features)))
            self.register_buffer("num_batches_tracked", Tensor(np.array(0.0)))
        else:
            self.running_mean = None
            self.running_var = None
            self.num_batches_tracked = None

    def _check_input_dim(self, input: Tensor) -> None:
        raise NotImplementedError

    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)
        if input.shape[1] != self.num_features:
            raise ValueError(
                f"expected {self.num_features} channels/features, got {input.shape[1]}"
            )

        if self.training and self.track_running_stats:
            self.num_batches_tracked.data[...] += 1.0
            self.num_batches_tracked._mark_dirty()
            if self.momentum is None:
                factor = 1.0 / float(self.num_batches_tracked.item())
            else:
                factor = self.momentum
        else:
            factor = 0.0 if self.momentum is None else self.momentum

        return F.batch_norm(
            input,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            training=self.training or not self.track_running_stats,
            momentum=factor,
            eps=self.eps,
        )

    def extra_repr(self) -> str:
        return (
            f"num_features={self.num_features}, eps={self.eps}, momentum={self.momentum}, "
            f"affine={self.affine}, track_running_stats={self.track_running_stats}"
        )


class BatchNorm1d(_BatchNorm):
    def _check_input_dim(self, input: Tensor) -> None:
        if input.ndim not in (2, 3):
            raise ValueError("BatchNorm1d expects input shape (N, C) or (N, C, L)")


class BatchNorm2d(_BatchNorm):
    def _check_input_dim(self, input: Tensor) -> None:
        if input.ndim != 4:
            raise ValueError("BatchNorm2d expects input shape (N, C, H, W)")
