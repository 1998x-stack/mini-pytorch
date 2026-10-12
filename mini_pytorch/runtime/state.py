"""Logical runtime state objects for mini-pytorch checkpoints."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


CHECKPOINT_SCHEMA_VERSION = 1


def _validate_counter(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


@dataclass
class TrainingState:
    """Explicit position and small metadata for a training loop."""

    epoch: int = 0
    global_step: int = 0
    batch_in_epoch: int = 0
    best_metric: float | None = None
    run_id: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        self.epoch = _validate_counter("epoch", self.epoch)
        self.global_step = _validate_counter("global_step", self.global_step)
        self.batch_in_epoch = _validate_counter("batch_in_epoch", self.batch_in_epoch)
        if self.best_metric is not None:
            self.best_metric = float(self.best_metric)
        if self.run_id is not None and not isinstance(self.run_id, str):
            raise TypeError("run_id must be a string or None")
        if self.notes is not None and not isinstance(self.notes, str):
            raise TypeError("notes must be a string or None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "batch_in_epoch": self.batch_in_epoch,
            "best_metric": self.best_metric,
            "run_id": self.run_id,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "TrainingState":
        if not isinstance(state, dict):
            raise TypeError("training state must be a dictionary")
        required = {"epoch", "global_step", "batch_in_epoch"}
        missing = sorted(required - set(state))
        if missing:
            raise KeyError(f"training state missing keys={missing}")
        return cls(
            epoch=state["epoch"],
            global_step=state["global_step"],
            batch_in_epoch=state["batch_in_epoch"],
            best_metric=state.get("best_metric"),
            run_id=state.get("run_id"),
            notes=state.get("notes"),
        )


@dataclass
class CheckpointState:
    """In-memory logical checkpoint independent from its file representation."""

    framework_version: str
    model_state: dict[str, np.ndarray]
    optimizer_state: dict[str, Any]
    scheduler_state: dict[str, Any] | None
    rng_state: dict[str, Any]
    training_state: TrainingState
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = CHECKPOINT_SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported checkpoint schema version {self.schema_version}; "
                f"expected {CHECKPOINT_SCHEMA_VERSION}"
            )
        if not isinstance(self.framework_version, str) or not self.framework_version:
            raise ValueError("framework_version must be a non-empty string")
        if not isinstance(self.model_state, dict):
            raise TypeError("model_state must be a dictionary")
        for key, value in self.model_state.items():
            if not isinstance(key, str):
                raise TypeError("model_state keys must be strings")
            if not isinstance(value, np.ndarray):
                raise TypeError(f"model_state[{key!r}] must be a NumPy array")
        if not isinstance(self.optimizer_state, dict):
            raise TypeError("optimizer_state must be a dictionary")
        if self.scheduler_state is not None and not isinstance(self.scheduler_state, dict):
            raise TypeError("scheduler_state must be a dictionary or None")
        if not isinstance(self.rng_state, dict):
            raise TypeError("rng_state must be a dictionary")
        if not isinstance(self.training_state, TrainingState):
            raise TypeError("training_state must be a TrainingState")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary")

    def summary(self) -> dict[str, Any]:
        """Small human-readable summary without tensor payloads."""
        self.validate()
        return {
            "schema_version": self.schema_version,
            "framework_version": self.framework_version,
            "model_keys": sorted(self.model_state),
            "has_scheduler": self.scheduler_state is not None,
            "training_state": self.training_state.to_dict(),
            "metadata": dict(self.metadata),
        }
