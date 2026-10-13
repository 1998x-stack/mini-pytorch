"""Optimizer base class with PyTorch-like parameter groups and checkpointing."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List

import numpy as np

from ..nn.module import Parameter


class Optimizer:
    """Base class for all optimizers.

    The central abstraction is ``param_groups``. Every optimizer-specific option
    lives in a parameter group, so one model can use different learning rates,
    weight decay values, or other hyperparameters without changing model code.

    Subclasses generally only need to:
      1. validate their hyperparameters,
      2. call ``super().__init__(params, defaults=...)``, and
      3. implement ``step()``.

    NumPy arrays stored in per-parameter ``state`` are parameter-shaped in this
    educational framework. Scalar metadata such as step counters should use
    Python scalars. That invariant lets checkpoint loading reject malformed state
    before a later optimizer step fails through accidental broadcasting.
    """

    def __init__(self, params: Iterable, defaults: Dict[str, Any]) -> None:
        self.defaults = dict(defaults)
        self.state: Dict[int, Dict[str, Any]] = {}
        self.param_groups: List[Dict[str, Any]] = []

        params = list(params)
        if not params:
            raise ValueError("optimizer got an empty parameter list")

        if isinstance(params[0], dict):
            for group in params:
                self.add_param_group(group)
        else:
            self.add_param_group({"params": params})

    def add_param_group(self, group: Dict[str, Any]) -> None:
        if "params" not in group:
            raise KeyError("parameter group must contain 'params'")

        params = group["params"]
        if isinstance(params, Parameter):
            params = [params]
        else:
            params = list(params)

        if not params:
            raise ValueError("parameter group is empty")
        if not all(isinstance(p, Parameter) for p in params):
            raise TypeError("optimizer can only optimize nn.Parameter objects")

        param_ids = [id(p) for p in params]
        if len(param_ids) != len(set(param_ids)):
            raise ValueError("a parameter appears more than once in the same parameter group")

        seen = {id(p) for g in self.param_groups for p in g["params"]}
        if any(pid in seen for pid in param_ids):
            raise ValueError("some parameters appear in more than one parameter group")

        normalized = dict(self.defaults)
        normalized.update({k: v for k, v in group.items() if k != "params"})
        normalized["params"] = params
        self.param_groups.append(normalized)

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear gradients for every parameter owned by this optimizer."""
        for group in self.param_groups:
            for p in group["params"]:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad = np.zeros_like(p.data)

    def step(self) -> None:
        raise NotImplementedError

    def state_dict(self) -> Dict[str, Any]:
        """Return a parameter-indexed, checkpoint-friendly state dictionary."""
        flat_params = [p for g in self.param_groups for p in g["params"]]
        index = {id(p): i for i, p in enumerate(flat_params)}

        groups = []
        for group in self.param_groups:
            copied = {k: deepcopy(v) for k, v in group.items() if k != "params"}
            copied["params"] = [index[id(p)] for p in group["params"]]
            groups.append(copied)

        state = {}
        for pid, values in self.state.items():
            if pid not in index:
                continue
            state[index[pid]] = {
                key: value.copy() if isinstance(value, np.ndarray) else deepcopy(value)
                for key, value in values.items()
            }
        return {"state": state, "param_groups": groups}

    @staticmethod
    def _normalize_state_index(value: Any, *, where: str) -> int:
        """Normalize checkpoint parameter indices without lossy coercion."""

        if isinstance(value, bool):
            raise TypeError(f"{where} parameter index must be an integer, not bool")
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, str):
            try:
                parsed = int(value)
            except ValueError as exc:
                raise TypeError(f"{where} parameter index must be an integer") from exc
            if value.strip() not in {str(parsed), f"+{parsed}" if parsed >= 0 else str(parsed)}:
                raise TypeError(f"{where} parameter index must be an integer")
            return parsed
        raise TypeError(f"{where} parameter index must be an integer")

    @staticmethod
    def _copy_and_validate_param_state(
        parameter: Parameter,
        values: Dict[str, Any],
        *,
        index: int,
    ) -> Dict[str, Any]:
        if not isinstance(values, dict):
            raise TypeError(f"optimizer state for parameter index {index} must be a dict")

        copied: Dict[str, Any] = {}
        for key, value in values.items():
            if isinstance(value, np.ndarray):
                if value.shape != parameter.shape:
                    raise ValueError(
                        f"optimizer state {key!r} for parameter index {index} has shape "
                        f"{value.shape}; expected {parameter.shape}"
                    )
                copied[key] = value.copy()
            else:
                copied[key] = deepcopy(value)
        return copied

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore optimizer state while keeping the current Parameter objects.

        Checkpoints identify parameters by stable positional indices rather than
        Python ``id`` values, because object ids change between processes.

        Validation is transactional: malformed checkpoint content is rejected
        before current parameter-group options or optimizer state are mutated.
        """
        if not isinstance(state_dict, dict):
            raise TypeError("optimizer state_dict must be a dict")
        if "state" not in state_dict or "param_groups" not in state_dict:
            raise KeyError("optimizer state_dict must contain state and param_groups")
        if not isinstance(state_dict["state"], dict):
            raise TypeError("optimizer state_dict['state'] must be a dict")

        saved_groups = state_dict["param_groups"]
        if not isinstance(saved_groups, (list, tuple)):
            raise TypeError("optimizer state_dict['param_groups'] must be a sequence")
        if len(saved_groups) != len(self.param_groups):
            raise ValueError("loaded state has a different number of parameter groups")

        current_flat: List[Parameter] = []
        saved_indices: List[int] = []
        pending_group_options: List[Dict[str, Any]] = []

        for group_number, (current_group, saved_group) in enumerate(
            zip(self.param_groups, saved_groups)
        ):
            if not isinstance(saved_group, dict):
                raise TypeError(f"loaded parameter group {group_number} must be a dict")
            if "params" not in saved_group:
                raise KeyError(f"loaded parameter group {group_number} is missing 'params'")

            current_params = current_group["params"]
            raw_indices = list(saved_group["params"])
            group_indices = [
                self._normalize_state_index(
                    value,
                    where=f"loaded parameter group {group_number}",
                )
                for value in raw_indices
            ]
            if len(current_params) != len(group_indices):
                raise ValueError("loaded state has a different parameter-group size")

            current_flat.extend(current_params)
            saved_indices.extend(group_indices)
            pending_group_options.append(
                {k: deepcopy(v) for k, v in saved_group.items() if k != "params"}
            )

        if len(set(saved_indices)) != len(saved_indices):
            raise ValueError("loaded optimizer state contains duplicate parameter indices")

        index_to_param = {idx: param for idx, param in zip(saved_indices, current_flat)}
        pending_state: Dict[int, Dict[str, Any]] = {}
        seen_state_indices: set[int] = set()

        for raw_idx, values in state_dict["state"].items():
            idx = self._normalize_state_index(raw_idx, where="loaded optimizer state")
            if idx in seen_state_indices:
                raise ValueError(
                    f"loaded optimizer state contains duplicate normalized parameter index {idx}"
                )
            seen_state_indices.add(idx)
            if idx not in index_to_param:
                raise ValueError(
                    f"loaded optimizer state references unknown parameter index {idx}"
                )
            param = index_to_param[idx]
            pending_state[id(param)] = self._copy_and_validate_param_state(
                param,
                values,
                index=idx,
            )

        # Apply only after the complete checkpoint has passed validation.
        for current_group, options in zip(self.param_groups, pending_group_options):
            for key, value in options.items():
                current_group[key] = value
        self.state = pending_state

    @property
    def learning_rates(self) -> List[float]:
        """Current learning rate for each parameter group."""
        return [float(group["lr"]) for group in self.param_groups]
