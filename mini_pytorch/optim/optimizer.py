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

        seen = {id(p) for g in self.param_groups for p in g["params"]}
        if any(id(p) in seen for p in params):
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

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore optimizer state while keeping the current Parameter objects.

        Checkpoints identify parameters by stable positional indices rather than
        Python ``id`` values, because object ids change between processes.
        """
        if "state" not in state_dict or "param_groups" not in state_dict:
            raise KeyError("optimizer state_dict must contain state and param_groups")

        saved_groups = state_dict["param_groups"]
        if len(saved_groups) != len(self.param_groups):
            raise ValueError("loaded state has a different number of parameter groups")

        current_flat: List[Parameter] = []
        saved_indices: List[int] = []
        for current_group, saved_group in zip(self.param_groups, saved_groups):
            current_params = current_group["params"]
            group_indices = list(saved_group["params"])
            if len(current_params) != len(group_indices):
                raise ValueError("loaded state has a different parameter-group size")
            current_flat.extend(current_params)
            saved_indices.extend(group_indices)

            for key, value in saved_group.items():
                if key != "params":
                    current_group[key] = deepcopy(value)

        if len(set(saved_indices)) != len(saved_indices):
            raise ValueError("loaded optimizer state contains duplicate parameter indices")

        index_to_param = {idx: param for idx, param in zip(saved_indices, current_flat)}
        self.state = {}
        for idx, values in state_dict["state"].items():
            idx = int(idx)
            if idx not in index_to_param:
                continue
            param = index_to_param[idx]
            self.state[id(param)] = {
                key: value.copy() if isinstance(value, np.ndarray) else deepcopy(value)
                for key, value in values.items()
            }

    @property
    def learning_rates(self) -> List[float]:
        """Current learning rate for each parameter group."""
        return [float(group["lr"]) for group in self.param_groups]
