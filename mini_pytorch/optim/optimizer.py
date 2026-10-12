"""Optimizer base class with PyTorch-like parameter groups."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np

from ..nn.module import Parameter


class Optimizer:
    """Base optimizer.

    Parameter groups make optimizer extensions possible without changing model
    code: different groups may use different learning rates or other options.
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
        duplicates = [p for p in params if id(p) in seen]
        if duplicates:
            raise ValueError("some parameters appear in more than one parameter group")

        normalized = dict(self.defaults)
        normalized.update({k: v for k, v in group.items() if k != "params"})
        normalized["params"] = params
        self.param_groups.append(normalized)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad = np.zeros_like(p.data)

    def step(self):
        raise NotImplementedError

    def state_dict(self) -> Dict[str, Any]:
        """Return an in-memory checkpoint-friendly representation."""
        flat_params = [p for g in self.param_groups for p in g["params"]]
        index = {id(p): i for i, p in enumerate(flat_params)}
        groups = []
        for group in self.param_groups:
            copied = {k: v for k, v in group.items() if k != "params"}
            copied["params"] = [index[id(p)] for p in group["params"]]
            groups.append(copied)

        state = {}
        for pid, values in self.state.items():
            if pid not in index:
                continue
            state[index[pid]] = {
                key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in values.items()
            }
        return {"state": state, "param_groups": groups}
