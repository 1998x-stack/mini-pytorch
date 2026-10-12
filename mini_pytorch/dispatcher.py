"""Educational operator dispatcher for mini-pytorch v0.5.

The real PyTorch dispatcher is substantially more sophisticated.  This module
keeps only the concepts needed to make the routing model visible:

logical operator -> schema -> DispatchKeySet -> kernel/fallback -> redispatch.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable, Optional


Kernel = Callable[..., Any]
Fallback = Callable[..., Any]


class DispatchError(RuntimeError):
    """Raised when operator definition or dispatch cannot be resolved."""


class DispatchKey(str, Enum):
    """Small educational subset of PyTorch-like dispatch keys."""

    AUTOGRAD = "Autograd"
    NOGRAD = "NoGrad"
    CUDA = "CUDA"  # reserved for a later device/backend version
    CPU = "CPU"


_PRIORITY: tuple[DispatchKey, ...] = (
    DispatchKey.AUTOGRAD,
    DispatchKey.NOGRAD,
    DispatchKey.CUDA,
    DispatchKey.CPU,
)


@dataclass(frozen=True)
class DispatchKeySet:
    """Immutable set with an explicit educational priority order."""

    keys: frozenset[DispatchKey]

    def __init__(self, keys: Iterable[DispatchKey] = ()) -> None:
        object.__setattr__(self, "keys", frozenset(DispatchKey(k) for k in keys))

    def __contains__(self, key: DispatchKey) -> bool:
        return DispatchKey(key) in self.keys

    def __iter__(self):
        for key in _PRIORITY:
            if key in self.keys:
                yield key

    def without(self, key: DispatchKey) -> "DispatchKeySet":
        return DispatchKeySet(k for k in self.keys if k != DispatchKey(key))

    def add(self, key: DispatchKey) -> "DispatchKeySet":
        return DispatchKeySet((*self.keys, DispatchKey(key)))

    def highest_priority(self) -> Optional[DispatchKey]:
        return next(iter(self), None)

    def __bool__(self) -> bool:
        return bool(self.keys)

    def __repr__(self) -> str:
        ordered = ", ".join(key.value for key in self)
        return f"DispatchKeySet({{{ordered}}})"


@dataclass(frozen=True)
class OperatorSchema:
    """Metadata describing one logical operator overload."""

    namespace: str
    name: str
    overload: str = "default"
    signature: str = ""

    def __post_init__(self) -> None:
        if not self.namespace or not self.name:
            raise ValueError("operator namespace and name must be non-empty")
        if "::" in self.namespace or "::" in self.name:
            raise ValueError("namespace/name must not contain '::'")
        if "." in self.name:
            raise ValueError("operator name must not contain overload syntax")
        if not self.overload:
            object.__setattr__(self, "overload", "default")

    @property
    def qualified_name(self) -> str:
        return f"{self.namespace}::{self.name}.{self.overload}"

    def __str__(self) -> str:
        suffix = f" {self.signature}" if self.signature else ""
        return f"{self.qualified_name}{suffix}"


@dataclass(frozen=True)
class DispatchDecision:
    operator: str
    keyset: DispatchKeySet
    selected_key: DispatchKey
    implementation: str


class Dispatcher:
    """Schema registry plus per-DispatchKey implementation table."""

    def __init__(self) -> None:
        self._schemas: dict[str, OperatorSchema] = {}
        self._kernels: dict[str, dict[DispatchKey, Kernel]] = {}
        self._fallbacks: dict[DispatchKey, Fallback] = {}

    # ---------- schema / registration ----------
    def define(self, schema: OperatorSchema) -> OperatorSchema:
        qname = schema.qualified_name
        if qname in self._schemas:
            raise DispatchError(f"operator schema already defined: {qname}")
        self._schemas[qname] = schema
        self._kernels[qname] = {}
        return schema

    def register(self, operator: str, key: DispatchKey, kernel: Kernel) -> None:
        if operator not in self._schemas:
            raise DispatchError(f"cannot register kernel for undefined operator: {operator}")
        key = DispatchKey(key)
        table = self._kernels[operator]
        if key in table:
            raise DispatchError(f"kernel already registered for {operator} at {key.value}")
        table[key] = kernel

    def register_fallback(self, key: DispatchKey, fallback: Fallback) -> None:
        key = DispatchKey(key)
        if key in self._fallbacks:
            raise DispatchError(f"fallback already registered for {key.value}")
        self._fallbacks[key] = fallback

    # ---------- introspection ----------
    def schema(self, operator: str) -> OperatorSchema:
        try:
            return self._schemas[operator]
        except KeyError as exc:
            raise DispatchError(f"unknown operator: {operator}") from exc

    def kernel(self, operator: str, key: DispatchKey) -> Optional[Kernel]:
        self.schema(operator)
        return self._kernels[operator].get(DispatchKey(key))

    def registered_keys(self, operator: str) -> tuple[DispatchKey, ...]:
        self.schema(operator)
        table = self._kernels[operator]
        return tuple(key for key in _PRIORITY if key in table)

    def list_operators(self) -> tuple[str, ...]:
        return tuple(sorted(self._schemas))

    # ---------- key inference ----------
    @staticmethod
    def _iter_tensors(value):
        from .tensor import Tensor

        if isinstance(value, Tensor):
            yield value
        elif isinstance(value, (tuple, list)):
            for item in value:
                yield from Dispatcher._iter_tensors(item)
        elif isinstance(value, dict):
            for item in value.values():
                yield from Dispatcher._iter_tensors(item)

    def keyset_for(self, *args, **kwargs) -> DispatchKeySet:
        from .autograd import is_grad_enabled

        tensors = list(self._iter_tensors(args)) + list(self._iter_tensors(kwargs))
        keys: list[DispatchKey] = [DispatchKey.CPU]
        if not is_grad_enabled():
            keys.append(DispatchKey.NOGRAD)
        elif any(t.requires_grad for t in tensors):
            keys.append(DispatchKey.AUTOGRAD)
        return DispatchKeySet(keys)

    # ---------- dispatch ----------
    def _decision(self, operator: str, keyset: DispatchKeySet) -> DispatchDecision:
        self.schema(operator)
        if not keyset:
            raise DispatchError(f"empty dispatch key set for {operator}")

        table = self._kernels[operator]
        for key in keyset:
            if key in table:
                return DispatchDecision(operator, keyset, key, "kernel")
            if key in self._fallbacks:
                return DispatchDecision(operator, keyset, key, "fallback")
            # If a key is explicitly present but has no implementation/fallback,
            # fail rather than silently skipping it. This makes unsupported
            # future backends (for example CUDA in v0.5) obvious.
            raise DispatchError(
                f"no kernel or fallback for {operator} at dispatch key {key.value}; "
                f"available kernels={[k.value for k in self.registered_keys(operator)]}"
            )

        raise DispatchError(f"no dispatchable key for {operator}: {keyset}")

    def explain(self, operator: str, *args, **kwargs) -> DispatchDecision:
        return self._decision(operator, self.keyset_for(*args, **kwargs))

    def dispatch(self, operator: str, *args, **kwargs):
        keyset = self.keyset_for(*args, **kwargs)
        return self.redispatch(operator, keyset, *args, **kwargs)

    def redispatch(self, operator: str, keyset: DispatchKeySet, *args, **kwargs):
        decision = self._decision(operator, keyset)
        key = decision.selected_key
        if decision.implementation == "kernel":
            return self._kernels[operator][key](*args, **kwargs)
        return self._fallbacks[key](operator, keyset, *args, **kwargs)


class Library:
    """Small registration facade inspired by ``torch.library``."""

    def __init__(self, namespace: str, dispatcher: Dispatcher) -> None:
        if not namespace:
            raise ValueError("namespace must be non-empty")
        self.namespace = namespace
        self.dispatcher = dispatcher

    def define(
        self,
        name: str,
        *,
        overload: str = "default",
        signature: str = "",
    ) -> str:
        schema = OperatorSchema(self.namespace, name, overload, signature)
        self.dispatcher.define(schema)
        return schema.qualified_name

    def impl(self, operator: str, key: DispatchKey, kernel: Kernel) -> None:
        if "::" not in operator:
            operator = f"{self.namespace}::{operator}"
        self.dispatcher.register(operator, key, kernel)


dispatcher = Dispatcher()


def call(operator: str, *args, **kwargs):
    """Call a logical operator through the process-global dispatcher."""

    return dispatcher.dispatch(operator, *args, **kwargs)
