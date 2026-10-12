"""mini-pytorch: a small, readable PyTorch-like learning framework."""
from .autograd import no_grad
from .device import Device
from .backend import Backend, backend_for, list_backends
from .tensor import Tensor
from .storage import Storage
from .dispatcher import (
    DispatchError,
    DispatchKey,
    DispatchKeySet,
    Dispatcher,
    Library,
    OperatorSchema,
    dispatcher,
)
from .operator_registry import register_builtin_operators
from . import functional
from . import nn
from . import optim
from . import data

register_builtin_operators()

__all__ = [
    "Tensor",
    "Storage",
    "Device",
    "Backend",
    "backend_for",
    "list_backends",
    "no_grad",
    "DispatchError",
    "DispatchKey",
    "DispatchKeySet",
    "Dispatcher",
    "Library",
    "OperatorSchema",
    "dispatcher",
    "functional",
    "nn",
    "optim",
    "data",
]
__version__ = "0.7.0"
