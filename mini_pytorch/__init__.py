"""mini-pytorch: a small, readable PyTorch-like learning framework."""
from .autograd import no_grad
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
__version__ = "0.5.0"
