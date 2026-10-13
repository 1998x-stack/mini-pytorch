"""mini-pytorch: a small, readable PyTorch-like learning framework."""
from ._version import __version__
from .autograd import Context, Function, GradFn, RemovableHandle, no_grad
from .autograd_engine import AutogradEngine, engine
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
from . import runtime
from . import distributed
from . import compiler
from .runtime import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointError,
    CheckpointState,
    TrainingState,
    collect_checkpoint_state,
    get_rng_state,
    load_checkpoint,
    load_checkpoint_state,
    manual_seed,
    restore_checkpoint_state,
    save_checkpoint,
    save_checkpoint_state,
    set_rng_state,
)

register_builtin_operators()

__all__ = [
    "Tensor",
    "Storage",
    "Device",
    "Backend",
    "backend_for",
    "list_backends",
    "Context",
    "Function",
    "GradFn",
    "RemovableHandle",
    "AutogradEngine",
    "engine",
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
    "runtime",
    "distributed",
    "compiler",
    "manual_seed",
    "get_rng_state",
    "set_rng_state",
    "TrainingState",
    "CheckpointState",
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointError",
    "collect_checkpoint_state",
    "save_checkpoint_state",
    "load_checkpoint_state",
    "restore_checkpoint_state",
    "save_checkpoint",
    "load_checkpoint",
    "__version__",
]
