"""Checkpoint collection, transparent serialization and restore orchestration."""
from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
import uuid

import numpy as np

from .._version import __version__
from .rng import get_rng_state, set_rng_state
from .state import CHECKPOINT_SCHEMA_VERSION, CheckpointState, TrainingState


FRAMEWORK_VERSION = __version__
_METADATA_FILE = "metadata.json"
_TENSORS_FILE = "tensors.npz"
_FORMAT_NAME = "mini-pytorch-checkpoint"


class CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be parsed or restored safely."""


def _key_sort_value(key: Any) -> tuple[str, str]:
    return (type(key).__name__, repr(key))


def _encode_key(key: Any) -> dict[str, Any]:
    if key is None:
        return {"type": "none", "value": None}
    if isinstance(key, bool):
        return {"type": "bool", "value": key}
    if isinstance(key, (int, np.integer)):
        return {"type": "int", "value": int(key)}
    if isinstance(key, str):
        return {"type": "str", "value": key}
    if isinstance(key, (float, np.floating)):
        value = float(key)
        if not math.isfinite(value):
            raise TypeError("non-finite float dictionary keys are not serializable")
        return {"type": "float", "value": value}
    raise TypeError(f"unsupported checkpoint dictionary key type: {type(key)!r}")


def _decode_key(encoded: dict[str, Any]) -> Any:
    if not isinstance(encoded, dict) or set(encoded) != {"type", "value"}:
        raise CheckpointError("malformed encoded dictionary key")
    kind = encoded["type"]
    value = encoded["value"]
    if kind == "none":
        return None
    if kind == "bool":
        return bool(value)
    if kind == "int":
        return int(value)
    if kind == "str":
        return str(value)
    if kind == "float":
        return float(value)
    raise CheckpointError(f"unknown encoded dictionary key type {kind!r}")


def _encode_value(value: Any, arrays: dict[str, np.ndarray]) -> Any:
    if isinstance(value, np.ndarray):
        name = f"array_{len(arrays):06d}"
        arrays[name] = np.array(value, copy=True)
        return {"__kind__": "ndarray", "name": name}
    if isinstance(value, np.generic):
        return _encode_value(value.item(), arrays)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError("non-finite floats are not supported in checkpoint JSON metadata")
        return value
    if isinstance(value, dict):
        return {
            "__kind__": "dict",
            "items": [
                {
                    "key": _encode_key(key),
                    "value": _encode_value(value[key], arrays),
                }
                for key in sorted(value, key=_key_sort_value)
            ],
        }
    if isinstance(value, tuple):
        return {
            "__kind__": "tuple",
            "items": [_encode_value(item, arrays) for item in value],
        }
    if isinstance(value, list):
        return {
            "__kind__": "list",
            "items": [_encode_value(item, arrays) for item in value],
        }
    raise TypeError(f"unsupported checkpoint value type: {type(value)!r}")


def _decode_value(value: Any, arrays: dict[str, np.ndarray]) -> Any:
    if not isinstance(value, dict) or "__kind__" not in value:
        return value
    kind = value["__kind__"]
    if kind == "ndarray":
        name = value.get("name")
        if not isinstance(name, str) or name not in arrays:
            raise CheckpointError(f"checkpoint references missing tensor array {name!r}")
        return np.array(arrays[name], copy=True)
    if kind == "dict":
        items = value.get("items")
        if not isinstance(items, list):
            raise CheckpointError("malformed encoded dictionary")
        result = {}
        for item in items:
            if not isinstance(item, dict) or set(item) != {"key", "value"}:
                raise CheckpointError("malformed encoded dictionary item")
            key = _decode_key(item["key"])
            if key in result:
                raise CheckpointError(f"duplicate decoded dictionary key {key!r}")
            result[key] = _decode_value(item["value"], arrays)
        return result
    if kind in {"tuple", "list"}:
        items = value.get("items")
        if not isinstance(items, list):
            raise CheckpointError(f"malformed encoded {kind}")
        decoded = [_decode_value(item, arrays) for item in items]
        return tuple(decoded) if kind == "tuple" else decoded
    raise CheckpointError(f"unknown encoded checkpoint value kind {kind!r}")


def _metadata_document(
    state: CheckpointState,
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    state.validate()
    return {
        "format": _FORMAT_NAME,
        "schema_version": state.schema_version,
        "framework_version": state.framework_version,
        "training_state": state.training_state.to_dict(),
        "model_state": _encode_value(state.model_state, arrays),
        "optimizer_state": _encode_value(state.optimizer_state, arrays),
        "scheduler_state": _encode_value(state.scheduler_state, arrays),
        "rng_state": _encode_value(state.rng_state, arrays),
        "metadata": _encode_value(state.metadata, arrays),
    }


def _state_from_document(
    document: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> CheckpointState:
    if not isinstance(document, dict):
        raise CheckpointError("checkpoint metadata root must be a dictionary")
    required = {
        "format",
        "schema_version",
        "framework_version",
        "training_state",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "rng_state",
        "metadata",
    }
    missing = sorted(required - set(document))
    if missing:
        raise CheckpointError(f"checkpoint missing required sections={missing}")
    if document["format"] != _FORMAT_NAME:
        raise CheckpointError(f"unsupported checkpoint format {document['format']!r}")
    try:
        schema_version = int(document["schema_version"])
    except (TypeError, ValueError) as exc:
        raise CheckpointError("checkpoint schema_version must be an integer") from exc
    if schema_version != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointError(
            f"unsupported checkpoint schema version {schema_version}; "
            f"expected {CHECKPOINT_SCHEMA_VERSION}"
        )

    try:
        state = CheckpointState(
            schema_version=schema_version,
            framework_version=document["framework_version"],
            model_state=_decode_value(document["model_state"], arrays),
            optimizer_state=_decode_value(document["optimizer_state"], arrays),
            scheduler_state=_decode_value(document["scheduler_state"], arrays),
            rng_state=_decode_value(document["rng_state"], arrays),
            training_state=TrainingState.from_dict(document["training_state"]),
            metadata=_decode_value(document["metadata"], arrays),
        )
        state.validate()
    except (TypeError, ValueError, KeyError) as exc:
        raise CheckpointError(f"invalid checkpoint state: {exc}") from exc
    return state


def collect_checkpoint_state(
    model,
    optimizer,
    scheduler=None,
    *,
    training_state: TrainingState | None = None,
    metadata: dict[str, Any] | None = None,
    framework_version: str = FRAMEWORK_VERSION,
) -> CheckpointState:
    """Collect logical training state without performing file I/O."""
    training_state = training_state or TrainingState()
    if not isinstance(training_state, TrainingState):
        raise TypeError("training_state must be a TrainingState")
    state = CheckpointState(
        framework_version=framework_version,
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scheduler_state=None if scheduler is None else scheduler.state_dict(),
        rng_state=get_rng_state(),
        training_state=TrainingState.from_dict(training_state.to_dict()),
        metadata=deepcopy(metadata or {}),
    )
    state.validate()
    return state


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())


def _publish_directory(staging: Path, target: Path, overwrite: bool) -> None:
    if target.exists() and not overwrite:
        raise FileExistsError(f"checkpoint path already exists: {target}")
    backup: Path | None = None
    try:
        if target.exists():
            backup = target.parent / f".{target.name}.backup-{uuid.uuid4().hex}"
            os.replace(target, backup)
        os.replace(staging, target)
    except Exception:
        if backup is not None and backup.exists() and not target.exists():
            os.replace(backup, target)
        raise
    else:
        if backup is not None and backup.exists():
            if backup.is_dir():
                shutil.rmtree(backup)
            else:
                backup.unlink()


def save_checkpoint_state(
    path: str | os.PathLike[str],
    state: CheckpointState,
    *,
    overwrite: bool = True,
) -> Path:
    """Serialize logical checkpoint state to ``metadata.json`` + ``tensors.npz``."""
    state.validate()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent))
    )
    try:
        arrays: dict[str, np.ndarray] = {}
        document = _metadata_document(state, arrays)
        _write_json(staging / _METADATA_FILE, document)
        _write_npz(staging / _TENSORS_FILE, arrays)
        _publish_directory(staging, target, overwrite)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def load_checkpoint_state(path: str | os.PathLike[str]) -> CheckpointState:
    """Parse and validate a checkpoint without mutating framework objects."""
    root = Path(path)
    metadata_path = root / _METADATA_FILE
    tensors_path = root / _TENSORS_FILE
    if not root.is_dir():
        raise CheckpointError(f"checkpoint directory does not exist: {root}")
    if not metadata_path.is_file():
        raise CheckpointError(f"checkpoint missing {_METADATA_FILE}")
    if not tensors_path.is_file():
        raise CheckpointError(f"checkpoint missing {_TENSORS_FILE}")
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointError("could not read checkpoint metadata.json") from exc
    try:
        with np.load(tensors_path, allow_pickle=False) as archive:
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    except (OSError, ValueError, EOFError) as exc:
        raise CheckpointError("could not read checkpoint tensors.npz") from exc
    return _state_from_document(document, arrays)


def restore_checkpoint_state(
    state: CheckpointState,
    model,
    optimizer,
    scheduler=None,
    *,
    strict: bool = True,
    restore_rng: bool = True,
) -> TrainingState:
    """Apply validated checkpoint state to freshly constructed training objects."""
    state.validate()
    model.load_state_dict(state.model_state, strict=strict)
    optimizer.load_state_dict(state.optimizer_state)

    if scheduler is not None:
        if state.scheduler_state is None:
            raise CheckpointError(
                "checkpoint has no scheduler state but scheduler restoration was requested"
            )
        scheduler.load_state_dict(state.scheduler_state)
    elif strict and state.scheduler_state is not None:
        raise CheckpointError(
            "checkpoint contains scheduler state but no target scheduler was provided"
        )

    restored_training_state = TrainingState.from_dict(state.training_state.to_dict())
    if restore_rng:
        # Restore RNG last: constructors may have consumed random values.
        set_rng_state(state.rng_state)
    return restored_training_state


def save_checkpoint(
    path: str | os.PathLike[str],
    model,
    optimizer,
    scheduler=None,
    *,
    training_state: TrainingState | None = None,
    metadata: dict[str, Any] | None = None,
    overwrite: bool = True,
) -> CheckpointState:
    state = collect_checkpoint_state(
        model,
        optimizer,
        scheduler,
        training_state=training_state,
        metadata=metadata,
    )
    save_checkpoint_state(path, state, overwrite=overwrite)
    return state


def load_checkpoint(
    path: str | os.PathLike[str],
    model,
    optimizer,
    scheduler=None,
    *,
    strict: bool = True,
    restore_rng: bool = True,
) -> TrainingState:
    state = load_checkpoint_state(path)
    return restore_checkpoint_state(
        state,
        model,
        optimizer,
        scheduler,
        strict=strict,
        restore_rng=restore_rng,
    )
