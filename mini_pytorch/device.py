"""Device identity for mini-pytorch v0.7."""
from __future__ import annotations

from dataclasses import dataclass


_SUPPORTED_DEVICE_TYPES = {"cpu", "edu"}


@dataclass(frozen=True)
class Device:
    """Small immutable device value object.

    Supported v0.7 device types:
    - ``cpu``: NumPy reference backend
    - ``edu``: simulated educational backend that also uses host NumPy arrays

    The second backend exists to exercise placement and dispatch architecture on
    ordinary CI; it is not a GPU emulator.
    """

    type: str
    index: int | None = None

    def __init__(self, spec: "Device | str" = "cpu", index: int | None = None) -> None:
        if isinstance(spec, Device):
            if index is not None:
                raise ValueError("index must not be supplied when spec is already a Device")
            object.__setattr__(self, "type", spec.type)
            object.__setattr__(self, "index", spec.index)
            return

        if not isinstance(spec, str):
            raise TypeError("device must be a Device or string")
        text = spec.strip().lower()
        if not text:
            raise ValueError("device string must be non-empty")

        parsed_index = None
        if ":" in text:
            device_type, raw_index = text.split(":", 1)
            if not raw_index or not raw_index.isdigit():
                raise ValueError(f"invalid device index in {spec!r}")
            parsed_index = int(raw_index)
        else:
            device_type = text

        if index is not None:
            if parsed_index is not None:
                raise ValueError("device index specified twice")
            parsed_index = int(index)

        if device_type not in _SUPPORTED_DEVICE_TYPES:
            raise ValueError(
                f"unsupported device type {device_type!r}; "
                f"supported={sorted(_SUPPORTED_DEVICE_TYPES)}"
            )
        if parsed_index is not None and parsed_index < 0:
            raise ValueError("device index must be non-negative")

        object.__setattr__(self, "type", device_type)
        object.__setattr__(self, "index", parsed_index)

    def __str__(self) -> str:
        return self.type if self.index is None else f"{self.type}:{self.index}"

    def __repr__(self) -> str:
        return f"device(type={self.type!r}, index={self.index!r})"


def as_device(value: Device | str | None) -> Device:
    return Device("cpu") if value is None else Device(value)
