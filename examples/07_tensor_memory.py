"""Explore Storage, stride, views, contiguous layout and version counters."""
from __future__ import annotations

import numpy as np

from mini_pytorch import Tensor, no_grad


def describe(name: str, tensor: Tensor) -> None:
    print(
        f"{name}: shape={tensor.shape}, stride={tensor.stride()}, "
        f"offset={tensor.storage_offset}, contiguous={tensor.is_contiguous()}, "
        f"storage_id={id(tensor.storage)}, version={tensor._version}"
    )
    print(tensor.data)
    print()


def main() -> None:
    print("=== 1. A contiguous Tensor ===")
    x = Tensor(np.arange(12).reshape(3, 4))
    describe("x", x)

    print("=== 2. transpose(): new Tensor, same Storage ===")
    xt = x.transpose(0, 1)
    describe("xt", xt)
    print("x.storage is xt.storage:", x.storage is xt.storage)
    print()

    print("=== 3. slicing changes offset/stride without copying ===")
    sliced = x[1:, 1:]
    describe("sliced", sliced)
    print("x.storage is sliced.storage:", x.storage is sliced.storage)
    print()

    print("=== 4. view() requires metadata-only compatibility ===")
    flat = x.view(12)
    describe("x.view(12)", flat)
    try:
        xt.view(12)
    except RuntimeError as exc:
        print("xt.view(12) failed as expected:")
        print(" ", exc)
    print()

    print("=== 5. reshape() may copy ===")
    reshaped = xt.reshape(12)
    describe("xt.reshape(12)", reshaped)
    print("xt.storage is reshaped.storage:", xt.storage is reshaped.storage)
    print()

    print("=== 6. contiguous() materializes canonical layout ===")
    contiguous = xt.contiguous()
    describe("xt.contiguous()", contiguous)
    print("xt.storage is contiguous.storage:", xt.storage is contiguous.storage)
    print()

    print("=== 7. aliases share a version counter ===")
    alias = x.T
    before = x._version
    alias.add_(100)
    print("version before:", before)
    print("version after: ", x._version)
    print("x now observes the mutation through alias:")
    print(x.data)
    print()

    print("=== 8. leaf requires_grad inplace policy ===")
    p = Tensor([1.0, 2.0], requires_grad=True)
    try:
        p.add_(1.0)
    except RuntimeError as exc:
        print("tracked inplace while grad enabled failed as expected:")
        print(" ", exc)
    with no_grad():
        p.add_(1.0)
    print("after no_grad() update:", p.data)
    print()

    print("=== 9. saved-value mutation invalidates backward ===")
    a = Tensor([2.0, 3.0], requires_grad=True)
    loss = (a * a).sum()
    detached_alias = a.detach()
    detached_alias.add_(1.0)
    try:
        loss.backward()
    except RuntimeError as exc:
        print("backward failed as expected:")
        print(" ", exc)


if __name__ == "__main__":
    main()
