"""Explore Device, backend-owned Storage and device-aware dispatch in v0.7."""
from __future__ import annotations

import numpy as np

from mini_pytorch import Device, Tensor, dispatcher, functional, list_backends, nn


def show(label, tensor):
    print(
        f"{label}: device={tensor.device}, shape={tensor.shape}, "
        f"storage_id={id(tensor.storage)}, grad_fn={tensor.grad_fn}"
    )
    print(tensor.data)


def main():
    print("registered backends:", list_backends())
    print("devices:", Device("cpu"), Device("edu"))
    print()

    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    show("x(cpu)", x)

    y = x.to("edu")
    show("y=x.to('edu')", y)
    print("shares storage with x?", y.storage is x.storage)
    print()

    decision = dispatcher.explain("aten::add.Tensor", y, y)
    print("EDU add dispatch:", decision)
    z = (y * y + 1.0).sum()
    show("z", z)
    z.backward()
    print("gradient arrived back on original CPU x:")
    print(x.grad)
    print()

    cpu = Tensor([1.0])
    edu = Tensor([2.0], device="edu")
    try:
        _ = cpu + edu
    except Exception as exc:
        print("mixed-device error:", type(exc).__name__, exc)
    print()

    model = nn.Sequential(
        nn.Conv2d(1, 2, 3, padding=1),
        nn.BatchNorm2d(2),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )
    model.to("edu")
    print("module parameter devices:", [str(p.device) for p in model.parameters()])
    print("module buffer devices:", [str(b.device) for _, b in model.named_buffers()])

    image = Tensor(np.arange(16.0).reshape(1, 1, 4, 4), device="edu")
    output = model(image)
    show("CNN output on EDU", output)

    weight = Tensor(np.ones((1, 1, 3, 3)), device="edu", requires_grad=True)
    conv = functional.conv2d(image, weight, padding=1, implementation="im2col")
    print("conv dispatch backend:", dispatcher.explain(
        "aten::conv2d.default",
        image,
        weight,
        Tensor(np.zeros(1), device="edu"),
        stride=(1, 1),
        padding=(1, 1),
        dilation=(1, 1),
        groups=1,
        implementation="im2col",
    ).selected_key)
    show("functional conv", conv)


if __name__ == "__main__":
    main()
