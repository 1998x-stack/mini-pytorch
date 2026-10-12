"""Builtin operator schemas and registrations for mini-pytorch v0.6."""
from __future__ import annotations

import numpy as np

from .autograd import Context, GradFn
from .dispatcher import DispatchKey, DispatchKeySet, Library, dispatcher
from . import kernels
from .nn_ops import Conv2dOp, MaxPool2dOp
from .ops import Add, MatMul, Mul, ReLU, Sum
from .tensor import Tensor


_REGISTERED = False


def _tensor_result(array, *inputs: Tensor) -> Tensor:
    out = np.asarray(array)
    return Tensor._from_operation_result(
        out,
        inputs=tuple(inputs),
        requires_grad=False,
    )


def _cpu_add(a: Tensor, b: Tensor) -> Tensor:
    return _tensor_result(kernels.add(a.data, b.data), a, b)


def _cpu_mul(a: Tensor, b: Tensor) -> Tensor:
    return _tensor_result(kernels.mul(a.data, b.data), a, b)


def _cpu_matmul(a: Tensor, b: Tensor) -> Tensor:
    return _tensor_result(kernels.matmul(a.data, b.data), a, b)


def _cpu_relu(a: Tensor) -> Tensor:
    return _tensor_result(kernels.relu(a.data), a)


def _cpu_sum(a: Tensor, *, axis=None, keepdims: bool = False) -> Tensor:
    return _tensor_result(kernels.sum(a.data, axis=axis, keepdims=keepdims), a)


def _cpu_conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor,
    *,
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    groups: int = 1,
    implementation: str = "reference",
) -> Tensor:
    if implementation == "reference":
        array = kernels.conv2d_reference(
            input.data,
            weight.data,
            bias.data,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
    elif implementation == "im2col":
        array = kernels.conv2d_im2col(
            input.data,
            weight.data,
            bias.data,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
    else:
        raise ValueError("conv2d implementation must be 'reference' or 'im2col'")
    return _tensor_result(array, input, weight, bias)


def _cpu_max_pool2d(
    input: Tensor,
    *,
    kernel_size,
    stride=None,
    padding=0,
) -> Tensor:
    return _tensor_result(
        kernels.max_pool2d(
            input.data,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ),
        input,
    )


def _attach_autograd(
    operator: str,
    function_cls,
    *inputs: Tensor,
    **kwargs,
) -> Tensor:
    """Autograd wrapper that redispatches to the backend, then attaches GradFn."""

    ctx = Context()
    raw_inputs = tuple(t.data for t in inputs)
    ctx._bind_inputs(tuple(inputs), raw_inputs)

    setup = getattr(function_cls, "setup_context", None)
    if setup is None:
        raise RuntimeError(f"{function_cls.__name__} must define setup_context for dispatcher Autograd")
    setup(ctx, *raw_inputs, **kwargs)

    keyset = dispatcher.keyset_for(*inputs, **kwargs).without(DispatchKey.AUTOGRAD)
    out = dispatcher.redispatch(operator, keyset, *inputs, **kwargs)
    out.requires_grad = True
    out._grad_fn = GradFn(function=function_cls, parents=tuple(inputs), ctx=ctx)
    return out


def _autograd_add(a: Tensor, b: Tensor) -> Tensor:
    return _attach_autograd("aten::add.Tensor", Add, a, b)


def _autograd_mul(a: Tensor, b: Tensor) -> Tensor:
    return _attach_autograd("aten::mul.Tensor", Mul, a, b)


def _autograd_matmul(a: Tensor, b: Tensor) -> Tensor:
    return _attach_autograd("aten::matmul.default", MatMul, a, b)


def _autograd_relu(a: Tensor) -> Tensor:
    return _attach_autograd("aten::relu.default", ReLU, a)


def _autograd_sum(a: Tensor, *, axis=None, keepdims: bool = False) -> Tensor:
    return _attach_autograd(
        "aten::sum.default",
        Sum,
        a,
        axis=axis,
        keepdims=keepdims,
    )


def _autograd_conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor,
    *,
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    groups: int = 1,
    implementation: str = "reference",
) -> Tensor:
    return _attach_autograd(
        "aten::conv2d.default",
        Conv2dOp,
        input,
        weight,
        bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        implementation=implementation,
    )


def _autograd_max_pool2d(
    input: Tensor,
    *,
    kernel_size,
    stride=None,
    padding=0,
) -> Tensor:
    return _attach_autograd(
        "aten::max_pool2d.default",
        MaxPool2dOp,
        input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )


def _no_grad_fallback(
    operator: str,
    keyset: DispatchKeySet,
    *args,
    **kwargs,
):
    """Strip NoGrad and redispatch to the backend numerical kernel."""

    return dispatcher.redispatch(
        operator,
        keyset.without(DispatchKey.NOGRAD),
        *args,
        **kwargs,
    )


def register_builtin_operators() -> None:
    global _REGISTERED
    if _REGISTERED:
        return

    lib = Library("aten", dispatcher)

    schemas = {
        "aten::add.Tensor": lib.define(
            "add",
            overload="Tensor",
            signature="(Tensor self, Tensor other) -> Tensor",
        ),
        "aten::mul.Tensor": lib.define(
            "mul",
            overload="Tensor",
            signature="(Tensor self, Tensor other) -> Tensor",
        ),
        "aten::matmul.default": lib.define(
            "matmul",
            signature="(Tensor self, Tensor other) -> Tensor",
        ),
        "aten::relu.default": lib.define(
            "relu",
            signature="(Tensor self) -> Tensor",
        ),
        "aten::sum.default": lib.define(
            "sum",
            signature="(Tensor self, axis=None, keepdims=False) -> Tensor",
        ),
        "aten::conv2d.default": lib.define(
            "conv2d",
            signature=(
                "(Tensor input, Tensor weight, Tensor bias, stride=(1,1), "
                "padding=(0,0), dilation=(1,1), groups=1, implementation='reference') -> Tensor"
            ),
        ),
        "aten::max_pool2d.default": lib.define(
            "max_pool2d",
            signature="(Tensor input, kernel_size, stride=None, padding=0) -> Tensor",
        ),
    }
    assert all(key == value for key, value in schemas.items())

    dispatcher.register("aten::add.Tensor", DispatchKey.CPU, _cpu_add)
    dispatcher.register("aten::add.Tensor", DispatchKey.AUTOGRAD, _autograd_add)

    dispatcher.register("aten::mul.Tensor", DispatchKey.CPU, _cpu_mul)
    dispatcher.register("aten::mul.Tensor", DispatchKey.AUTOGRAD, _autograd_mul)

    dispatcher.register("aten::matmul.default", DispatchKey.CPU, _cpu_matmul)
    dispatcher.register("aten::matmul.default", DispatchKey.AUTOGRAD, _autograd_matmul)

    dispatcher.register("aten::relu.default", DispatchKey.CPU, _cpu_relu)
    dispatcher.register("aten::relu.default", DispatchKey.AUTOGRAD, _autograd_relu)

    dispatcher.register("aten::sum.default", DispatchKey.CPU, _cpu_sum)
    dispatcher.register("aten::sum.default", DispatchKey.AUTOGRAD, _autograd_sum)

    dispatcher.register("aten::conv2d.default", DispatchKey.CPU, _cpu_conv2d)
    dispatcher.register("aten::conv2d.default", DispatchKey.AUTOGRAD, _autograd_conv2d)

    dispatcher.register("aten::max_pool2d.default", DispatchKey.CPU, _cpu_max_pool2d)
    dispatcher.register("aten::max_pool2d.default", DispatchKey.AUTOGRAD, _autograd_max_pool2d)

    dispatcher.register_fallback(DispatchKey.NOGRAD, _no_grad_fallback)
    _REGISTERED = True
