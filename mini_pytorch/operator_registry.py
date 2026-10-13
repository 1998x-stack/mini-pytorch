"""Builtin operator schemas and backend registrations for mini-pytorch."""
from __future__ import annotations

import numpy as np

from .autograd import Context, GradFn
from .dispatcher import DispatchKey, DispatchKeySet, Library, dispatcher
from . import aot_kernels, kernels
from .nn_ops import Conv2dOp, MaxPool2dOp
from .ops import Add, MatMul, Mul, ReLU, Sum
from .tensor import Tensor


_REGISTERED = False


def _tensor_result(array, *inputs: Tensor) -> Tensor:
    return Tensor._from_operation_result(
        np.asarray(array),
        inputs=tuple(inputs),
        requires_grad=False,
    )


# CPU and EDU intentionally share NumPy numerical implementations. They are
# still registered under distinct DispatchKeys so backend routing is explicit.
def _backend_add(a: Tensor, b: Tensor) -> Tensor:
    return _tensor_result(kernels.add(a.data, b.data), a, b)


def _backend_mul(a: Tensor, b: Tensor) -> Tensor:
    return _tensor_result(kernels.mul(a.data, b.data), a, b)


def _backend_matmul(a: Tensor, b: Tensor) -> Tensor:
    return _tensor_result(kernels.matmul(a.data, b.data), a, b)


def _backend_relu(a: Tensor) -> Tensor:
    return _tensor_result(kernels.relu(a.data), a)


def _backend_sum(a: Tensor, *, axis=None, keepdims: bool = False) -> Tensor:
    return _tensor_result(kernels.sum(a.data, axis=axis, keepdims=keepdims), a)


def _backend_conv2d(
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


def _backend_max_pool2d(input: Tensor, *, kernel_size, stride=None, padding=0) -> Tensor:
    return _tensor_result(
        kernels.max_pool2d(
            input.data,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ),
        input,
    )


# v0.13 AOT helper operators. These are first-order backend kernels. They are
# intentionally executed under no_grad by AOTProgram.backward; no higher-order
# Autograd support is claimed.
def _backend_aot_unbroadcast(grad: Tensor, *, shape) -> Tensor:
    return _tensor_result(aot_kernels.unbroadcast(grad.data, shape), grad)


def _backend_aot_sum_backward(grad: Tensor, *, input_shape, axis=None, keepdims=False) -> Tensor:
    return _tensor_result(
        aot_kernels.sum_backward(
            grad.data,
            input_shape,
            axis=axis,
            keepdims=keepdims,
        ),
        grad,
    )


def _backend_aot_transpose2d(input: Tensor) -> Tensor:
    return _tensor_result(aot_kernels.transpose2d(input.data), input)


def _backend_aot_relu_backward(grad: Tensor, input: Tensor) -> Tensor:
    return _tensor_result(aot_kernels.relu_backward(grad.data, input.data), grad, input)


def _attach_autograd(operator: str, function_cls, *inputs: Tensor, **kwargs) -> Tensor:
    """Autograd policy wrapper that preserves the concrete backend on redispatch."""

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
    return _attach_autograd("aten::sum.default", Sum, a, axis=axis, keepdims=keepdims)


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


def _autograd_max_pool2d(input: Tensor, *, kernel_size, stride=None, padding=0) -> Tensor:
    return _attach_autograd(
        "aten::max_pool2d.default",
        MaxPool2dOp,
        input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )


def _no_grad_fallback(operator: str, keyset: DispatchKeySet, *args, **kwargs):
    return dispatcher.redispatch(
        operator,
        keyset.without(DispatchKey.NOGRAD),
        *args,
        **kwargs,
    )


def _register_backend_pair(operator: str, kernel) -> None:
    dispatcher.register(operator, DispatchKey.CPU, kernel)
    dispatcher.register(operator, DispatchKey.EDU, kernel)


def register_builtin_operators() -> None:
    global _REGISTERED
    if _REGISTERED:
        return

    lib = Library("aten", dispatcher)
    schemas = {
        "aten::add.Tensor": lib.define(
            "add", overload="Tensor", signature="(Tensor self, Tensor other) -> Tensor"
        ),
        "aten::mul.Tensor": lib.define(
            "mul", overload="Tensor", signature="(Tensor self, Tensor other) -> Tensor"
        ),
        "aten::matmul.default": lib.define(
            "matmul", signature="(Tensor self, Tensor other) -> Tensor"
        ),
        "aten::relu.default": lib.define("relu", signature="(Tensor self) -> Tensor"),
        "aten::sum.default": lib.define(
            "sum", signature="(Tensor self, axis=None, keepdims=False) -> Tensor"
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

    for operator, backend_kernel, autograd_kernel in (
        ("aten::add.Tensor", _backend_add, _autograd_add),
        ("aten::mul.Tensor", _backend_mul, _autograd_mul),
        ("aten::matmul.default", _backend_matmul, _autograd_matmul),
        ("aten::relu.default", _backend_relu, _autograd_relu),
        ("aten::sum.default", _backend_sum, _autograd_sum),
        ("aten::conv2d.default", _backend_conv2d, _autograd_conv2d),
        ("aten::max_pool2d.default", _backend_max_pool2d, _autograd_max_pool2d),
    ):
        _register_backend_pair(operator, backend_kernel)
        dispatcher.register(operator, DispatchKey.AUTOGRAD, autograd_kernel)

    aot_lib = Library("aot", dispatcher)
    aot_schemas = {
        "aot::unbroadcast.default": aot_lib.define(
            "unbroadcast", signature="(Tensor grad, shape) -> Tensor"
        ),
        "aot::sum_backward.default": aot_lib.define(
            "sum_backward",
            signature="(Tensor grad, input_shape, axis=None, keepdims=False) -> Tensor",
        ),
        "aot::transpose2d.default": aot_lib.define(
            "transpose2d", signature="(Tensor input) -> Tensor"
        ),
        "aot::relu_backward.default": aot_lib.define(
            "relu_backward", signature="(Tensor grad, Tensor input) -> Tensor"
        ),
    }
    assert all(key == value for key, value in aot_schemas.items())
    for operator, kernel in (
        ("aot::unbroadcast.default", _backend_aot_unbroadcast),
        ("aot::sum_backward.default", _backend_aot_sum_backward),
        ("aot::transpose2d.default", _backend_aot_transpose2d),
        ("aot::relu_backward.default", _backend_aot_relu_backward),
    ):
        _register_backend_pair(operator, kernel)

    dispatcher.register_fallback(DispatchKey.NOGRAD, _no_grad_fallback)
    _REGISTERED = True
