"""Walk through mini-pytorch v0.8 advanced Autograd Engine concepts."""
from __future__ import annotations

import numpy as np

from mini_pytorch import Function, Tensor


class SquarePlusOne(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * x + 1.0

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return (grad_output * 2.0 * x,)


def graph_lifetime_demo():
    print("\n=== graph lifetime / retain_graph ===")
    x = Tensor(2.0, requires_grad=True)
    loss = (x * x).sum()

    loss.backward(retain_graph=True)
    print("after retained backward, x.grad =", x.grad)

    loss.backward()
    print("after final backward, x.grad =", x.grad, "(accumulated)")

    try:
        loss.backward()
    except RuntimeError as exc:
        print("third backward fails because graph state is released:")
        print(" ", exc)


def retained_grad_and_hook_demo():
    print("\n=== non-leaf retain_grad + gradient hook ===")
    x = Tensor([2.0, 3.0], requires_grad=True)
    hidden = x * x
    hidden.retain_grad()

    def halve_gradient(grad):
        print("hook sees dL/dHidden before transform:", grad)
        return grad * 0.5

    handle = hidden.register_hook(halve_gradient)
    loss = (hidden * 3.0).sum()
    loss.backward()

    print("hidden.grad =", hidden.grad)
    print("x.grad      =", x.grad)
    handle.remove()


def custom_function_demo():
    print("\n=== custom Function on EDU placement ===")
    x = Tensor([2.0, -3.0], requires_grad=True, device="edu")
    y = SquarePlusOne.apply(x)
    print("y.device =", y.device)
    print("y.data   =", y.data)
    y.sum().backward()
    print("x.grad   =", x.grad)


def higher_order_boundary_demo():
    print("\n=== higher-order boundary ===")
    x = Tensor(2.0, requires_grad=True)
    loss = (x * x).sum()
    try:
        loss.backward(create_graph=True)
    except NotImplementedError as exc:
        print("create_graph=True intentionally fails in v0.8:")
        print(" ", exc)

    # The rejected create_graph request does not consume the graph.
    loss.backward()
    np.testing.assert_allclose(x.grad, 4.0)
    print("normal backward still works: x.grad =", x.grad)


if __name__ == "__main__":
    graph_lifetime_demo()
    retained_grad_and_hook_demo()
    custom_function_demo()
    higher_order_boundary_demo()
