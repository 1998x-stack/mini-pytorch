"""v0.13: inspect an explicit forward Graph and its generated backward Graph."""
import numpy as np

from mini_pytorch import Tensor
from mini_pytorch.compiler import aot_compile


def fn(x, w):
    hidden = (x @ w).relu()
    return (hidden * hidden).sum()


x_example = Tensor(
    np.array([[1.0, -2.0, 3.0], [0.5, 1.0, -1.0]]),
    requires_grad=True,
)
w_example = Tensor(
    np.array([[1.0, 0.0], [0.5, -1.0], [2.0, 1.5]]),
    requires_grad=True,
)

program = aot_compile(fn, x_example, w_example)

print(program.format())

x = Tensor(np.array([[2.0, -1.0, 1.0], [1.0, 0.5, -2.0]]), requires_grad=True)
w = Tensor(np.array([[1.0, 0.0], [0.5, -1.0], [2.0, 1.5]]), requires_grad=True)

forward = program.forward(x, w)
dx, dw = program.backward(forward.context)

print("\noutput:", forward.output.data)
print("dx:\n", dx.data)
print("dw:\n", dw.data)
print("saved forward value ids:", program.spec.saved_value_ids)
