"""See the dynamic graph and chain rule in the smallest possible example."""
from mini_pytorch import Tensor
from mini_pytorch.graph import to_dot

x = Tensor(2.0, requires_grad=True, name="x")
y = Tensor(3.0, requires_grad=True, name="y")
z = x * y + x ** 2
z.name = "z"

print("z =", z)
print("z.grad_fn =", z.grad_fn)
z.backward()
print("dz/dx =", x.grad, "(expected 7)")
print("dz/dy =", y.grad, "(expected 2)")

print("\nDOT graph:\n")
print(to_dot(z))
