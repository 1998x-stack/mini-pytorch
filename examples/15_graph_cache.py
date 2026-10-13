"""v0.12: inspect guarded graph reuse and recompilation."""
import numpy as np

from mini_pytorch import Tensor, no_grad
from mini_pytorch.compiler import compile_graph


def fn(x):
    return ((x * 2.0) + x).relu().sum()


compiled = compile_graph(fn)

print("=== specialization A ===")
compiled(Tensor(np.ones((2, 3)), requires_grad=True))
print(compiled.stats())

print("\n=== same signature: cache hit ===")
x = Tensor(np.full((2, 3), 2.0), requires_grad=True)
out = compiled(x)
out.backward()
print("output:", out.data)
print("grad shape:", x.grad.shape)
print(compiled.stats())

print("\n=== shape change: compile specialization B ===")
compiled(Tensor(np.ones((4, 3)), requires_grad=True))
print(compiled.stats())

print("\n=== original shape: hit specialization A again ===")
compiled(Tensor(np.ones((2, 3)), requires_grad=True))
print(compiled.stats())

print("\n=== explain an unseen shape ===")
print(compiled.explain(Tensor(np.ones((8, 3)), requires_grad=True)).format())

print("\n=== grad mode is a separate specialization ===")
with no_grad():
    compiled(Tensor(np.ones((2, 3)), requires_grad=True))
print(compiled.stats())

print("\n=== cached guards ===")
for entry in compiled.cache.entries:
    print(f"\ncompile_id={entry.compile_id}, hits={entry.hit_count}")
    print(entry.guards.format())
