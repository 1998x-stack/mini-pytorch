"""v0.14: inspect graph lowering, pointwise fusion, and AOT backward lowering."""
import numpy as np

from mini_pytorch import Tensor
from mini_pytorch.compiler import (
    GraphInterpreter,
    LoweredExecutor,
    aot_compile,
    capture,
    lower_graph,
)


def pointwise(x, y):
    return ((x * y) + x).relu().sum()


x_example = Tensor([1.0, -2.0, 3.0])
y_example = Tensor([4.0, 5.0, -1.0])
captured = capture(pointwise, x_example, y_example)

print("=== HIGH-LEVEL GRAPH ===")
print(captured.graph.format())

lowered = lower_graph(captured.graph)
print("\n=== OPTIMIZED GRAPH ===")
print(lowered.optimized_graph.format())
print("\n=== LOWERED PROGRAM ===")
print(lowered.program.format())
print("\n=== LOWERING EXPLAIN ===")
print(lowered.explain())

x = Tensor([2.0, -1.0, 4.0])
y = Tensor([3.0, 2.0, -0.5])
reference = GraphInterpreter(captured.graph)(x, y)
lowered_out = LoweredExecutor(lowered.program)(
    Tensor(x.data.copy()),
    Tensor(y.data.copy()),
)
print("\nreference output:", reference.data)
print("lowered output:  ", lowered_out.data)

print("\n=== AOT FORWARD + BACKWARD LOWERING ===")
aot = aot_compile(
    pointwise,
    Tensor([1.0, -2.0, 3.0], requires_grad=True),
    Tensor([4.0, 5.0, -1.0], requires_grad=True),
)

forward_lowered = lower_graph(aot.forward_graph)
backward_lowered = lower_graph(aot.backward_graph)

forward_out, saved = LoweredExecutor(forward_lowered.program).run_with_values(
    Tensor([2.0, -1.0, 4.0], requires_grad=True),
    Tensor([3.0, 2.0, -0.5], requires_grad=True),
    value_ids=[
        forward_lowered.mapped_value_id(value_id)
        for value_id in aot.spec.saved_value_ids
    ],
)
seed = Tensor(np.ones_like(forward_out.data), device=forward_out.device)
grads = LoweredExecutor(backward_lowered.program)(*saved, seed)

print("forward lowered stats:", forward_lowered.stats().format())
print("backward lowered stats:", backward_lowered.stats().format())
print("forward output:", forward_out.data)
for index, grad in enumerate(grads):
    print(f"grad[{index}] =", grad.data)
