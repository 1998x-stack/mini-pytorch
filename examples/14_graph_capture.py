"""v0.11: capture eager Dispatcher operations into Graph IR and replay them."""
from __future__ import annotations

import numpy as np

from mini_pytorch import Tensor
from mini_pytorch.compiler import GraphInterpreter, capture


def program(x: Tensor, y: Tensor) -> Tensor:
    return (((x * y) + x).relu()).sum()


def main() -> None:
    trace_x = Tensor([1.0, -2.0], requires_grad=True)
    trace_y = Tensor([3.0, 4.0], requires_grad=True)

    captured = capture(program, trace_x, trace_y)

    print("=== Captured Graph IR ===")
    print(captured.graph.format())
    print()
    print("trace eager output:", captured.eager_output.item())

    x = Tensor([2.0, -1.0], requires_grad=True)
    y = Tensor([4.0, 5.0], requires_grad=True)
    eager = program(x, y)
    eager.backward()

    replay_x = Tensor([2.0, -1.0], requires_grad=True)
    replay_y = Tensor([4.0, 5.0], requires_grad=True)
    replay = GraphInterpreter(captured.graph)(replay_x, replay_y)
    replay.backward()

    print("\n=== Replay equivalence ===")
    print("eager output :", eager.item())
    print("replay output:", replay.item())
    print("eager dx     :", x.grad)
    print("replay dx    :", replay_x.grad)
    print("eager dy     :", y.grad)
    print("replay dy    :", replay_y.grad)

    np.testing.assert_allclose(replay.data, eager.data)
    np.testing.assert_allclose(replay_x.grad, x.grad)
    np.testing.assert_allclose(replay_y.grad, y.grad)
    print("\nforward/backward equivalence: OK")


if __name__ == "__main__":
    main()
