"""v0.15: profile, plan buffers, execute, and benchmark multiple runtime paths."""
from __future__ import annotations

import numpy as np

from mini_pytorch import Tensor
from mini_pytorch.compiler import (
    BenchmarkConfig,
    GraphInterpreter,
    LoweredExecutor,
    PlannedExecutor,
    benchmark,
    build_execution_plan,
    capture,
    lower_graph,
    profile_lowered,
)


def compare_workload(name, fn, *example_inputs):
    print(f"\n=== {name} ===")

    captured = capture(fn, *example_inputs)
    lowered = lower_graph(captured.graph)
    plan = build_execution_plan(lowered.program)

    graph_executor = GraphInterpreter(captured.graph)
    lowered_executor = LoweredExecutor(lowered.program)
    planned_executor = PlannedExecutor(plan)

    eager_out = fn(*example_inputs)
    graph_out = graph_executor(*example_inputs)
    lowered_out = lowered_executor(*example_inputs)
    planned_result = planned_executor.run_with_stats(*example_inputs)
    planned_out = planned_result.output

    np.testing.assert_allclose(graph_out.data, eager_out.data)
    np.testing.assert_allclose(lowered_out.data, eager_out.data)
    np.testing.assert_allclose(planned_out.data, eager_out.data)

    print("\nLowered program")
    print(lowered.program.format())

    print("\nExecution plan")
    print(plan.format())

    print("\nPlanned execution metrics")
    print(planned_result.stats)

    profile = profile_lowered(lowered.program, *example_inputs)
    print("\nPer-instruction profile")
    print(profile.profile.format())

    # Capture/lowering/planning are intentionally complete before this timing.
    # The numbers are observations from this machine, not speed guarantees.
    config = BenchmarkConfig(warmup=3, repeats=10, iterations_per_repeat=10)
    reports = {
        "eager": benchmark(lambda: fn(*example_inputs), config=config),
        "graph": benchmark(lambda: graph_executor(*example_inputs), config=config),
        "lowered": benchmark(lambda: lowered_executor(*example_inputs), config=config),
        "planned": benchmark(lambda: planned_executor(*example_inputs), config=config),
    }

    print("\nSteady-state benchmark (median per iteration)")
    for executor_name, report in reports.items():
        print(
            f"{executor_name:>8}: {report.median_ms:.6f} ms "
            f"(min={report.min_ns / 1_000_000:.6f}, "
            f"max={report.max_ns / 1_000_000:.6f})"
        )

    print("\nInterpretation")
    print(
        "Lower instruction count / fewer planned buffers are structural facts. "
        "Wall-clock results above are measurements, and PlannedExecutor is not "
        "assumed to be faster on this educational Python/NumPy runtime."
    )


def main():
    x = Tensor(np.linspace(-2.0, 2.0, 1024))
    y = Tensor(np.linspace(0.5, 1.5, 1024))

    compare_workload(
        "pointwise-heavy",
        lambda a, b: (((((a * b) + a).relu() * b) + a).relu()).sum(),
        x,
        y,
    )

    a = Tensor(np.arange(32 * 32, dtype=float).reshape(32, 32) / 100.0)
    b = Tensor(np.eye(32, dtype=float) * 0.5 + 0.1)
    compare_workload(
        "barrier-heavy",
        lambda lhs, rhs: (lhs @ rhs).relu().sum(),
        a,
        b,
    )


if __name__ == "__main__":
    main()
