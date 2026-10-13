import unittest

import numpy as np

from mini_pytorch import Tensor
from mini_pytorch.compiler import (
    ExecutionPlanningError,
    GraphInterpreter,
    LoweredExecutor,
    PlannedExecutor,
    aot_compile,
    build_execution_plan,
    capture,
    lower_graph,
    plan_memory,
)


class TestMemoryPlanning(unittest.TestCase):
    def _pointwise_lowered(self, device="cpu"):
        fn = lambda x, y: (((((x * y) + x).relu() * y) + x).relu())
        graph = capture(
            fn,
            Tensor([1.0, -2.0, 3.0], device=device),
            Tensor([4.0, 5.0, -1.0], device=device),
        ).graph
        return lower_graph(graph)

    def test_linear_pointwise_chain_reuses_exact_spec_slots(self):
        lowered = self._pointwise_lowered()
        memory = plan_memory(lowered.program)

        self.assertGreaterEqual(memory.assigned_value_count, 4)
        self.assertLess(len(memory.slots), memory.assigned_value_count)
        self.assertGreater(memory.reuse_count, 0)
        self.assertGreater(memory.naive_allocation_bytes, memory.planned_buffer_bytes)
        self.assertEqual(
            memory.bytes_avoided_by_reuse,
            memory.naive_allocation_bytes - memory.planned_buffer_bytes,
        )

    def test_slot_assignments_never_overlap(self):
        lowered = self._pointwise_lowered()
        memory = plan_memory(lowered.program)
        by_id = {interval.value_id: interval for interval in memory.intervals}

        for slot in memory.slots:
            assigned = [by_id[value_id] for value_id in slot.value_ids]
            for left, right in zip(assigned, assigned[1:]):
                self.assertFalse(left.overlaps(right))
                self.assertTrue(left.spec.compatible_with(right.spec))

    def test_graph_outputs_are_not_reused(self):
        lowered = self._pointwise_lowered()
        output_id = lowered.program.graph.outputs.value_id
        memory = plan_memory(lowered.program)
        self.assertNotIn(output_id, memory.value_to_slot)
        interval = next(x for x in memory.intervals if x.value_id == output_id)
        self.assertTrue(interval.is_graph_output)
        self.assertFalse(interval.eligible_for_reuse)

    def test_retained_value_is_excluded_from_reuse(self):
        lowered = self._pointwise_lowered()
        initial = plan_memory(lowered.program)
        retained_id = next(iter(initial.value_to_slot))

        retained = plan_memory(lowered.program, retained_value_ids=[retained_id])
        self.assertNotIn(retained_id, retained.value_to_slot)
        interval = next(x for x in retained.intervals if x.value_id == retained_id)
        self.assertTrue(interval.is_retained)
        self.assertFalse(interval.eligible_for_reuse)

    def test_plan_is_deterministic_and_inspectable(self):
        lowered = self._pointwise_lowered()
        first = build_execution_plan(lowered.program)
        second = build_execution_plan(lowered.program)
        self.assertEqual(first.memory_plan, second.memory_plan)
        self.assertEqual(first.steps, second.steps)
        self.assertEqual(first.format(), second.format())
        self.assertIn("MemoryPlan:", first.format())
        self.assertIn("reuse_count=", first.format())

    def test_unknown_retained_value_fails(self):
        lowered = self._pointwise_lowered()
        with self.assertRaisesRegex(ExecutionPlanningError, "do not exist"):
            plan_memory(lowered.program, retained_value_ids=[999])


class TestPlannedExecution(unittest.TestCase):
    def _lowered(self, device="cpu"):
        fn = lambda x, y: (((((x * y) + x).relu() * y) + x).relu()).sum()
        graph = capture(
            fn,
            Tensor([1.0, -2.0, 3.0], device=device),
            Tensor([4.0, 5.0, -1.0], device=device),
        ).graph
        return graph, lower_graph(graph)

    def test_matches_graph_and_lowered_reference_on_cpu_and_edu(self):
        for device in ("cpu", "edu"):
            with self.subTest(device=device):
                graph, lowered = self._lowered(device)
                plan = build_execution_plan(lowered.program)
                x = Tensor([2.0, -1.0, 4.0], device=device)
                y = Tensor([3.0, 2.0, -0.5], device=device)

                graph_out = GraphInterpreter(graph)(x, y)
                lowered_out = LoweredExecutor(lowered.program)(
                    Tensor(x.data.copy(), device=device),
                    Tensor(y.data.copy(), device=device),
                )
                planned_out = PlannedExecutor(plan)(
                    Tensor(x.data.copy(), device=device),
                    Tensor(y.data.copy(), device=device),
                )

                np.testing.assert_allclose(planned_out.data, graph_out.data)
                np.testing.assert_allclose(planned_out.data, lowered_out.data)
                self.assertEqual(str(planned_out.device), device)
                self.assertFalse(planned_out.requires_grad)

    def test_runtime_stats_expose_allocations_reuse_and_release(self):
        _, lowered = self._lowered()
        plan = build_execution_plan(lowered.program)
        result = PlannedExecutor(plan).run_with_stats(
            Tensor([2.0, -1.0, 4.0]),
            Tensor([3.0, 2.0, -0.5]),
        )
        stats = result.stats
        self.assertEqual(stats.slot_allocations, len(plan.memory_plan.slots))
        self.assertEqual(stats.slot_reuses, plan.memory_plan.reuse_count)
        self.assertEqual(stats.planned_buffer_bytes, plan.memory_plan.planned_buffer_bytes)
        self.assertEqual(stats.naive_allocation_bytes, plan.memory_plan.naive_allocation_bytes)
        self.assertGreaterEqual(stats.released_value_count, 1)
        self.assertGreaterEqual(stats.wall_elapsed_ns, 0)

    def test_external_ops_remain_dispatcher_execution(self):
        graph = capture(
            lambda x, y: (x @ y).sum(),
            Tensor(np.arange(6, dtype=float).reshape(2, 3)),
            Tensor(np.arange(6, dtype=float).reshape(3, 2)),
        ).graph
        lowered = lower_graph(graph)
        plan = build_execution_plan(lowered.program)
        self.assertEqual(len(plan.memory_plan.slots), 0)

        x = Tensor(np.arange(6, dtype=float).reshape(2, 3))
        y = Tensor(np.arange(6, dtype=float).reshape(3, 2))
        expected = LoweredExecutor(lowered.program)(x, y)
        actual = PlannedExecutor(plan)(
            Tensor(x.data.copy()),
            Tensor(y.data.copy()),
        )
        np.testing.assert_allclose(actual.data, expected.data)

    def test_fresh_slot_pool_per_run_prevents_cross_run_output_alias(self):
        _, lowered = self._lowered()
        executor = PlannedExecutor(build_execution_plan(lowered.program))
        first = executor(
            Tensor([2.0, -1.0, 4.0]),
            Tensor([3.0, 2.0, -0.5]),
        )
        snapshot = first.data.copy()
        second = executor(
            Tensor([10.0, 20.0, 30.0]),
            Tensor([2.0, 3.0, 4.0]),
        )
        self.assertIsNot(first.storage, second.storage)
        np.testing.assert_allclose(first.data, snapshot)

    def test_selected_value_must_be_retained_by_plan(self):
        _, lowered = self._lowered()
        memory = plan_memory(lowered.program)
        internal_id = next(iter(memory.value_to_slot))
        executor = PlannedExecutor(build_execution_plan(lowered.program))
        with self.assertRaisesRegex(ExecutionPlanningError, "retained"):
            executor.run_with_values(
                Tensor([2.0, -1.0, 4.0]),
                Tensor([3.0, 2.0, -0.5]),
                value_ids=[internal_id],
            )

    def test_retained_internal_value_matches_lowered_reference(self):
        _, lowered = self._lowered()
        memory = plan_memory(lowered.program)
        internal_id = next(iter(memory.value_to_slot))
        plan = build_execution_plan(
            lowered.program,
            retained_value_ids=[internal_id],
        )
        x = Tensor([2.0, -1.0, 4.0])
        y = Tensor([3.0, 2.0, -0.5])
        ref_out, ref_values = LoweredExecutor(lowered.program).run_with_values(
            x,
            y,
            value_ids=[internal_id],
        )
        out, values = PlannedExecutor(plan).run_with_values(
            Tensor(x.data.copy()),
            Tensor(y.data.copy()),
            value_ids=[internal_id],
        )
        np.testing.assert_allclose(out.data, ref_out.data)
        np.testing.assert_allclose(values[0].data, ref_values[0].data)

    def test_aot_saved_values_and_backward_survive_planning(self):
        fn = lambda x, y: (x.relu() * y).sum()
        aot = aot_compile(
            fn,
            Tensor([-2.0, 1.5, 3.0], requires_grad=True),
            Tensor([4.0, -1.0, 2.0], requires_grad=True),
        )
        x_data = np.array([-3.0, 2.0, 5.0])
        y_data = np.array([1.5, -2.0, 0.5])

        reference = aot.forward(
            Tensor(x_data, requires_grad=True),
            Tensor(y_data, requires_grad=True),
        )
        reference_grads = aot.backward(reference.context)

        forward_lowered = lower_graph(aot.forward_graph)
        retained_ids = [
            forward_lowered.mapped_value_id(source_id)
            for source_id in aot.spec.saved_value_ids
        ]
        forward_plan = build_execution_plan(
            forward_lowered.program,
            retained_value_ids=retained_ids,
        )
        lowered_out, saved = PlannedExecutor(forward_plan).run_with_values(
            Tensor(x_data, requires_grad=True),
            Tensor(y_data, requires_grad=True),
            value_ids=retained_ids,
        )
        np.testing.assert_allclose(lowered_out.data, reference.output.data)

        backward_lowered = lower_graph(aot.backward_graph)
        backward_plan = build_execution_plan(backward_lowered.program)
        seed = Tensor(np.ones_like(lowered_out.data), device=lowered_out.device)
        planned_grads = PlannedExecutor(backward_plan)(*saved, seed)
        self.assertIsInstance(planned_grads, tuple)
        self.assertEqual(len(planned_grads), len(reference_grads))
        for got, expected in zip(planned_grads, reference_grads):
            np.testing.assert_allclose(got.data, expected.data)


if __name__ == "__main__":
    unittest.main()
