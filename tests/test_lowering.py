import unittest

import numpy as np

from mini_pytorch import Tensor
from mini_pytorch.compiler import (
    Graph,
    GraphInterpreter,
    LoweredExecutor,
    LoweringError,
    ValueRef,
    analyze_graph,
    aot_compile,
    capture,
    eliminate_dead_code,
    fold_constants,
    lower_graph,
    normalize_graph,
    plan_fusions,
)


class TestGraphAnalysisAndTransforms(unittest.TestCase):
    def test_analysis_tracks_shared_uses_and_dead_value(self):
        def fn(x):
            dead = x.relu()
            shared = x * x
            return shared + shared

        graph = capture(fn, Tensor([-2.0, 3.0])).graph
        analysis = analyze_graph(graph)

        relu_node = graph.nodes[0]
        mul_node = graph.nodes[1]
        relu_out = relu_node.outputs.value_id
        mul_out = mul_node.outputs.value_id

        self.assertFalse(analysis.value(relu_out).is_live)
        self.assertTrue(analysis.value(mul_out).is_live)
        self.assertEqual(analysis.value(mul_out).use_count, 2)
        self.assertTrue(analysis.value(graph.outputs.value_id).is_graph_output)

    def test_normalization_is_non_mutating_deterministic_and_equivalent(self):
        graph = capture(
            lambda x, y: ((x * y) + x).relu(),
            Tensor([1.0, -2.0]),
            Tensor([3.0, 4.0]),
        ).graph
        before = graph.format()
        n1 = normalize_graph(graph)
        n2 = normalize_graph(graph)

        self.assertEqual(graph.format(), before)
        self.assertEqual(n1.format(), n2.format())

        expected = GraphInterpreter(graph)(Tensor([2.0, -1.0]), Tensor([5.0, 3.0]))
        actual = GraphInterpreter(n1)(Tensor([2.0, -1.0]), Tensor([5.0, 3.0]))
        np.testing.assert_allclose(actual.data, expected.data)

    def test_dce_removes_unreachable_pure_node(self):
        def fn(x):
            dead = x * x
            live = x + x
            return live

        graph = capture(fn, Tensor([1.0, 2.0])).graph
        optimized = eliminate_dead_code(normalize_graph(graph))
        self.assertEqual([node.operator for node in optimized.nodes], ["aten::add.Tensor"])

        expected = GraphInterpreter(graph)(Tensor([3.0, 4.0]))
        actual = GraphInterpreter(optimized)(Tensor([3.0, 4.0]))
        np.testing.assert_allclose(actual.data, expected.data)

    def test_safe_constant_folding_collapses_constant_pointwise_chain(self):
        c1 = Tensor([2.0, -3.0])
        c2 = Tensor([4.0, 5.0])
        graph = capture(
            lambda x: x + (c1 * c2).relu(),
            Tensor([1.0, 1.0]),
        ).graph

        normalized = eliminate_dead_code(normalize_graph(graph))
        folded = fold_constants(normalized)
        self.assertEqual(len(folded.folded_node_ids), 2)
        self.assertEqual([node.operator for node in folded.graph.nodes], ["aten::add.Tensor"])

        expected = GraphInterpreter(graph)(Tensor([10.0, 20.0]))
        actual = GraphInterpreter(folded.graph)(Tensor([10.0, 20.0]))
        np.testing.assert_allclose(actual.data, expected.data)

    def test_grad_requiring_constants_are_not_folded(self):
        c = Tensor([2.0, 3.0], requires_grad=True)
        graph = capture(lambda x: x + (c * c), Tensor([1.0, 1.0])).graph
        folded = fold_constants(normalize_graph(graph))
        self.assertEqual(folded.folded_node_ids, ())
        self.assertEqual(len(folded.graph.nodes), len(graph.nodes))


class TestFusionPlanning(unittest.TestCase):
    def test_pointwise_chain_forms_one_fusion_group(self):
        graph = normalize_graph(
            capture(
                lambda x, y: ((x * y) + x).relu(),
                Tensor([1.0, -2.0]),
                Tensor([3.0, 4.0]),
            ).graph
        )
        groups = plan_fusions(graph)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0].node_ids), 3)
        self.assertEqual(groups[0].node_ids, tuple(node.id for node in graph.nodes))

    def test_shared_branch_prevents_producer_from_being_fused(self):
        def fn(x, y):
            shared = x * y
            left = shared + x
            right = shared * y
            return left + right

        graph = normalize_graph(
            capture(fn, Tensor([1.0, 2.0]), Tensor([3.0, 4.0])).graph
        )
        analysis = analyze_graph(graph)
        shared_output = graph.nodes[0].outputs.value_id
        self.assertEqual(analysis.value(shared_output).use_count, 2)

        groups = plan_fusions(graph, analysis)
        self.assertTrue(all(graph.nodes[0].id not in group.node_ids for group in groups))

    def test_matmul_and_sum_are_fusion_barriers(self):
        graph = normalize_graph(
            capture(
                lambda x, y: (x @ y).relu().sum(),
                Tensor(np.arange(6, dtype=float).reshape(2, 3)),
                Tensor(np.arange(6, dtype=float).reshape(3, 2)),
            ).graph
        )
        self.assertEqual(plan_fusions(graph), ())


class TestLoweredExecution(unittest.TestCase):
    def _pointwise_graph(self, device):
        return capture(
            lambda x, y: ((x * y) + x).relu(),
            Tensor([1.0, -2.0], device=device),
            Tensor([3.0, 4.0], device=device),
        ).graph

    def test_lowered_pointwise_matches_graph_interpreter_on_cpu_and_edu(self):
        for device in ("cpu", "edu"):
            with self.subTest(device=device):
                graph = self._pointwise_graph(device)
                lowered = lower_graph(graph)
                self.assertEqual(lowered.stats().fusion_groups, 1)
                self.assertEqual(lowered.stats().fused_nodes, 3)

                expected = GraphInterpreter(graph)(
                    Tensor([2.0, -1.0], device=device),
                    Tensor([5.0, 3.0], device=device),
                )
                actual = LoweredExecutor(lowered.program)(
                    Tensor([2.0, -1.0], device=device),
                    Tensor([5.0, 3.0], device=device),
                )
                np.testing.assert_allclose(actual.data, expected.data)
                self.assertEqual(str(actual.device), device)
                self.assertFalse(actual.requires_grad)

    def test_external_barrier_nodes_still_execute_through_dispatcher(self):
        graph = capture(
            lambda x, y: (x @ y).relu().sum(),
            Tensor(np.arange(6, dtype=float).reshape(2, 3)),
            Tensor(np.arange(6, dtype=float).reshape(3, 2)),
        ).graph
        lowered = lower_graph(graph)
        self.assertEqual(lowered.stats().fusion_groups, 0)
        self.assertEqual(lowered.stats().external_nodes, 3)

        x = Tensor(np.arange(6, dtype=float).reshape(2, 3))
        y = Tensor(np.arange(6, dtype=float).reshape(3, 2))
        expected = GraphInterpreter(graph)(x, y)
        actual = LoweredExecutor(lowered.program)(
            Tensor(x.data.copy()),
            Tensor(y.data.copy()),
        )
        np.testing.assert_allclose(actual.data, expected.data)

    def test_aot_forward_and_backward_graphs_share_lowering_pipeline(self):
        fn = lambda x, y: (x.relu() * y).sum()
        program = aot_compile(
            fn,
            Tensor([-2.0, 1.5, 3.0], requires_grad=True),
            Tensor([4.0, -1.0, 2.0], requires_grad=True),
        )

        ref_run = program.forward(
            Tensor([-3.0, 2.0, 5.0], requires_grad=True),
            Tensor([1.5, -2.0, 0.5], requires_grad=True),
        )
        ref_grads = program.backward(ref_run.context)

        forward_lower = lower_graph(program.forward_graph)
        lowered_out, saved = LoweredExecutor(forward_lower.program).run_with_values(
            Tensor([-3.0, 2.0, 5.0], requires_grad=True),
            Tensor([1.5, -2.0, 0.5], requires_grad=True),
            value_ids=[
                forward_lower.mapped_value_id(value_id)
                for value_id in program.spec.saved_value_ids
            ],
        )
        np.testing.assert_allclose(lowered_out.data, ref_run.output.data)

        backward_lower = lower_graph(program.backward_graph)
        seed = Tensor(np.ones_like(lowered_out.data), device=lowered_out.device)
        lowered_grads = LoweredExecutor(backward_lower.program)(*saved, seed)
        self.assertIsInstance(lowered_grads, tuple)
        self.assertEqual(len(lowered_grads), len(ref_grads))
        for got, expected in zip(lowered_grads, ref_grads):
            np.testing.assert_allclose(got.data, expected.data)

    def test_stats_and_explain_are_deterministic(self):
        graph = self._pointwise_graph("cpu")
        first = lower_graph(graph)
        second = lower_graph(graph)
        self.assertEqual(first.stats(), second.stats())
        self.assertEqual(first.explain(), second.explain())
        self.assertIn("Fusion groups:", first.explain())
        self.assertIn("LoweringStats(", first.explain())

    def test_executor_input_count_error(self):
        graph = self._pointwise_graph("cpu")
        lowered = lower_graph(graph)
        with self.assertRaisesRegex(ValueError, "expects 2 inputs"):
            LoweredExecutor(lowered.program)(Tensor([1.0, 2.0]))

    def test_fusion_rejects_mixed_runtime_devices(self):
        graph = Graph()
        x = graph.add_input(
            metadata={"shape": (2,), "dtype": "float64", "device": "cpu", "requires_grad": False}
        )
        y = graph.add_input(
            metadata={"shape": (2,), "dtype": "float64", "device": "edu", "requires_grad": False}
        )
        z = graph.add_intermediate(
            metadata={"shape": (2,), "dtype": "float64", "device": "cpu", "requires_grad": False}
        )
        out = graph.add_intermediate(
            metadata={"shape": (2,), "dtype": "float64", "device": "cpu", "requires_grad": False}
        )
        graph.add_node("aten::add.Tensor", (ValueRef(x.id), ValueRef(y.id)), {}, ValueRef(z.id))
        graph.add_node("aten::relu.default", (ValueRef(z.id),), {}, ValueRef(out.id))
        graph.outputs = ValueRef(out.id)

        lowered = lower_graph(graph)
        with self.assertRaisesRegex(LoweringError, "inconsistent devices"):
            LoweredExecutor(lowered.program)(
                Tensor([1.0, 2.0], device="cpu"),
                Tensor([3.0, 4.0], device="edu"),
            )

    def test_malformed_graph_fails_explicitly(self):
        graph = Graph()
        x = graph.add_input(
            metadata={"shape": (2,), "dtype": "float64", "device": "cpu", "requires_grad": False}
        )
        out = graph.add_intermediate(
            metadata={"shape": (2,), "dtype": "float64", "device": "cpu", "requires_grad": False}
        )
        graph.add_node(
            "aten::add.Tensor",
            (ValueRef(x.id), ValueRef(999)),
            {},
            ValueRef(out.id),
        )
        graph.outputs = ValueRef(out.id)

        with self.assertRaisesRegex(LoweringError, "malformed Graph"):
            lower_graph(graph)


if __name__ == "__main__":
    unittest.main()
