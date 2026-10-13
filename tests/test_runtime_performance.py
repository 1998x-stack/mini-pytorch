import unittest

import numpy as np

from mini_pytorch import Tensor
from mini_pytorch.compiler import (
    BenchmarkConfig,
    ExternalOp,
    Graph,
    LoweredExecutor,
    LoweredProgram,
    PointwiseFusion,
    RuntimeAnalysisError,
    ValueRef,
    analyze_value_lifetimes,
    aot_compile,
    benchmark,
    capture,
    instruction_produced_value_ids,
    lower_graph,
    profile_lowered,
)


class TestBenchmarkPrimitives(unittest.TestCase):
    def test_config_validation(self):
        with self.assertRaisesRegex(ValueError, "warmup"):
            BenchmarkConfig(warmup=-1)
        with self.assertRaisesRegex(ValueError, "repeats"):
            BenchmarkConfig(repeats=0)
        with self.assertRaisesRegex(ValueError, "iterations_per_repeat"):
            BenchmarkConfig(iterations_per_repeat=0)
        with self.assertRaisesRegex(TypeError, "warmup"):
            BenchmarkConfig(warmup=True)

    def test_warmup_and_repeat_call_counts_are_exact(self):
        calls = []

        def fn(x):
            calls.append(x)
            return x + 1

        config = BenchmarkConfig(warmup=2, repeats=3, iterations_per_repeat=4)
        report = benchmark(fn, 7, config=config)

        self.assertEqual(len(calls), 2 + 3 * 4)
        self.assertEqual(len(report.samples_ns), 3)
        self.assertEqual(report.config, config)

    def test_report_statistics_are_self_consistent_without_speed_thresholds(self):
        report = benchmark(
            lambda: sum(range(10)),
            config=BenchmarkConfig(warmup=1, repeats=5, iterations_per_repeat=2),
        )
        self.assertGreaterEqual(report.min_ns, 0.0)
        self.assertGreaterEqual(report.max_ns, report.min_ns)
        self.assertGreaterEqual(report.median_ns, report.min_ns)
        self.assertLessEqual(report.median_ns, report.max_ns)
        self.assertGreaterEqual(report.mean_ns, report.min_ns)
        self.assertLessEqual(report.mean_ns, report.max_ns)
        self.assertIn("median_ms=", report.format())


class TestLoweredProfiler(unittest.TestCase):
    def _lowered(self, device="cpu"):
        captured = capture(
            lambda x, y: ((x * y) + x).relu().sum(),
            Tensor([1.0, -2.0, 3.0], device=device),
            Tensor([4.0, 5.0, -1.0], device=device),
        )
        return lower_graph(captured.graph)

    def test_profiled_execution_matches_ordinary_lowered_execution(self):
        for device in ("cpu", "edu"):
            with self.subTest(device=device):
                lowered = self._lowered(device)
                x = Tensor([2.0, -1.0, 4.0], device=device)
                y = Tensor([3.0, 2.0, -0.5], device=device)
                expected = LoweredExecutor(lowered.program)(x, y)
                profiled = profile_lowered(
                    lowered.program,
                    Tensor(x.data.copy(), device=device),
                    Tensor(y.data.copy(), device=device),
                )
                np.testing.assert_allclose(profiled.output.data, expected.data)
                self.assertEqual(str(profiled.output.device), device)

    def test_profile_has_one_ordered_event_per_lowered_instruction(self):
        lowered = self._lowered()
        result = profile_lowered(
            lowered.program,
            Tensor([2.0, -1.0, 4.0]),
            Tensor([3.0, 2.0, -0.5]),
        )
        events = result.profile.events

        self.assertEqual(len(events), len(lowered.program.instructions))
        self.assertEqual(
            tuple(event.instruction_index for event in events),
            tuple(range(len(events))),
        )
        self.assertIsInstance(lowered.program.instructions[0], PointwiseFusion)
        self.assertIsInstance(lowered.program.instructions[1], ExternalOp)
        self.assertEqual(events[0].instruction_kind, "PointwiseFusion")
        self.assertEqual(events[1].instruction_kind, "ExternalOp")
        self.assertTrue(all(event.elapsed_ns >= 0 for event in events))
        self.assertGreaterEqual(result.profile.wall_elapsed_ns, 0)
        self.assertEqual(
            result.profile.instruction_elapsed_ns,
            sum(event.elapsed_ns for event in events),
        )

    def test_profile_value_ids_match_instruction_boundaries(self):
        lowered = self._lowered()
        result = profile_lowered(
            lowered.program,
            Tensor([2.0, -1.0, 4.0]),
            Tensor([3.0, 2.0, -0.5]),
        )
        fusion = lowered.program.instructions[0]
        reduction = lowered.program.instructions[1]
        self.assertIsInstance(fusion, PointwiseFusion)
        self.assertIsInstance(reduction, ExternalOp)

        self.assertEqual(
            result.profile.events[0].input_value_ids,
            tuple(fusion.group.input_value_ids),
        )
        self.assertEqual(
            result.profile.events[0].output_value_ids,
            tuple(fusion.group.output_value_ids),
        )
        self.assertEqual(
            result.profile.events[1].input_value_ids,
            tuple(fusion.group.output_value_ids),
        )


class TestValueLifetimeAnalysis(unittest.TestCase):
    def _lowered(self):
        graph = capture(
            lambda x, y: ((x * y) + x).relu().sum(),
            Tensor([1.0, -2.0, 3.0]),
            Tensor([4.0, 5.0, -1.0]),
        ).graph
        return lower_graph(graph)

    def test_birth_and_last_use_across_fusion_and_reduction(self):
        lowered = self._lowered()
        program = lowered.program
        self.assertEqual(len(program.instructions), 2)
        fusion = program.instructions[0]
        reduction = program.instructions[1]
        self.assertIsInstance(fusion, PointwiseFusion)
        self.assertIsInstance(reduction, ExternalOp)

        analysis = analyze_value_lifetimes(program)
        x_id, y_id = program.graph.inputs
        self.assertEqual(analysis.value(x_id).birth_instruction, -1)
        self.assertEqual(analysis.value(y_id).birth_instruction, -1)
        self.assertEqual(analysis.value(x_id).last_use_instruction, 0)
        self.assertEqual(analysis.value(y_id).last_use_instruction, 0)

        produced = instruction_produced_value_ids(fusion)
        self.assertGreaterEqual(len(produced), 3)
        first_internal = analysis.value(produced[0])
        self.assertEqual(first_internal.birth_instruction, 0)
        self.assertEqual(first_internal.last_use_instruction, 0)

        fusion_output = fusion.group.output_value_ids[0]
        self.assertEqual(analysis.value(fusion_output).birth_instruction, 0)
        self.assertEqual(analysis.value(fusion_output).last_use_instruction, 1)

        graph_output = program.graph.outputs.value_id
        output_life = analysis.value(graph_output)
        self.assertEqual(output_life.birth_instruction, 1)
        self.assertTrue(output_life.is_output)
        self.assertEqual(output_life.last_use_instruction, len(program.instructions))

    def test_retained_fused_internal_value_extends_to_execution_boundary(self):
        lowered = self._lowered()
        fusion = lowered.program.instructions[0]
        self.assertIsInstance(fusion, PointwiseFusion)
        internal_id = instruction_produced_value_ids(fusion)[0]

        normal = analyze_value_lifetimes(lowered.program)
        retained = analyze_value_lifetimes(
            lowered.program,
            retained_value_ids=[internal_id],
        )

        self.assertEqual(normal.value(internal_id).last_use_instruction, 0)
        retained_life = retained.value(internal_id)
        self.assertTrue(retained_life.is_retained)
        self.assertEqual(
            retained_life.last_use_instruction,
            len(lowered.program.instructions),
        )

    def test_constant_and_output_ownership_metadata(self):
        constant = Tensor([2.0, 3.0, 4.0])
        graph = capture(lambda x: x + constant, Tensor([1.0, 1.0, 1.0])).graph
        lowered = lower_graph(graph)
        analysis = analyze_value_lifetimes(lowered.program)

        constant_ids = [
            value_id
            for value_id, value in lowered.program.graph.values.items()
            if value.kind == "constant"
        ]
        self.assertEqual(len(constant_ids), 1)
        life = analysis.value(constant_ids[0])
        self.assertTrue(life.is_constant)
        self.assertEqual(life.birth_instruction, -1)
        self.assertEqual(life.last_use_instruction, 0)
        self.assertEqual(life.estimated_nbytes, 24)

        output_id = lowered.program.graph.outputs.value_id
        self.assertTrue(analysis.value(output_id).is_output)

    def test_aot_saved_values_can_be_marked_retained_after_value_remap(self):
        program = aot_compile(
            lambda x, y: (x.relu() * y).sum(),
            Tensor([-2.0, 1.5, 3.0], requires_grad=True),
            Tensor([4.0, -1.0, 2.0], requires_grad=True),
        )
        lowered = lower_graph(program.forward_graph)
        retained_ids = [
            lowered.mapped_value_id(source_id)
            for source_id in program.spec.saved_value_ids
        ]
        analysis = analyze_value_lifetimes(
            lowered.program,
            retained_value_ids=retained_ids,
        )
        boundary = len(lowered.program.instructions)

        for value_id in retained_ids:
            life = analysis.value(value_id)
            self.assertTrue(life.is_retained)
            self.assertEqual(life.last_use_instruction, boundary)

    def test_missing_retained_value_fails_explicitly(self):
        lowered = self._lowered()
        with self.assertRaisesRegex(RuntimeAnalysisError, "do not exist"):
            analyze_value_lifetimes(lowered.program, retained_value_ids=[999])

    def test_intermediate_without_runtime_producer_fails_explicitly(self):
        graph = Graph()
        value = graph.add_intermediate(
            metadata={
                "shape": (2,),
                "dtype": "float64",
                "device": "cpu",
                "requires_grad": False,
            }
        )
        graph.outputs = ValueRef(value.id)
        program = LoweredProgram(graph=graph, instructions=())

        with self.assertRaisesRegex(RuntimeAnalysisError, "no runtime producer"):
            analyze_value_lifetimes(program)


if __name__ == "__main__":
    unittest.main()
