import unittest

import numpy as np

from mini_pytorch import Tensor, no_grad
from mini_pytorch.compiler import (
    GraphBreakError,
    GuardSet,
    InputSignature,
    TensorSignature,
    compile_graph,
)


class TestSignaturesAndGuards(unittest.TestCase):
    def test_tensor_signature_equality_and_hash(self):
        a = TensorSignature.from_tensor(Tensor(np.zeros((2, 3)), requires_grad=True))
        b = TensorSignature.from_tensor(Tensor(np.ones((2, 3)), requires_grad=True))
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))

    def test_signature_specializes_shape_dtype_device_and_requires_grad(self):
        base = InputSignature.from_inputs(Tensor(np.zeros((2, 3)), requires_grad=True))
        shape = InputSignature.from_inputs(Tensor(np.zeros((4, 3)), requires_grad=True))
        dtype = InputSignature.from_inputs(Tensor(np.zeros((2, 3), dtype=np.float32), requires_grad=True))
        device = InputSignature.from_inputs(Tensor(np.zeros((2, 3)), requires_grad=True, device="edu"))
        grad = InputSignature.from_inputs(Tensor(np.zeros((2, 3)), requires_grad=False))
        self.assertNotEqual(base, shape)
        self.assertNotEqual(base, dtype)
        self.assertNotEqual(base, device)
        self.assertNotEqual(base, grad)

    def test_grad_mode_is_in_signature(self):
        x = Tensor([1.0], requires_grad=True)
        enabled = InputSignature.from_inputs(x)
        with no_grad():
            disabled = InputSignature.from_inputs(x)
        self.assertTrue(enabled.grad_enabled)
        self.assertFalse(disabled.grad_enabled)
        self.assertNotEqual(enabled, disabled)

    def test_guard_failure_is_inspectable(self):
        expected = InputSignature.from_inputs(Tensor(np.zeros((2, 3))))
        guards = GuardSet.from_signature(expected)
        actual = InputSignature.from_inputs(Tensor(np.zeros((4, 3))))
        result = guards.check(actual)
        self.assertFalse(result.passed)
        labels = [failure.label for failure in result.failures]
        self.assertIn("input[0].shape", labels)
        self.assertIn("expected (2, 3)", result.format())


class TestCompiledFunction(unittest.TestCase):
    @staticmethod
    def fn(x):
        return ((x * 2.0) + x).relu().sum()

    def test_first_call_miss_second_identical_call_hit(self):
        compiled = compile_graph(self.fn)
        out1 = compiled(Tensor([1.0, -2.0]))
        out2 = compiled(Tensor([3.0, -1.0]))
        self.assertEqual(float(out1.data), 3.0)
        self.assertEqual(float(out2.data), 9.0)
        stats = compiled.stats()
        self.assertEqual(stats.compile_count, 1)
        self.assertEqual(stats.cache_entries, 1)
        self.assertEqual(stats.miss_count, 1)
        self.assertEqual(stats.hit_count, 1)

    def test_shape_change_compiles_second_specialization_then_old_shape_hits(self):
        compiled = compile_graph(lambda x: (x + x).sum())
        compiled(Tensor(np.ones((2, 3))))   # miss -> compile 0
        compiled(Tensor(np.ones((2, 3))))   # hit 0
        compiled(Tensor(np.ones((4, 3))))   # miss -> compile 1
        compiled(Tensor(np.ones((2, 3))))   # hit 0
        stats = compiled.stats()
        self.assertEqual(stats.compile_count, 2)
        self.assertEqual(stats.recompile_count, 1)
        self.assertEqual(stats.cache_entries, 2)
        self.assertEqual(stats.hit_count, 2)
        self.assertEqual(stats.miss_count, 2)
        self.assertEqual([e.hit_count for e in compiled.cache.entries], [2, 0])

    def test_dtype_device_and_requires_grad_create_specializations(self):
        compiled = compile_graph(lambda x: (x * x).sum())
        compiled(Tensor([1.0, 2.0], dtype=np.float64, requires_grad=True))
        compiled(Tensor([1.0, 2.0], dtype=np.float32, requires_grad=True))
        compiled(Tensor([1.0, 2.0], dtype=np.float64, requires_grad=True, device="edu"))
        compiled(Tensor([1.0, 2.0], dtype=np.float64, requires_grad=False))
        self.assertEqual(compiled.stats().compile_count, 4)

    def test_grad_mode_creates_distinct_specialization(self):
        compiled = compile_graph(lambda x: (x * x).sum())
        x = Tensor([2.0, 3.0], requires_grad=True)
        out = compiled(x)
        out.backward()
        np.testing.assert_allclose(x.grad, [4.0, 6.0])

        with no_grad():
            no_grad_out = compiled(Tensor([2.0, 3.0], requires_grad=True))
            self.assertFalse(no_grad_out.requires_grad)

        replay_x = Tensor([4.0, 5.0], requires_grad=True)
        replay_out = compiled(replay_x)
        replay_out.backward()
        np.testing.assert_allclose(replay_x.grad, [8.0, 10.0])
        stats = compiled.stats()
        self.assertEqual(stats.compile_count, 2)
        self.assertEqual(stats.hit_count, 1)
        self.assertEqual(stats.miss_count, 2)

    def test_compiled_forward_and_backward_match_eager(self):
        def fn(x, y):
            return (((x * y) + x).relu()).sum()

        compiled = compile_graph(fn)
        # Warm/compile on one set of values.
        compiled(Tensor([1.0, 2.0], requires_grad=True), Tensor([3.0, 4.0], requires_grad=True))

        ex = Tensor([2.0, -1.0], requires_grad=True)
        ey = Tensor([4.0, 5.0], requires_grad=True)
        eager = fn(ex, ey)
        eager.backward()

        cx = Tensor([2.0, -1.0], requires_grad=True)
        cy = Tensor([4.0, 5.0], requires_grad=True)
        compiled_out = compiled(cx, cy)
        compiled_out.backward()

        np.testing.assert_allclose(compiled_out.data, eager.data)
        np.testing.assert_allclose(cx.grad, ex.grad)
        np.testing.assert_allclose(cy.grad, ey.grad)

    def test_multi_input_specialization_returns_to_original_entry(self):
        compiled = compile_graph(lambda x, y: ((x * y) + x).sum())

        compiled(Tensor(np.ones((2, 3))), Tensor(np.ones((2, 3))))  # compile 0
        compiled(Tensor(np.full((2, 3), 2.0)), Tensor(np.full((2, 3), 3.0)))  # hit 0
        compiled(Tensor(np.ones((4, 3))), Tensor(np.ones((4, 3))))  # compile 1
        compiled(Tensor(np.full((2, 3), 4.0)), Tensor(np.full((2, 3), 5.0)))  # hit 0

        stats = compiled.stats()
        self.assertEqual(stats.compile_count, 2)
        self.assertEqual(stats.hit_count, 2)
        self.assertEqual(stats.miss_count, 2)
        self.assertEqual([entry.hit_count for entry in compiled.cache.entries], [2, 0])

    def test_cpu_and_edu_same_shape_use_distinct_entries_then_cpu_hits_again(self):
        compiled = compile_graph(lambda x: (x * x).sum())

        compiled(Tensor([1.0, 2.0], device="cpu"))  # compile 0
        compiled(Tensor([1.0, 2.0], device="edu"))  # compile 1
        out = compiled(Tensor([3.0, 4.0], device="cpu"))  # hit 0

        self.assertEqual(str(out.device), "cpu")
        self.assertEqual(float(out.data), 25.0)
        stats = compiled.stats()
        self.assertEqual(stats.compile_count, 2)
        self.assertEqual(stats.hit_count, 1)
        self.assertEqual(stats.miss_count, 2)
        self.assertEqual([entry.signature.tensors[0].device for entry in compiled.cache.entries], ["cpu", "edu"])
        self.assertEqual([entry.hit_count for entry in compiled.cache.entries], [1, 0])

    def test_explain_is_non_mutating_and_reports_failure_reason(self):
        compiled = compile_graph(lambda x: (x + x).sum())
        compiled(Tensor(np.ones((2, 3))))
        before = compiled.stats()
        explanation = compiled.explain(Tensor(np.ones((8, 3))))
        after = compiled.stats()
        self.assertEqual(before, after)
        self.assertIsNone(explanation.matching_compile_id)
        text = explanation.format()
        self.assertIn("entry 0: FAIL", text)
        self.assertIn("input[0].shape", text)
        self.assertIn("(2, 3)", text)
        self.assertIn("(8, 3)", text)

    def test_explain_reports_matching_entry(self):
        compiled = compile_graph(lambda x: x + x)
        compiled(Tensor([1.0, 2.0]))
        explanation = compiled.explain(Tensor([9.0, 8.0]))
        self.assertEqual(explanation.matching_compile_id, 0)
        self.assertIn("entry 0: PASS", explanation.format())

    def test_graph_break_does_not_install_cache_entry(self):
        compiled = compile_graph(lambda x: x.exp() + x)
        with self.assertRaises(GraphBreakError):
            compiled(Tensor([1.0, 2.0], requires_grad=True))
        stats = compiled.stats()
        self.assertEqual(stats.cache_entries, 0)
        self.assertEqual(stats.compile_count, 0)
        self.assertEqual(stats.graph_break_count, 1)
        self.assertEqual(stats.miss_count, 1)

    def test_captured_closure_tensor_is_frozen_by_value(self):
        weight = Tensor([2.0, 3.0])
        compiled = compile_graph(lambda x: x * weight)
        first = compiled(Tensor([4.0, 5.0]))
        np.testing.assert_allclose(first.data, [8.0, 15.0])

        weight.data[...] = [10.0, 20.0]
        second = compiled(Tensor([4.0, 5.0]))
        # v0.12 policy: successful capture snapshots Tensor constants by value.
        np.testing.assert_allclose(second.data, [8.0, 15.0])
        self.assertEqual(compiled.stats().compile_count, 1)
        self.assertEqual(compiled.stats().hit_count, 1)

    def test_clear_cache_resets_entries_and_counters(self):
        compiled = compile_graph(lambda x: x + x)
        compiled(Tensor([1.0]))
        compiled(Tensor([2.0]))
        compiled.clear_cache()
        self.assertEqual(compiled.stats().cache_entries, 0)
        self.assertEqual(compiled.stats().compile_count, 0)
        self.assertEqual(compiled.stats().hit_count, 0)
        self.assertEqual(compiled.stats().miss_count, 0)


if __name__ == "__main__":
    unittest.main()
