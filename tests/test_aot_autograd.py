import unittest

import numpy as np

from mini_pytorch import Tensor
from mini_pytorch.compiler import AOTDifferentiationError, aot_compile


class TestAOTAutograd(unittest.TestCase):
    def _compare(self, fn, arrays, *, device="cpu"):
        eager_inputs = [Tensor(a, requires_grad=True, device=device) for a in arrays]
        eager_out = fn(*eager_inputs)
        eager_out.backward()
        eager_grads = [np.array(x.grad, copy=True) for x in eager_inputs]

        examples = [Tensor(a, requires_grad=True, device=device) for a in arrays]
        program = aot_compile(fn, *examples)
        runtime = [Tensor(a, requires_grad=True, device=device) for a in arrays]
        forward = program.forward(*runtime)
        grads = program.backward(forward.context)

        np.testing.assert_allclose(forward.output.data, eager_out.data)
        self.assertEqual(len(grads), len(eager_grads))
        for got, expected in zip(grads, eager_grads):
            np.testing.assert_allclose(got.data, expected)
            self.assertFalse(got.requires_grad)
        return program

    def test_add_sum_matches_eager(self):
        self._compare(lambda x, y: (x + y).sum(), [np.arange(6).reshape(2, 3), np.ones((2, 3))])

    def test_mul_broadcast_matches_eager(self):
        self._compare(
            lambda x, y: (x * y).sum(),
            [np.arange(6, dtype=float).reshape(2, 3), np.array([2.0, 3.0, 4.0])],
        )

    def test_relu_matches_eager_and_saves_input(self):
        program = self._compare(lambda x: x.relu().sum(), [np.array([-2.0, 0.0, 3.0])])
        self.assertEqual(len(program.spec.saved_value_ids), 1)
        self.assertIn("aot::relu_backward.default", program.backward_graph.format())

    def test_matmul_matches_eager(self):
        self._compare(
            lambda x, y: (x @ y).sum(),
            [np.arange(6, dtype=float).reshape(2, 3), np.arange(12, dtype=float).reshape(3, 4)],
        )

    def test_shared_subgraph_accumulates_gradient_explicitly(self):
        program = self._compare(lambda x: ((x * x) + (x * x)).sum(), [np.array([2.0, -3.0])])
        # Reverse graph must contain explicit add nodes for multiple contributions.
        self.assertGreaterEqual(program.backward_graph.format().count("aten::add.Tensor"), 1)

    def test_non_grad_input_is_excluded_from_backward_outputs(self):
        def fn(x, y):
            return (x * y).sum()

        x0 = Tensor([1.0, 2.0], requires_grad=True)
        y0 = Tensor([3.0, 4.0], requires_grad=False)
        program = aot_compile(fn, x0, y0)
        self.assertEqual(len(program.spec.forward_input_ids), 1)
        run = program.forward(Tensor([5.0, 6.0], requires_grad=True), Tensor([7.0, 8.0]))
        grads = program.backward(run.context)
        self.assertEqual(len(grads), 1)
        np.testing.assert_allclose(grads[0].data, [7.0, 8.0])

    def test_cpu_and_edu_match(self):
        arrays = [np.array([-2.0, 1.5, 3.0])]
        cpu = self._compare(lambda x: (x.relu() * x).sum(), arrays, device="cpu")
        edu = self._compare(lambda x: (x.relu() * x).sum(), arrays, device="edu")
        self.assertEqual(cpu.backward_graph.format(), edu.backward_graph.format())

    def test_non_scalar_output_accepts_explicit_grad_output(self):
        fn = lambda x: x * x
        example = Tensor([1.0, 2.0], requires_grad=True)
        program = aot_compile(fn, example)
        run = program.forward(Tensor([3.0, 4.0], requires_grad=True))
        seed = Tensor([10.0, 20.0])
        grads = program.backward(run.context, seed)
        np.testing.assert_allclose(grads[0].data, [60.0, 160.0])

    def test_non_scalar_output_without_seed_fails(self):
        program = aot_compile(lambda x: x * x, Tensor([1.0, 2.0], requires_grad=True))
        run = program.forward(Tensor([3.0, 4.0], requires_grad=True))
        with self.assertRaisesRegex(ValueError, "explicit grad_output"):
            program.backward(run.context)

    def test_grad_output_shape_and_device_are_validated(self):
        program = aot_compile(lambda x: x * x, Tensor([1.0, 2.0], requires_grad=True))
        run = program.forward(Tensor([3.0, 4.0], requires_grad=True))
        with self.assertRaisesRegex(ValueError, "shape"):
            program.backward(run.context, Tensor([1.0]))

        edu_program = aot_compile(
            lambda x: x * x,
            Tensor([1.0, 2.0], requires_grad=True, device="edu"),
        )
        edu_run = edu_program.forward(Tensor([3.0, 4.0], requires_grad=True, device="edu"))
        with self.assertRaisesRegex(ValueError, "device"):
            edu_program.backward(edu_run.context, Tensor([1.0, 1.0], device="cpu"))

    def test_sum_axis_keepdims_matches_eager(self):
        def fn(x):
            return (x.sum(axis=1, keepdims=True) * x.sum(axis=1, keepdims=True)).sum()

        self._compare(fn, [np.arange(6, dtype=float).reshape(2, 3)])

    def test_unsupported_operator_fails_explicitly(self):
        from mini_pytorch import functional

        x = Tensor(np.ones((1, 1, 3, 3)), requires_grad=True)
        w = Tensor(np.ones((1, 1, 2, 2)), requires_grad=True)
        b = Tensor(np.zeros(1), requires_grad=True)
        with self.assertRaisesRegex(AOTDifferentiationError, "no graph VJP rule"):
            aot_compile(lambda a, k, bias: functional.conv2d(a, k, bias).sum(), x, w, b)

    def test_program_format_shows_forward_backward_and_saved_values(self):
        program = aot_compile(lambda x: (x * x).sum(), Tensor([1.0, 2.0], requires_grad=True))
        text = program.format()
        self.assertIn("FORWARD GRAPH", text)
        self.assertIn("BACKWARD GRAPH", text)
        self.assertIn("SAVED VALUES", text)
        self.assertIn("aot::sum_backward.default", text)

    def test_backward_helpers_are_first_order(self):
        program = aot_compile(lambda x: (x.relu() * x).sum(), Tensor([1.0, 2.0], requires_grad=True))
        run = program.forward(Tensor([3.0, 4.0], requires_grad=True))
        (grad,) = program.backward(run.context)
        self.assertFalse(grad.requires_grad)
        self.assertIsNone(grad._grad_fn)


if __name__ == "__main__":
    unittest.main()
