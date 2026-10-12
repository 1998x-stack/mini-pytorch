import unittest

import numpy as np

from mini_pytorch import Function, Tensor


class SquarePlusOne(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * x + 1.0

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return (grad_output * 2.0 * x,)


class TestGraphLifetime(unittest.TestCase):
    def test_default_backward_releases_graph(self):
        x = Tensor(2.0, requires_grad=True)
        loss = (x * x).sum()
        loss.backward()
        self.assertAlmostEqual(float(x.grad), 4.0)
        with self.assertRaisesRegex(RuntimeError, "released"):
            loss.backward()

    def test_retain_graph_allows_one_more_execution(self):
        x = Tensor(2.0, requires_grad=True)
        loss = (x * x).sum()
        loss.backward(retain_graph=True)
        self.assertAlmostEqual(float(x.grad), 4.0)
        loss.backward()
        self.assertAlmostEqual(float(x.grad), 8.0)
        with self.assertRaisesRegex(RuntimeError, "released"):
            loss.backward()

    def test_shared_node_releases_once_and_accumulates_correctly(self):
        x = Tensor(3.0, requires_grad=True)
        shared = x * x
        loss = (shared + shared).sum()
        loss.backward()
        self.assertAlmostEqual(float(x.grad), 12.0)
        self.assertTrue(shared._grad_fn.ctx.is_released)

    def test_failed_create_graph_does_not_consume_graph(self):
        x = Tensor(2.0, requires_grad=True)
        loss = (x * x).sum()
        with self.assertRaisesRegex(NotImplementedError, "NumPy gradients"):
            loss.backward(create_graph=True)
        loss.backward()
        self.assertAlmostEqual(float(x.grad), 4.0)

    def test_cpu_and_edu_release_semantics_match(self):
        for device in ("cpu", "edu"):
            x = Tensor(2.0, requires_grad=True, device=device)
            loss = (x * x).sum()
            loss.backward()
            self.assertAlmostEqual(float(x.grad), 4.0)
            with self.assertRaisesRegex(RuntimeError, "released"):
                loss.backward()


class TestRetainGrad(unittest.TestCase):
    def test_nonleaf_grad_absent_by_default(self):
        x = Tensor([2.0, 3.0], requires_grad=True)
        hidden = x * x
        hidden.sum().backward()
        self.assertIsNone(hidden.grad)

    def test_retain_grad_materializes_nonleaf_grad(self):
        x = Tensor([2.0, 3.0], requires_grad=True)
        hidden = x * x
        hidden.retain_grad()
        (hidden * 3.0).sum().backward()
        np.testing.assert_allclose(hidden.grad, [3.0, 3.0])
        np.testing.assert_allclose(x.grad, [12.0, 18.0])
        self.assertTrue(hidden.retains_grad)

    def test_retained_grad_accumulates_across_retained_execution(self):
        x = Tensor([2.0, 3.0], requires_grad=True)
        hidden = x * x
        hidden.retain_grad()
        loss = (hidden * 3.0).sum()
        loss.backward(retain_graph=True)
        loss.backward()
        np.testing.assert_allclose(hidden.grad, [6.0, 6.0])
        np.testing.assert_allclose(x.grad, [24.0, 36.0])

    def test_retain_grad_rejects_non_differentiable_tensor(self):
        x = Tensor([1.0, 2.0])
        with self.assertRaisesRegex(RuntimeError, "does not require grad"):
            x.retain_grad()

    def test_retain_grad_works_on_edu(self):
        x = Tensor([2.0, 3.0], requires_grad=True, device="edu")
        hidden = x * x
        hidden.retain_grad()
        hidden.sum().backward()
        np.testing.assert_allclose(hidden.grad, [1.0, 1.0])
        np.testing.assert_allclose(x.grad, [4.0, 6.0])


class TestGradientHooks(unittest.TestCase):
    def test_hooks_observe_transform_and_run_in_order(self):
        x = Tensor([2.0, 3.0], requires_grad=True)
        hidden = x * 2.0
        events = []

        def first(grad):
            events.append(("first", grad.copy()))
            return grad * 3.0

        def second(grad):
            events.append(("second", grad.copy()))
            return None

        hidden.register_hook(first)
        hidden.register_hook(second)
        hidden.sum().backward()

        self.assertEqual([name for name, _ in events], ["first", "second"])
        np.testing.assert_allclose(events[0][1], [1.0, 1.0])
        np.testing.assert_allclose(events[1][1], [3.0, 3.0])
        np.testing.assert_allclose(x.grad, [6.0, 6.0])

    def test_hook_transformation_is_also_retained_grad(self):
        x = Tensor([1.0, 2.0], requires_grad=True)
        hidden = x * 4.0
        hidden.retain_grad()
        hidden.register_hook(lambda grad: grad * 0.5)
        hidden.sum().backward()
        np.testing.assert_allclose(hidden.grad, [0.5, 0.5])
        np.testing.assert_allclose(x.grad, [2.0, 2.0])

    def test_handle_remove_is_idempotent(self):
        x = Tensor(2.0, requires_grad=True)
        handle = x.register_hook(lambda grad: grad * 0.0)
        handle.remove()
        handle.remove()
        (x * 3.0).backward()
        self.assertAlmostEqual(float(x.grad), 3.0)

    def test_hook_replacement_shape_is_validated(self):
        x = Tensor([1.0, 2.0], requires_grad=True)
        hidden = x * 2.0
        hidden.register_hook(lambda grad: np.ones((1,)))
        with self.assertRaisesRegex(RuntimeError, "hook returned gradient shape"):
            hidden.sum().backward()
        # The failed execution did not release the graph.
        hidden._backward_hooks.clear()
        hidden.sum().backward()
        np.testing.assert_allclose(x.grad, [2.0, 2.0])

    def test_register_hook_rejects_non_differentiable_tensor(self):
        x = Tensor(1.0)
        with self.assertRaisesRegex(RuntimeError, "does not require grad"):
            x.register_hook(lambda grad: grad)

    def test_hook_tensor_replacement_must_match_device(self):
        x = Tensor([1.0], requires_grad=True, device="edu")
        hidden = x * 2.0
        hidden.register_hook(lambda grad: Tensor(grad, device="cpu"))
        with self.assertRaisesRegex(RuntimeError, "hook returned gradient on"):
            hidden.sum().backward()


class TestCustomFunction(unittest.TestCase):
    def test_custom_function_forward_backward_and_release(self):
        x = Tensor([2.0, -3.0], requires_grad=True)
        y = SquarePlusOne.apply(x)
        np.testing.assert_allclose(y.data, [5.0, 10.0])
        y.sum().backward()
        np.testing.assert_allclose(x.grad, [4.0, -6.0])
        with self.assertRaisesRegex(RuntimeError, "released"):
            y.backward(np.ones_like(y.data))

    def test_custom_function_saved_version_safety(self):
        x = Tensor([2.0, 3.0], requires_grad=True)
        y = SquarePlusOne.apply(x)
        x.detach().add_(1.0)
        with self.assertRaisesRegex(RuntimeError, "modified by an inplace operation"):
            y.sum().backward()

    def test_custom_function_preserves_edu_device(self):
        x = Tensor([2.0, -3.0], requires_grad=True, device="edu")
        y = SquarePlusOne.apply(x)
        self.assertEqual(str(y.device), "edu")
        y.sum().backward()
        np.testing.assert_allclose(x.grad, [4.0, -6.0])

    def test_tensor_gradient_seed_checks_device(self):
        x = Tensor([2.0, 3.0], requires_grad=True, device="edu")
        y = x * 2.0
        seed = Tensor([1.0, 1.0], device="cpu")
        with self.assertRaisesRegex(ValueError, "gradient device mismatch"):
            y.backward(seed)


if __name__ == "__main__":
    unittest.main()
