import unittest
import numpy as np

from mini_pytorch import Tensor, no_grad
from mini_pytorch.utils import gradcheck


class TestAutograd(unittest.TestCase):
    def test_scalar_chain_rule(self):
        x = Tensor(2.0, requires_grad=True)
        y = Tensor(3.0, requires_grad=True)
        z = x * y + x ** 2
        z.backward()
        self.assertAlmostEqual(float(x.grad), 7.0)
        self.assertAlmostEqual(float(y.grad), 2.0)

    def test_shared_node_accumulates_inside_graph(self):
        x = Tensor(3.0, requires_grad=True)
        y = x * x + x
        y.backward()
        self.assertAlmostEqual(float(x.grad), 7.0)

    def test_leaf_grad_accumulates_across_backward_calls(self):
        x = Tensor(2.0, requires_grad=True)
        (x * 3).backward()
        (x * 4).backward()
        self.assertAlmostEqual(float(x.grad), 7.0)

    def test_broadcast_backward(self):
        x = Tensor(np.arange(6).reshape(2, 3), requires_grad=True)
        b = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        (x + b).sum().backward()
        np.testing.assert_allclose(x.grad, np.ones((2, 3)))
        np.testing.assert_allclose(b.grad, np.array([2.0, 2.0, 2.0]))

    def test_matmul_gradients(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = Tensor([[2.0, 0.0], [1.0, 2.0]], requires_grad=True)
        (a @ b).sum().backward()
        np.testing.assert_allclose(a.grad, np.array([[2.0, 3.0], [2.0, 3.0]]))
        np.testing.assert_allclose(b.grad, np.array([[4.0, 4.0], [6.0, 6.0]]))

    def test_non_scalar_requires_seed(self):
        x = Tensor([1.0, 2.0], requires_grad=True)
        with self.assertRaises(RuntimeError):
            (x * 2).backward()

    def test_no_grad(self):
        x = Tensor(2.0, requires_grad=True)
        with no_grad():
            y = x * 3
        self.assertFalse(y.requires_grad)
        self.assertIsNone(y.grad_fn)

    def test_gradcheck_composite_expression(self):
        ok, _ = gradcheck(
            lambda a, b: ((a * b).relu() + a ** 2).mean(),
            [np.array([[0.5, -0.2], [1.0, 0.3]]), np.array([[1.2, 0.7], [-0.4, 2.0]])],
        )
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
