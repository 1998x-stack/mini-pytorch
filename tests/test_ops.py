import unittest
import numpy as np

from mini_pytorch import Tensor
from mini_pytorch.utils import gradcheck


class TestOps(unittest.TestCase):
    def test_reductions(self):
        x = Tensor(np.arange(6.0).reshape(2, 3), requires_grad=True)
        y = x.mean(axis=0).sum()
        y.backward()
        np.testing.assert_allclose(x.grad, np.ones((2, 3)) / 2)

    def test_reshape_transpose(self):
        x = Tensor(np.arange(6.0).reshape(2, 3), requires_grad=True)
        y = x.T.reshape(6).sum()
        y.backward()
        np.testing.assert_allclose(x.grad, np.ones((2, 3)))

    def test_div_exp_log_gradcheck(self):
        ok, _ = gradcheck(
            lambda a, b: ((a.exp() / b).log()).sum(),
            [np.array([[0.2, -0.1], [0.4, 0.3]]), np.array([[1.2, 0.8], [1.5, 2.0]])],
        )
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
