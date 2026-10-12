import unittest
import numpy as np

from mini_pytorch import Tensor, nn


class TestClassification(unittest.TestCase):
    def test_cross_entropy_forward_and_backward(self):
        logits = Tensor([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]], requires_grad=True)
        targets = Tensor([0, 2])
        loss = nn.CrossEntropyLoss()(logits, targets)
        self.assertAlmostEqual(loss.item(), 0.4076059644, places=8)
        loss.backward()
        self.assertEqual(logits.grad.shape, (2, 3))
        np.testing.assert_allclose(logits.grad.sum(axis=1), [0.0, 0.0], atol=1e-12)
        self.assertLess(logits.grad[0, 0], 0)
        self.assertLess(logits.grad[1, 2], 0)

    def test_cross_entropy_reduction_none(self):
        logits = Tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        targets = Tensor([0, 1])
        losses = nn.CrossEntropyLoss(reduction="none")(logits, targets)
        self.assertEqual(losses.shape, (2,))
        losses.backward([1.0, 2.0])
        self.assertEqual(logits.grad.shape, (2, 2))

    def test_flatten(self):
        x = Tensor(np.arange(24).reshape(2, 3, 4), requires_grad=True)
        y = nn.Flatten()(x)
        self.assertEqual(y.shape, (2, 12))
        y.sum().backward()
        np.testing.assert_allclose(x.grad, np.ones((2, 3, 4)))


if __name__ == "__main__":
    unittest.main()
