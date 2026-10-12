import unittest
import numpy as np

from mini_pytorch import Tensor, nn, optim


class TestOptim(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def _fit_linear(self, optimizer_cls, lr, steps):
        x = Tensor(np.linspace(-1, 1, 32).reshape(-1, 1))
        y = Tensor(3 * x.data - 2)
        model = nn.Linear(1, 1)
        loss_fn = nn.MSELoss()
        optimizer = optimizer_cls(model.parameters(), lr=lr)
        first_loss = loss_fn(model(x), y).item()
        for _ in range(steps):
            loss = loss_fn(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        final_loss = loss_fn(model(x), y).item()
        return first_loss, final_loss, model

    def test_sgd_converges(self):
        first, final, model = self._fit_linear(optim.SGD, lr=0.1, steps=120)
        self.assertLess(final, first * 1e-4)
        self.assertAlmostEqual(model.weight.data.item(), 3.0, places=2)
        self.assertAlmostEqual(model.bias.data.item(), -2.0, places=2)

    def test_adam_converges(self):
        first, final, model = self._fit_linear(optim.Adam, lr=0.08, steps=250)
        self.assertLess(final, first * 1e-4)
        self.assertAlmostEqual(model.weight.data.item(), 3.0, places=2)
        self.assertAlmostEqual(model.bias.data.item(), -2.0, places=2)

    def test_parameter_groups(self):
        a = nn.Linear(2, 2)
        b = nn.Linear(2, 1)
        opt = optim.SGD(
            [
                {"params": a.parameters(), "lr": 0.1},
                {"params": b.parameters(), "lr": 0.01},
            ],
            lr=1.0,
        )
        self.assertEqual(opt.param_groups[0]["lr"], 0.1)
        self.assertEqual(opt.param_groups[1]["lr"], 0.01)


if __name__ == "__main__":
    unittest.main()
