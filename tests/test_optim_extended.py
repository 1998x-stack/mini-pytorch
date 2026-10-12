import unittest
import numpy as np

from mini_pytorch import Tensor, nn, optim


class TestExtendedOptimizers(unittest.TestCase):
    def setUp(self):
        np.random.seed(7)

    def _fit_linear(self, optimizer_cls, lr, steps=300, **kwargs):
        x = Tensor(np.linspace(-1, 1, 48).reshape(-1, 1))
        y = Tensor(2.5 * x.data - 1.25)
        model = nn.Linear(1, 1)
        loss_fn = nn.MSELoss()
        optimizer = optimizer_cls(model.parameters(), lr=lr, **kwargs)
        first = loss_fn(model(x), y).item()
        for _ in range(steps):
            loss = loss_fn(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        final = loss_fn(model(x), y).item()
        return first, final

    def test_adamw_converges(self):
        first, final = self._fit_linear(optim.AdamW, lr=0.05, weight_decay=0.0)
        self.assertLess(final, first * 1e-4)

    def test_rmsprop_converges(self):
        first, final = self._fit_linear(optim.RMSprop, lr=0.02, steps=400)
        self.assertLess(final, first * 1e-3)

    def test_adagrad_converges(self):
        first, final = self._fit_linear(optim.Adagrad, lr=0.5, steps=500)
        self.assertLess(final, first * 1e-3)

    def test_adamax_converges(self):
        first, final = self._fit_linear(optim.Adamax, lr=0.06, steps=350)
        self.assertLess(final, first * 1e-4)

    def test_sgd_nesterov(self):
        p = nn.Parameter(np.array([1.0]))
        p.grad = np.array([2.0])
        opt = optim.SGD([p], lr=0.1, momentum=0.9, nesterov=True)
        opt.step()
        self.assertAlmostEqual(p.data.item(), 1.0 - 0.1 * (2.0 + 0.9 * 2.0))

    def test_optimizer_state_round_trip(self):
        p1 = nn.Parameter(np.array([1.0, 2.0]))
        opt1 = optim.Adam([p1], lr=0.03)
        p1.grad = np.array([0.2, -0.1])
        opt1.step()
        state = opt1.state_dict()
        p2 = nn.Parameter(np.array([1.0, 2.0]))
        opt2 = optim.Adam([p2], lr=9.0)
        opt2.load_state_dict(state)
        self.assertAlmostEqual(opt2.param_groups[0]["lr"], 0.03)
        self.assertEqual(opt2.state[id(p2)]["step"], 1)
        np.testing.assert_allclose(opt2.state[id(p2)]["exp_avg"], opt1.state[id(p1)]["exp_avg"])


if __name__ == "__main__":
    unittest.main()
