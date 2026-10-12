import unittest
import numpy as np

from mini_pytorch import nn, optim


class TestLRSchedulers(unittest.TestCase):
    def _optimizer(self, lr=1.0):
        return optim.SGD([nn.Parameter(np.array([1.0]))], lr=lr)

    def test_step_lr(self):
        opt = self._optimizer(1.0)
        sched = optim.StepLR(opt, step_size=2, gamma=0.1)
        self.assertEqual(sched.get_last_lr(), [1.0])
        sched.step()
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 1.0)
        sched.step()
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.1)
        sched.step(); sched.step()
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.01)

    def test_multi_step_lr(self):
        opt = self._optimizer(1.0)
        sched = optim.MultiStepLR(opt, milestones=[1, 3], gamma=0.5)
        sched.step(); self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.5)
        sched.step(); self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.5)
        sched.step(); self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.25)

    def test_exponential_lr(self):
        opt = self._optimizer(2.0)
        sched = optim.ExponentialLR(opt, gamma=0.5)
        sched.step(); self.assertAlmostEqual(opt.param_groups[0]["lr"], 1.0)
        sched.step(); self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.5)

    def test_cosine_annealing_lr(self):
        opt = self._optimizer(1.0)
        sched = optim.CosineAnnealingLR(opt, T_max=4, eta_min=0.0)
        sched.step(epoch=2); self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.5, places=7)
        sched.step(epoch=4); self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.0, places=7)

    def test_linear_lr(self):
        opt = self._optimizer(1.0)
        sched = optim.LinearLR(opt, start_factor=0.2, end_factor=1.0, total_iters=4)
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.2)
        sched.step(epoch=2); self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.6)
        sched.step(epoch=4); self.assertAlmostEqual(opt.param_groups[0]["lr"], 1.0)

    def test_lambda_lr_parameter_groups(self):
        p1, p2 = nn.Parameter(np.array([1.0])), nn.Parameter(np.array([2.0]))
        opt = optim.SGD([{"params": [p1], "lr": 1.0}, {"params": [p2], "lr": 2.0}])
        sched = optim.LambdaLR(opt, [lambda e: 0.5**e, lambda e: 1.0 / (e + 1)])
        sched.step()
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.5)
        self.assertAlmostEqual(opt.param_groups[1]["lr"], 1.0)

    def test_reduce_on_plateau(self):
        opt = self._optimizer(1.0)
        sched = optim.ReduceLROnPlateau(opt, factor=0.5, patience=1)
        sched.step(1.0); sched.step(1.1)
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 1.0)
        sched.step(1.2)
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.5)

    def test_scheduler_state_round_trip(self):
        opt1 = self._optimizer(1.0)
        sched1 = optim.StepLR(opt1, step_size=2, gamma=0.1)
        sched1.step(); sched1.step()
        state = sched1.state_dict()
        opt2 = self._optimizer(1.0)
        sched2 = optim.StepLR(opt2, step_size=2, gamma=0.1)
        sched2.load_state_dict(state)
        self.assertEqual(sched2.last_epoch, sched1.last_epoch)
        self.assertEqual(sched2.get_last_lr(), sched1.get_last_lr())
        self.assertAlmostEqual(opt2.param_groups[0]["lr"], 0.1)


if __name__ == "__main__":
    unittest.main()
