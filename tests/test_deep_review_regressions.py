import json
import unittest

import numpy as np

from mini_pytorch import Tensor, nn, optim
from mini_pytorch.distributed import create_process_group


class TestOptimizerMathHardening(unittest.TestCase):
    def test_adam_matches_bias_corrected_reference_with_large_epsilon(self):
        p = nn.Parameter(np.array([1.0, -2.0]))
        opt = optim.Adam(p for p in [p])
        opt.param_groups[0].update({"lr": 0.2, "betas": (0.8, 0.9), "eps": 0.5})

        m = np.zeros(2)
        v = np.zeros(2)
        expected = p.data.copy()
        grads = [np.array([1e-4, -2e-4]), np.array([0.3, -0.1]), np.array([-0.2, 0.4])]
        beta1, beta2 = 0.8, 0.9
        for step, grad in enumerate(grads, start=1):
            p.grad = grad.copy()
            opt.step()
            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * grad * grad
            m_hat = m / (1.0 - beta1**step)
            v_hat = v / (1.0 - beta2**step)
            expected -= 0.2 * m_hat / (np.sqrt(v_hat) + 0.5)
            np.testing.assert_allclose(p.data, expected, rtol=1e-13, atol=1e-13)

    def test_adamw_matches_decoupled_reference_with_large_epsilon(self):
        p = nn.Parameter(np.array([2.0]))
        opt = optim.AdamW([p], lr=0.1, betas=(0.5, 0.75), eps=0.25, weight_decay=0.2)
        grad = np.array([0.4])
        p.grad = grad.copy()
        opt.step()

        expected = np.array([2.0]) * (1.0 - 0.1 * 0.2)
        m = (1.0 - 0.5) * grad
        v = (1.0 - 0.75) * grad * grad
        m_hat = m / (1.0 - 0.5)
        v_hat = v / (1.0 - 0.75)
        expected -= 0.1 * m_hat / (np.sqrt(v_hat) + 0.25)
        np.testing.assert_allclose(p.data, expected, rtol=1e-13, atol=1e-13)


class TestCrossEntropyHardening(unittest.TestCase):
    def test_extreme_logits_do_not_cap_loss(self):
        logits = Tensor([[1000.0, -1000.0]], requires_grad=True)
        target = Tensor([1.0])
        loss = nn.CrossEntropyLoss()(logits, target)
        self.assertAlmostEqual(float(loss.item()), 2000.0, places=10)
        loss.backward()
        np.testing.assert_allclose(logits.grad, [[1.0, -1.0]], atol=0.0, rtol=0.0)

    def test_fractional_or_nonfinite_class_indices_are_rejected(self):
        logits = Tensor([[1.0, 2.0, 3.0]])
        loss = nn.CrossEntropyLoss()
        with self.assertRaisesRegex(ValueError, "integer class indices"):
            loss(logits, Tensor([1.5]))
        with self.assertRaisesRegex(ValueError, "finite integer class indices"):
            loss(logits, Tensor([np.nan]))

    def test_backward_is_finite_difference_of_same_forward(self):
        data = np.array([[0.2, -0.7, 1.1], [1.4, 0.3, -0.2]], dtype=float)
        target = Tensor([2.0, 0.0])
        logits = Tensor(data.copy(), requires_grad=True)
        loss_fn = nn.CrossEntropyLoss()
        loss_fn(logits, target).backward()
        analytical = logits.grad.copy()

        eps = 1e-6
        numerical = np.zeros_like(data)
        for index in np.ndindex(data.shape):
            plus = data.copy(); plus[index] += eps
            minus = data.copy(); minus[index] -= eps
            fp = float(loss_fn(Tensor(plus), target).item())
            fm = float(loss_fn(Tensor(minus), target).item())
            numerical[index] = (fp - fm) / (2.0 * eps)
        np.testing.assert_allclose(analytical, numerical, rtol=2e-6, atol=2e-7)


class TestOperatorEdgeHardening(unittest.TestCase):
    def test_power_zero_has_finite_zero_gradient_at_zero(self):
        x = Tensor([0.0, 2.0, -3.0], requires_grad=True)
        y = (x ** 0).sum()
        self.assertEqual(float(y.item()), 3.0)
        y.backward()
        np.testing.assert_array_equal(x.grad, np.zeros(3))
        self.assertTrue(np.all(np.isfinite(x.grad)))

    def test_transpose_rejects_out_of_range_dimensions(self):
        x = Tensor(np.arange(6.0).reshape(2, 3), requires_grad=True)
        with self.assertRaisesRegex(IndexError, "dimension out of range"):
            x.transpose(2, 0)
        with self.assertRaisesRegex(IndexError, "dimension out of range"):
            x.transpose(-3, 0)

        valid = x.transpose(-1, -2)
        self.assertEqual(valid.shape, (3, 2))

        scalar = Tensor(1.0)
        with self.assertRaisesRegex(IndexError, "at least one dimension"):
            scalar.transpose(0, 0)


class TestLeafSemanticsHardening(unittest.TestCase):
    def test_leaf_property_tracks_graph_provenance_not_requires_grad(self):
        constant = Tensor([1.0, 2.0])
        parameter_like = Tensor([1.0, 2.0], requires_grad=True)
        nonleaf = parameter_like * 2.0
        detached = nonleaf.detach()

        self.assertTrue(constant.is_leaf)
        self.assertTrue(parameter_like.is_leaf)
        self.assertFalse(nonleaf.is_leaf)
        self.assertTrue(detached.is_leaf)
        self.assertFalse(detached.requires_grad)


class TestSchedulerStateHardening(unittest.TestCase):
    def test_step_lr_restore_recovers_saved_configuration_and_future_trajectory(self):
        source_p = nn.Parameter(np.array([1.0]))
        source_opt = optim.SGD([source_p], lr=0.2)
        source = optim.StepLR(source_opt, step_size=2, gamma=0.5)
        source.step()
        source.step()
        state = source.state_dict()

        target_p = nn.Parameter(np.array([1.0]))
        target_opt = optim.SGD([target_p], lr=0.9)
        target = optim.StepLR(target_opt, step_size=9, gamma=0.9)
        target.load_state_dict(state)

        self.assertEqual(target.step_size, 2)
        self.assertEqual(target.gamma, 0.5)
        self.assertEqual(target.get_last_lr(), source.get_last_lr())
        for _ in range(6):
            source.step()
            target.step()
            np.testing.assert_allclose(target.get_last_lr(), source.get_last_lr())

    def test_scheduler_type_mismatch_fails_explicitly(self):
        p = nn.Parameter(np.array([1.0]))
        source_opt = optim.SGD([p], lr=0.1)
        state = optim.StepLR(source_opt, step_size=2, gamma=0.5).state_dict()

        q = nn.Parameter(np.array([1.0]))
        target_opt = optim.SGD([q], lr=0.1)
        target = optim.ExponentialLR(target_opt, gamma=0.9)
        with self.assertRaisesRegex(ValueError, "scheduler type mismatch"):
            target.load_state_dict(state)

    def test_scheduler_group_count_mismatch_fails_instead_of_zip_truncation(self):
        p = nn.Parameter(np.array([1.0]))
        source_opt = optim.SGD([p], lr=0.1)
        state = optim.StepLR(source_opt, step_size=2).state_dict()

        a = nn.Parameter(np.array([1.0]))
        b = nn.Parameter(np.array([2.0]))
        target_opt = optim.SGD(
            [
                {"params": [a], "lr": 0.1},
                {"params": [b], "lr": 0.2},
            ],
            lr=0.1,
        )
        target = optim.StepLR(target_opt, step_size=2)
        with self.assertRaisesRegex(ValueError, "optimizer groups"):
            target.load_state_dict(state)

    def test_lambda_lr_state_remains_transparently_serializable(self):
        p = nn.Parameter(np.array([1.0]))
        optimizer = optim.SGD([p], lr=0.1)
        scheduler = optim.LambdaLR(optimizer, lambda epoch: 1.0 / (epoch + 1.0))
        state = scheduler.state_dict()
        self.assertEqual(state["config"], {"num_lr_lambdas": 1})
        encoded = json.dumps(state)
        self.assertIsInstance(encoded, str)


class TestProcessGroupLifetimeHardening(unittest.TestCase):
    def test_completed_collective_slots_are_reclaimed_after_all_ranks_read(self):
        groups = create_process_group(3)
        for step in range(25):
            works = [
                group.all_reduce(np.array([float(step + group.rank)]), op="mean")
                for group in groups
            ]
            first_results = [work.result() for work in works]
            self.assertEqual(len(groups[0]._world._slots), 0)
            for work, expected in zip(works, first_results):
                np.testing.assert_array_equal(work.result(), expected)
                self.assertTrue(work.is_completed)
        self.assertEqual(len(groups[0]._world._slots), 0)


if __name__ == "__main__":
    unittest.main()
