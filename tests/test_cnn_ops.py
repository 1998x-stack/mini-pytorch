import unittest

import numpy as np

from mini_pytorch import DispatchKey, Tensor, dispatcher, functional as F, nn, optim
import mini_pytorch.kernels as kernels


class TestConv2d(unittest.TestCase):
    def test_reference_forward_known_values(self):
        x = np.array(
            [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]]
        )
        w = np.array([[[[1.0, 0.0], [0.0, -1.0]]]])
        b = np.array([0.5])
        out = kernels.conv2d_reference(x, w, b)
        expected = np.array([[[[-3.5, -3.5], [-3.5, -3.5]]]])
        np.testing.assert_allclose(out, expected)

    def test_im2col_matches_reference(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(2, 2, 5, 4))
        w = rng.normal(size=(3, 2, 3, 2))
        b = rng.normal(size=(3,))
        kwargs = dict(stride=(2, 1), padding=(1, 0), dilation=(1, 1), groups=1)
        reference = kernels.conv2d_reference(x, w, b, **kwargs)
        lowered = kernels.conv2d_im2col(x, w, b, **kwargs)
        np.testing.assert_allclose(lowered, reference, atol=1e-10, rtol=1e-10)

    def test_dispatcher_routes_autograd_then_cpu(self):
        x = Tensor(np.ones((1, 1, 3, 3)), requires_grad=True)
        w = Tensor(np.ones((1, 1, 2, 2)), requires_grad=True)
        b = Tensor(np.zeros(1), requires_grad=True)
        decision = dispatcher.explain(
            "aten::conv2d.default",
            x,
            w,
            b,
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            groups=1,
            implementation="reference",
        )
        self.assertEqual(decision.selected_key, DispatchKey.AUTOGRAD)
        out = F.conv2d(x, w, b)
        self.assertEqual(out.grad_fn, "Conv2dOpBackward")

    def test_groups_other_than_one_fail_clearly(self):
        x = Tensor(np.ones((1, 2, 4, 4)))
        w = Tensor(np.ones((2, 2, 3, 3)))
        with self.assertRaises(NotImplementedError):
            F.conv2d(x, w, groups=2)

    def test_backward_matches_finite_differences(self):
        rng = np.random.default_rng(3)
        x_np = rng.normal(size=(1, 1, 3, 3))
        w_np = rng.normal(size=(1, 1, 2, 2))
        b_np = rng.normal(size=(1,))

        x = Tensor(x_np.copy(), requires_grad=True)
        w = Tensor(w_np.copy(), requires_grad=True)
        b = Tensor(b_np.copy(), requires_grad=True)
        loss = F.conv2d(x, w, b, padding=1).sum()
        loss.backward()

        eps = 1e-6

        def objective(xv, wv, bv):
            return kernels.conv2d_reference(xv, wv, bv, padding=(1, 1)).sum()

        grad_x = np.zeros_like(x_np)
        for index in np.ndindex(x_np.shape):
            plus = x_np.copy(); plus[index] += eps
            minus = x_np.copy(); minus[index] -= eps
            grad_x[index] = (objective(plus, w_np, b_np) - objective(minus, w_np, b_np)) / (2 * eps)

        grad_w = np.zeros_like(w_np)
        for index in np.ndindex(w_np.shape):
            plus = w_np.copy(); plus[index] += eps
            minus = w_np.copy(); minus[index] -= eps
            grad_w[index] = (objective(x_np, plus, b_np) - objective(x_np, minus, b_np)) / (2 * eps)

        grad_b = np.zeros_like(b_np)
        for index in np.ndindex(b_np.shape):
            plus = b_np.copy(); plus[index] += eps
            minus = b_np.copy(); minus[index] -= eps
            grad_b[index] = (objective(x_np, w_np, plus) - objective(x_np, w_np, minus)) / (2 * eps)

        np.testing.assert_allclose(x.grad, grad_x, atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(w.grad, grad_w, atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(b.grad, grad_b, atol=2e-5, rtol=2e-5)


class TestMaxPool2d(unittest.TestCase):
    def test_forward_and_backward(self):
        values = np.arange(1.0, 17.0).reshape(1, 1, 4, 4)
        x = Tensor(values, requires_grad=True)
        out = F.max_pool2d(x, 2, stride=2)
        expected = np.array([[[[6.0, 8.0], [14.0, 16.0]]]])
        np.testing.assert_allclose(out.data, expected)

        out.sum().backward()
        expected_grad = np.zeros_like(values)
        expected_grad[0, 0, 1, 1] = 1.0
        expected_grad[0, 0, 1, 3] = 1.0
        expected_grad[0, 0, 3, 1] = 1.0
        expected_grad[0, 0, 3, 3] = 1.0
        np.testing.assert_allclose(x.grad, expected_grad)


class TestDropout(unittest.TestCase):
    def test_train_eval_and_seeded_sequence(self):
        x = Tensor(np.ones((8, 8)), requires_grad=True)
        a = nn.Dropout(0.5, seed=123)
        b = nn.Dropout(0.5, seed=123)

        y1 = a(x)
        y2 = b(x)
        np.testing.assert_array_equal(y1.data, y2.data)
        self.assertTrue(set(np.unique(y1.data)).issubset({0.0, 2.0}))

        a.eval()
        y_eval = a(x)
        self.assertIs(y_eval, x)


class TestBatchNorm(unittest.TestCase):
    def test_parameters_buffers_and_state_dict(self):
        bn = nn.BatchNorm2d(3)
        self.assertEqual([name for name, _ in bn.named_parameters()], ["weight", "bias"])
        self.assertEqual(
            [name for name, _ in bn.named_buffers()],
            ["running_mean", "running_var", "num_batches_tracked"],
        )
        self.assertEqual(
            set(bn.state_dict()),
            {"weight", "bias", "running_mean", "running_var", "num_batches_tracked"},
        )

    def test_train_then_eval_uses_running_stats(self):
        rng = np.random.default_rng(4)
        x_np = rng.normal(loc=3.0, scale=2.0, size=(5, 2, 3, 3))
        x = Tensor(x_np, requires_grad=True)
        bn = nn.BatchNorm2d(2, momentum=0.5)

        train_out = bn(x)
        np.testing.assert_allclose(train_out.data.mean(axis=(0, 2, 3)), np.zeros(2), atol=1e-7)
        self.assertFalse(np.allclose(bn.running_mean.data, 0.0))
        self.assertEqual(bn.num_batches_tracked.item(), 1.0)

        running_mean = bn.running_mean.data.copy()
        running_var = bn.running_var.data.copy()
        bn.eval()
        eval_out = bn(Tensor(x_np))
        expected = (x_np - running_mean.reshape(1, 2, 1, 1)) / np.sqrt(
            running_var.reshape(1, 2, 1, 1) + bn.eps
        )
        np.testing.assert_allclose(eval_out.data, expected, atol=1e-10, rtol=1e-10)


class TestCNNIntegration(unittest.TestCase):
    def test_cnn_training_step(self):
        np.random.seed(7)
        model = nn.Sequential(
            nn.Conv2d(1, 2, 3, padding=1, implementation="im2col"),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(2 * 4 * 4, 3),
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.05)

        rng = np.random.default_rng(8)
        x = Tensor(rng.normal(size=(4, 1, 8, 8)))
        target = Tensor(np.array([0, 1, 2, 1]))

        before = model[0].weight.data.copy()
        logits = model(x)
        self.assertEqual(logits.shape, (4, 3))
        loss = criterion(logits, target)
        optimizer.zero_grad()
        loss.backward()

        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(np.all(np.isfinite(parameter.grad)))

        optimizer.step()
        self.assertFalse(np.allclose(model[0].weight.data, before))


if __name__ == "__main__":
    unittest.main()
