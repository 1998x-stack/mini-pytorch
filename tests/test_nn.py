import unittest
import numpy as np

from mini_pytorch import Tensor, nn


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc2(self.fc1(x).relu())


class TestNN(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_recursive_parameter_registration(self):
        model = TinyModel()
        names = [name for name, _ in model.named_parameters()]
        self.assertEqual(names, ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"])

    def test_linear_shape_and_backward(self):
        layer = nn.Linear(3, 2)
        x = Tensor(np.ones((5, 3)), requires_grad=True)
        y = layer(x)
        self.assertEqual(y.shape, (5, 2))
        y.sum().backward()
        self.assertEqual(x.grad.shape, (5, 3))
        self.assertEqual(layer.weight.grad.shape, (2, 3))
        self.assertEqual(layer.bias.grad.shape, (2,))

    def test_state_dict_round_trip(self):
        model = TinyModel()
        state = model.state_dict()
        original = {k: v.copy() for k, v in state.items()}
        for p in model.parameters():
            p.data += 10
        model.load_state_dict(original)
        restored = model.state_dict()
        for key in original:
            np.testing.assert_allclose(restored[key], original[key])

    def test_train_eval_propagates(self):
        model = TinyModel()
        model.eval()
        self.assertFalse(model.training)
        self.assertTrue(all(not m.training for m in model.modules()))
        model.train()
        self.assertTrue(all(m.training for m in model.modules()))

    def test_sequential(self):
        model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 1))
        out = model(Tensor(np.ones((2, 3))))
        self.assertEqual(out.shape, (2, 1))
        self.assertEqual(len(model), 3)


if __name__ == "__main__":
    unittest.main()
