import copy
import unittest

import numpy as np

from mini_pytorch import nn, optim


class TestOptimizerStateValidation(unittest.TestCase):
    def _adam_with_state(self, value=1.0, *, lr=0.1):
        parameter = nn.Parameter(np.array([value, -value], dtype=np.float64))
        optimizer = optim.Adam([parameter], lr=lr)
        parameter.grad = np.array([0.25, -0.5], dtype=np.float64)
        optimizer.step()
        return parameter, optimizer

    def test_unknown_state_index_fails_instead_of_being_silently_dropped(self):
        _, source = self._adam_with_state()
        state = copy.deepcopy(source.state_dict())
        state["state"][99] = {
            "step": 1,
            "exp_avg": np.zeros((2,)),
            "exp_avg_sq": np.zeros((2,)),
        }

        target_parameter = nn.Parameter(np.array([3.0, -3.0]))
        target = optim.Adam([target_parameter], lr=0.7)
        with self.assertRaisesRegex(ValueError, "unknown parameter index 99"):
            target.load_state_dict(state)

    def test_wrong_shaped_array_state_fails_at_load_boundary(self):
        _, source = self._adam_with_state()
        state = copy.deepcopy(source.state_dict())
        state["state"][0]["exp_avg"] = np.zeros((1,))

        target_parameter = nn.Parameter(np.array([3.0, -3.0]))
        target = optim.Adam([target_parameter], lr=0.7)
        with self.assertRaisesRegex(ValueError, "exp_avg.*shape"):
            target.load_state_dict(state)

    def test_failed_load_is_transactional(self):
        _, source = self._adam_with_state(lr=0.2)
        state = copy.deepcopy(source.state_dict())
        state["param_groups"][0]["lr"] = 0.01
        state["state"][0]["exp_avg_sq"] = np.zeros((3,))

        target_parameter = nn.Parameter(np.array([4.0, -4.0]))
        target = optim.Adam([target_parameter], lr=0.9)
        before_lr = target.learning_rates
        before_state = copy.deepcopy(target.state)

        with self.assertRaises(ValueError):
            target.load_state_dict(state)

        self.assertEqual(target.learning_rates, before_lr)
        self.assertEqual(target.state, before_state)

    def test_string_state_keys_round_trip_for_json_style_checkpoints(self):
        _, source = self._adam_with_state()
        state = copy.deepcopy(source.state_dict())
        state["state"] = {str(index): values for index, values in state["state"].items()}

        target_parameter = nn.Parameter(np.array([5.0, -5.0]))
        target = optim.Adam([target_parameter], lr=0.9)
        target.load_state_dict(state)

        restored = target.state[id(target_parameter)]
        self.assertEqual(restored["step"], 1)
        np.testing.assert_allclose(restored["exp_avg"], state["state"]["0"]["exp_avg"])
        np.testing.assert_allclose(restored["exp_avg_sq"], state["state"]["0"]["exp_avg_sq"])
        self.assertEqual(target.learning_rates, source.learning_rates)


if __name__ == "__main__":
    unittest.main()
