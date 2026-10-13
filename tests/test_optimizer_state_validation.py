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

    def _optimizer_with_one_step(self, optimizer_cls, **kwargs):
        parameter = nn.Parameter(np.array([1.0, -1.0], dtype=np.float64))
        optimizer = optimizer_cls([parameter], **kwargs)
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

    def test_missing_required_group_option_is_rejected(self):
        _, source = self._adam_with_state()
        state = copy.deepcopy(source.state_dict())
        del state["param_groups"][0]["eps"]

        target_parameter = nn.Parameter(np.array([3.0, -3.0]))
        target = optim.Adam([target_parameter], lr=0.7)
        with self.assertRaisesRegex(KeyError, "missing optimizer options.*eps"):
            target.load_state_dict(state)

    def test_adam_missing_or_unexpected_state_key_is_rejected(self):
        _, source = self._adam_with_state()

        missing = copy.deepcopy(source.state_dict())
        del missing["state"][0]["exp_avg_sq"]
        target = optim.Adam([nn.Parameter(np.array([2.0, -2.0]))])
        with self.assertRaisesRegex(ValueError, "missing=.*exp_avg_sq"):
            target.load_state_dict(missing)

        unexpected = copy.deepcopy(source.state_dict())
        unexpected["state"][0]["mystery"] = 123
        target = optim.Adam([nn.Parameter(np.array([2.0, -2.0]))])
        with self.assertRaisesRegex(ValueError, "unexpected=.*mystery"):
            target.load_state_dict(unexpected)

    def test_step_metadata_requires_nonnegative_integer(self):
        _, source = self._adam_with_state()

        bad_bool = copy.deepcopy(source.state_dict())
        bad_bool["state"][0]["step"] = True
        target = optim.Adam([nn.Parameter(np.array([2.0, -2.0]))])
        with self.assertRaisesRegex(TypeError, "step.*integer"):
            target.load_state_dict(bad_bool)

        negative = copy.deepcopy(source.state_dict())
        negative["state"][0]["step"] = -1
        target = optim.Adam([nn.Parameter(np.array([2.0, -2.0]))])
        with self.assertRaisesRegex(ValueError, "step.*non-negative"):
            target.load_state_dict(negative)

    def test_sgd_momentum_state_schema_is_conditional(self):
        _, source = self._optimizer_with_one_step(optim.SGD, lr=0.1, momentum=0.9)
        state = copy.deepcopy(source.state_dict())
        del state["state"][0]["momentum_buffer"]

        target = optim.SGD(
            [nn.Parameter(np.array([2.0, -2.0]))],
            lr=0.1,
            momentum=0.9,
        )
        with self.assertRaisesRegex(ValueError, "missing=.*momentum_buffer"):
            target.load_state_dict(state)

    def test_rmsprop_conditional_state_schema_is_validated(self):
        _, source = self._optimizer_with_one_step(
            optim.RMSprop,
            lr=0.1,
            momentum=0.9,
            centered=True,
        )
        state = copy.deepcopy(source.state_dict())
        del state["state"][0]["grad_avg"]

        target = optim.RMSprop(
            [nn.Parameter(np.array([2.0, -2.0]))],
            lr=0.1,
            momentum=0.9,
            centered=True,
        )
        with self.assertRaisesRegex(ValueError, "missing=.*grad_avg"):
            target.load_state_dict(state)

    def test_adamax_and_adagrad_state_schemas_are_validated(self):
        for optimizer_cls, kwargs, removed in [
            (optim.Adamax, {}, "exp_inf"),
            (optim.Adagrad, {}, "sum"),
        ]:
            with self.subTest(optimizer=optimizer_cls.__name__):
                _, source = self._optimizer_with_one_step(optimizer_cls, **kwargs)
                state = copy.deepcopy(source.state_dict())
                del state["state"][0][removed]
                target = optimizer_cls([nn.Parameter(np.array([2.0, -2.0]))], **kwargs)
                with self.assertRaisesRegex(ValueError, f"missing=.*{removed}"):
                    target.load_state_dict(state)

    def test_parameter_without_state_entry_remains_valid(self):
        parameter = nn.Parameter(np.array([1.0, -1.0]))
        source = optim.Adam([parameter], lr=0.1)
        state = copy.deepcopy(source.state_dict())
        self.assertEqual(state["state"], {})

        target_parameter = nn.Parameter(np.array([2.0, -2.0]))
        target = optim.Adam([target_parameter], lr=0.9)
        target.load_state_dict(state)
        self.assertEqual(target.state, {})
        self.assertEqual(target.learning_rates, [0.1])


if __name__ == "__main__":
    unittest.main()
