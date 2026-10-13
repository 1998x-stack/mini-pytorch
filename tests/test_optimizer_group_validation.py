import copy
import unittest

import numpy as np

from mini_pytorch import nn, optim


class TestOptimizerGroupValidation(unittest.TestCase):
    def _parameter(self, value=1.0):
        return nn.Parameter(np.array([value], dtype=np.float64))

    def test_sgd_group_override_cannot_bypass_negative_lr_validation(self):
        p = self._parameter()
        with self.assertRaisesRegex(ValueError, "lr must be non-negative"):
            optim.SGD([{"params": [p], "lr": -0.1}], lr=0.1)

    def test_sgd_group_override_validates_nesterov_combination(self):
        p = self._parameter()
        with self.assertRaisesRegex(ValueError, "Nesterov"):
            optim.SGD(
                [{"params": [p], "momentum": 0.0, "nesterov": True}],
                lr=0.1,
            )

    def test_adam_family_group_overrides_are_validated(self):
        cases = [
            (optim.Adam, {"betas": (1.0, 0.999)}, "betas"),
            (optim.AdamW, {"eps": 0.0}, "eps"),
            (optim.Adamax, {"weight_decay": -1.0}, "weight_decay"),
        ]
        for optimizer_cls, override, message in cases:
            with self.subTest(optimizer=optimizer_cls.__name__):
                p = self._parameter()
                with self.assertRaisesRegex(ValueError, message):
                    optimizer_cls([{"params": [p], **override}])

    def test_rmsprop_group_override_is_validated(self):
        p = self._parameter()
        with self.assertRaisesRegex(ValueError, "alpha"):
            optim.RMSprop([{"params": [p], "alpha": 1.0}])

        p = self._parameter()
        with self.assertRaisesRegex(TypeError, "centered must be bool"):
            optim.RMSprop([{"params": [p], "centered": 1}])

    def test_adagrad_group_override_is_validated(self):
        p = self._parameter()
        with self.assertRaisesRegex(ValueError, "initial_accumulator_value"):
            optim.Adagrad([{"params": [p], "initial_accumulator_value": -1.0}])

    def test_later_add_param_group_uses_same_validation_hook(self):
        p1 = self._parameter(1.0)
        p2 = self._parameter(2.0)
        optimizer = optim.SGD([p1], lr=0.1)

        with self.assertRaisesRegex(ValueError, "lr must be non-negative"):
            optimizer.add_param_group({"params": [p2], "lr": -0.5})

        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertIs(optimizer.param_groups[0]["params"][0], p1)

    def test_checkpoint_group_options_are_validated_transactionally(self):
        p1 = self._parameter(1.0)
        source = optim.Adam([p1], lr=0.1)
        p1.grad = np.array([0.5])
        source.step()
        state = copy.deepcopy(source.state_dict())
        state["param_groups"][0]["betas"] = (1.0, 0.999)

        p2 = self._parameter(2.0)
        target = optim.Adam([p2], lr=0.7)
        before_lr = target.learning_rates
        before_betas = target.param_groups[0]["betas"]
        before_state = copy.deepcopy(target.state)

        with self.assertRaisesRegex(ValueError, "betas"):
            target.load_state_dict(state)

        self.assertEqual(target.learning_rates, before_lr)
        self.assertEqual(target.param_groups[0]["betas"], before_betas)
        self.assertEqual(target.state, before_state)


if __name__ == "__main__":
    unittest.main()
