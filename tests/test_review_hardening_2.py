import unittest

import numpy as np

from mini_pytorch import Tensor, nn, optim
from mini_pytorch.data import default_collate


class TestOptimizerRegistrationHardening(unittest.TestCase):
    def test_duplicate_parameter_within_one_group_is_rejected(self):
        parameter = nn.Parameter(np.array([1.0]))
        with self.assertRaisesRegex(ValueError, "more than once in the same parameter group"):
            optim.SGD([parameter, parameter], lr=0.1)

    def test_duplicate_parameter_across_groups_remains_rejected(self):
        parameter = nn.Parameter(np.array([1.0]))
        with self.assertRaisesRegex(ValueError, "more than one parameter group"):
            optim.SGD(
                [
                    {"params": [parameter], "lr": 0.1},
                    {"params": [parameter], "lr": 0.2},
                ],
                lr=0.1,
            )


class TestDefaultCollateHardening(unittest.TestCase):
    def test_ragged_lists_are_rejected_instead_of_zip_truncation(self):
        with self.assertRaisesRegex(ValueError, "equal-size list"):
            default_collate([[1.0, 2.0], [3.0]])

    def test_ragged_tuples_are_rejected_instead_of_zip_truncation(self):
        with self.assertRaisesRegex(ValueError, "equal-size tuple"):
            default_collate([(1.0, 2.0), (3.0,)])

    def test_heterogeneous_tensor_batch_is_rejected_explicitly(self):
        with self.assertRaisesRegex(TypeError, "homogeneous Tensor"):
            default_collate([Tensor([1.0]), np.array([2.0])])

    def test_equal_size_nested_lists_preserve_all_columns(self):
        result = default_collate([[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(len(result), 2)
        np.testing.assert_array_equal(result[0].data, [1.0, 3.0])
        np.testing.assert_array_equal(result[1].data, [2.0, 4.0])


if __name__ == "__main__":
    unittest.main()
