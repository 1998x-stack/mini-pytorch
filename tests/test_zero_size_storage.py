import unittest

import numpy as np

from mini_pytorch import Tensor


class TestZeroSizeStorageViews(unittest.TestCase):
    def test_empty_tensor_data_is_backed_by_its_storage(self):
        x = Tensor(np.empty((0, 3)))
        self.assertIs(x.data.base, x.storage._array)
        self.assertEqual(x.shape, (0, 3))
        self.assertEqual(x.stride(), (3, 1))

    def test_empty_transpose_preserves_storage_identity(self):
        x = Tensor(np.empty((0, 3)), requires_grad=True)
        y = x.transpose(0, 1)

        self.assertIs(y.storage, x.storage)
        self.assertEqual(y.shape, (3, 0))
        self.assertEqual(y.stride(), (1, 3))
        self.assertIs(y.data.base, x.storage._array)

    def test_empty_basic_slice_preserves_storage_and_backward(self):
        x = Tensor(np.arange(6.0).reshape(2, 3), requires_grad=True)
        y = x[1:1]

        self.assertIs(y.storage, x.storage)
        self.assertEqual(y.shape, (0, 3))
        y.sum().backward()
        np.testing.assert_array_equal(x.grad, np.zeros_like(x.data))

    def test_empty_view_preserves_storage_identity(self):
        x = Tensor(np.empty((0, 3)), requires_grad=True)
        y = x.view(0, 1, 3)

        self.assertIs(y.storage, x.storage)
        self.assertEqual(y.shape, (0, 1, 3))
        self.assertIs(y.data.base, x.storage._array)


if __name__ == "__main__":
    unittest.main()
