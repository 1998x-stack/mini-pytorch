import unittest
import numpy as np

from mini_pytorch import Tensor, no_grad
from mini_pytorch.storage import Storage, contiguous_strides


class TestStorageAndViews(unittest.TestCase):
    def test_contiguous_stride_generation(self):
        self.assertEqual(contiguous_strides((2, 3, 4)), (12, 4, 1))
        self.assertEqual(contiguous_strides((5,)), (1,))
        self.assertEqual(contiguous_strides(()), ())

    def test_tensor_exposes_storage_metadata(self):
        x = Tensor(np.arange(12).reshape(3, 4))
        self.assertIsInstance(x.storage, Storage)
        self.assertEqual(x.shape, (3, 4))
        self.assertEqual(x.stride(), (4, 1))
        self.assertEqual(x.storage_offset, 0)
        self.assertTrue(x.is_contiguous())

    def test_transpose_shares_storage_and_changes_stride(self):
        x = Tensor(np.arange(6).reshape(2, 3))
        y = x.transpose(0, 1)
        self.assertIs(x.storage, y.storage)
        self.assertEqual(y.shape, (3, 2))
        self.assertEqual(y.stride(), (1, 3))
        self.assertFalse(y.is_contiguous())
        np.testing.assert_allclose(y.data, x.data.T)

    def test_permute_shares_storage(self):
        x = Tensor(np.arange(24).reshape(2, 3, 4))
        y = x.permute(2, 0, 1)
        self.assertIs(x.storage, y.storage)
        self.assertEqual(y.shape, (4, 2, 3))
        self.assertEqual(y.stride(), (1, 12, 4))
        np.testing.assert_allclose(y.data, np.transpose(x.data, (2, 0, 1)))

    def test_slice_is_view_with_offset(self):
        x = Tensor(np.arange(12).reshape(3, 4))
        y = x[1:, 1:]
        self.assertIs(x.storage, y.storage)
        self.assertEqual(y.storage_offset, 5)
        self.assertEqual(y.stride(), (4, 1))
        np.testing.assert_allclose(
            y.data,
            np.array([[5, 6, 7], [9, 10, 11]], dtype=float),
        )

    def test_view_on_contiguous_shares_storage(self):
        x = Tensor(np.arange(12).reshape(3, 4))
        y = x.view(2, 6)
        self.assertIs(x.storage, y.storage)
        self.assertTrue(y.is_contiguous())
        np.testing.assert_allclose(y.data, x.data.reshape(2, 6))

    def test_view_fails_on_incompatible_noncontiguous_layout(self):
        x = Tensor(np.arange(6).reshape(2, 3)).T
        with self.assertRaises(RuntimeError):
            x.view(6)

    def test_reshape_may_copy_noncontiguous_input(self):
        x = Tensor(np.arange(6).reshape(2, 3)).T
        y = x.reshape(6)
        self.assertIsNot(x.storage, y.storage)
        self.assertTrue(y.is_contiguous())
        np.testing.assert_allclose(y.data, x.data.reshape(6))

    def test_contiguous_allocates_only_when_needed(self):
        x = Tensor(np.arange(6).reshape(2, 3))
        self.assertIs(x.contiguous(), x)
        t = x.T
        c = t.contiguous()
        self.assertIsNot(c.storage, t.storage)
        self.assertTrue(c.is_contiguous())
        np.testing.assert_allclose(c.data, t.data)

    def test_alias_inplace_mutation_updates_all_views_and_version(self):
        x = Tensor(np.arange(6).reshape(2, 3))
        y = x.T
        before = x._version
        y.add_(10)
        self.assertEqual(x._version, before + 1)
        self.assertEqual(y._version, x._version)
        np.testing.assert_allclose(
            x.data,
            np.array([[10, 11, 12], [13, 14, 15]], dtype=float),
        )

    def test_leaf_requires_grad_inplace_requires_no_grad(self):
        x = Tensor([1.0, 2.0], requires_grad=True)
        with self.assertRaises(RuntimeError):
            x.add_(1)
        with no_grad():
            x.add_(1)
        np.testing.assert_allclose(x.data, [2.0, 3.0])

    def test_saved_tensor_version_check_detects_alias_mutation(self):
        x = Tensor([2.0, 3.0], requires_grad=True)
        y = (x * x).sum()
        alias = x.detach()
        alias.add_(1.0)
        with self.assertRaisesRegex(RuntimeError, "modified by an inplace operation"):
            y.backward()

    def test_view_backward(self):
        x = Tensor(np.arange(6.0).reshape(2, 3), requires_grad=True)
        y = x.T.reshape(6).sum()
        y.backward()
        np.testing.assert_allclose(x.grad, np.ones((2, 3)))

    def test_slice_backward_scatter(self):
        x = Tensor(np.arange(6.0), requires_grad=True)
        y = x[1:5:2].sum()
        y.backward()
        np.testing.assert_allclose(x.grad, [0, 1, 0, 1, 0, 0])


if __name__ == "__main__":
    unittest.main()
