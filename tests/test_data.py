import unittest
import numpy as np

from mini_pytorch.data import DataLoader, TensorDataset


class TestData(unittest.TestCase):
    def test_dataloader_batches(self):
        x = np.arange(30).reshape(10, 3)
        y = np.arange(10)
        loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)
        batches = list(loader)
        self.assertEqual(len(batches), 3)
        self.assertEqual(batches[0][0].shape, (4, 3))
        self.assertEqual(batches[-1][0].shape, (2, 3))
        np.testing.assert_allclose(batches[0][1].data, [0, 1, 2, 3])

    def test_drop_last(self):
        dataset = TensorDataset(np.arange(10), np.arange(10))
        loader = DataLoader(dataset, batch_size=4, drop_last=True)
        self.assertEqual(len(loader), 2)
        self.assertEqual(len(list(loader)), 2)

    def test_shuffle_is_reproducible(self):
        dataset = TensorDataset(np.arange(12), np.arange(12))
        a = DataLoader(dataset, batch_size=12, shuffle=True, seed=7)
        b = DataLoader(dataset, batch_size=12, shuffle=True, seed=7)
        labels_a = next(iter(a))[1].data
        labels_b = next(iter(b))[1].data
        np.testing.assert_allclose(labels_a, labels_b)
        self.assertFalse(np.array_equal(labels_a, np.arange(12)))


if __name__ == "__main__":
    unittest.main()
