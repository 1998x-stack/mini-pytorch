import unittest
import numpy as np

from mini_pytorch.data import (
    BatchSampler,
    DataLoader,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
)


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

    def test_sequential_sampler(self):
        dataset = TensorDataset(np.arange(5))
        sampler = SequentialSampler(dataset)
        self.assertEqual(list(sampler), [0, 1, 2, 3, 4])
        self.assertEqual(len(sampler), 5)

    def test_random_sampler_is_reproducible_across_instances(self):
        dataset = TensorDataset(np.arange(12))
        a = RandomSampler(dataset, seed=7)
        b = RandomSampler(dataset, seed=7)
        first_a = list(a)
        first_b = list(b)
        self.assertEqual(first_a, first_b)
        self.assertNotEqual(first_a, list(range(12)))

        # RNG state advances, so one sampler reshuffles on the next epoch.
        second_a = list(a)
        self.assertNotEqual(first_a, second_a)

    def test_random_sampler_with_replacement(self):
        dataset = TensorDataset(np.arange(3))
        sampler = RandomSampler(dataset, replacement=True, num_samples=10, seed=1)
        indices = list(sampler)
        self.assertEqual(len(indices), 10)
        self.assertTrue(all(0 <= index < 3 for index in indices))

    def test_batch_sampler(self):
        sampler = SequentialSampler(TensorDataset(np.arange(10)))
        keep = list(BatchSampler(sampler, batch_size=4, drop_last=False))
        drop = list(BatchSampler(sampler, batch_size=4, drop_last=True))
        self.assertEqual(keep, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]])
        self.assertEqual(drop, [[0, 1, 2, 3], [4, 5, 6, 7]])

    def test_dataloader_accepts_custom_sampler(self):
        dataset = TensorDataset(np.arange(5), np.arange(5))

        class ReverseSampler(SequentialSampler):
            def __iter__(self):
                return iter(range(len(self.data_source) - 1, -1, -1))

        loader = DataLoader(dataset, batch_size=5, sampler=ReverseSampler(dataset))
        labels = next(iter(loader))[1].data
        np.testing.assert_allclose(labels, [4, 3, 2, 1, 0])

    def test_dataloader_accepts_custom_batch_sampler(self):
        dataset = TensorDataset(np.arange(6), np.arange(6))
        batches = [[0, 2, 4], [1, 3, 5]]
        loader = DataLoader(dataset, batch_sampler=batches)
        observed = [batch[1].data.tolist() for batch in loader]
        self.assertEqual(observed, batches)
        self.assertEqual(len(loader), 2)

    def test_sampler_and_shuffle_are_mutually_exclusive(self):
        dataset = TensorDataset(np.arange(4))
        with self.assertRaises(ValueError):
            DataLoader(
                dataset,
                batch_size=2,
                sampler=SequentialSampler(dataset),
                shuffle=True,
            )

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
