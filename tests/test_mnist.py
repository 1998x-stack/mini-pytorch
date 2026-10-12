import gzip
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mini_pytorch.data import MNIST


class TestMNIST(unittest.TestCase):
    def _write_split(self, raw: Path, prefix: str, images: np.ndarray, labels: np.ndarray):
        image_name = "train-images-idx3-ubyte.gz" if prefix == "train" else "t10k-images-idx3-ubyte.gz"
        label_name = "train-labels-idx1-ubyte.gz" if prefix == "train" else "t10k-labels-idx1-ubyte.gz"
        with gzip.open(raw / image_name, "wb") as f:
            f.write(struct.pack(">IIII", 2051, len(images), images.shape[1], images.shape[2]))
            f.write(images.astype(np.uint8).tobytes())
        with gzip.open(raw / label_name, "wb") as f:
            f.write(struct.pack(">II", 2049, len(labels)))
            f.write(labels.astype(np.uint8).tobytes())

    def test_reads_idx_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "MNIST" / "raw"
            raw.mkdir(parents=True)
            images = np.stack([
                np.zeros((28, 28), dtype=np.uint8),
                np.full((28, 28), 255, dtype=np.uint8),
            ])
            labels = np.array([3, 7], dtype=np.uint8)
            self._write_split(raw, "train", images, labels)

            dataset = MNIST(tmp, train=True, normalize=False)
            self.assertEqual(len(dataset), 2)
            image, label = dataset[1]
            self.assertEqual(image.shape, (28, 28))
            self.assertEqual(label, 7)
            self.assertAlmostEqual(float(image.mean()), 1.0, places=6)

    def test_limit_and_flatten(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "MNIST" / "raw"
            raw.mkdir(parents=True)
            images = np.zeros((3, 28, 28), dtype=np.uint8)
            labels = np.array([1, 2, 3], dtype=np.uint8)
            self._write_split(raw, "test", images, labels)

            dataset = MNIST(tmp, train=False, normalize=False, flatten=True, limit=2)
            self.assertEqual(len(dataset), 2)
            image, _ = dataset[0]
            self.assertEqual(image.shape, (784,))


if __name__ == "__main__":
    unittest.main()
