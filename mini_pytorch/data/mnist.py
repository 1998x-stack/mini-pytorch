"""MNIST dataset loader using the original gzip-compressed IDX files."""
from __future__ import annotations

import gzip
import shutil
import struct
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np

from .dataset import Dataset


_MNIST_BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
_RESOURCES = {
    "train-images-idx3-ubyte.gz": "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz": "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz": "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz": "t10k-labels-idx1-ubyte.gz",
}


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "mini-pytorch-learning/0.3"})
    try:
        with urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
    except (OSError, URLError) as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"failed to download MNIST resource: {url}") from exc
    temporary.replace(destination)


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as stream:
        header = stream.read(16)
        if len(header) != 16:
            raise ValueError(f"invalid MNIST image header: {path}")
        magic, count, rows, cols = struct.unpack(">IIII", header)
        if magic != 2051:
            raise ValueError(f"unexpected MNIST image magic number {magic} in {path}")
        payload = stream.read()
    expected = count * rows * cols
    values = np.frombuffer(payload, dtype=np.uint8)
    if values.size != expected:
        raise ValueError(f"truncated MNIST image file: expected {expected} bytes, got {values.size}")
    return values.reshape(count, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as stream:
        header = stream.read(8)
        if len(header) != 8:
            raise ValueError(f"invalid MNIST label header: {path}")
        magic, count = struct.unpack(">II", header)
        if magic != 2049:
            raise ValueError(f"unexpected MNIST label magic number {magic} in {path}")
        payload = stream.read()
    values = np.frombuffer(payload, dtype=np.uint8)
    if values.size != count:
        raise ValueError(f"truncated MNIST label file: expected {count} bytes, got {values.size}")
    return values


class MNIST(Dataset):
    """The MNIST handwritten-digit dataset.

    Images are returned as ``(28, 28)`` NumPy arrays. DataLoader's default
    collate function converts batches to mini-pytorch Tensor objects.
    """

    classes = tuple(str(i) for i in range(10))
    mean = 0.1307
    std = 0.3081

    def __init__(
        self,
        root: str | Path = "./data",
        *,
        train: bool = True,
        download: bool = False,
        normalize: bool = True,
        flatten: bool = False,
        limit: int | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.raw_dir = self.root / "MNIST" / "raw"
        self.train = bool(train)
        self.normalize = bool(normalize)
        self.flatten = bool(flatten)

        if download:
            self.download()

        if self.train:
            image_path = self.raw_dir / "train-images-idx3-ubyte.gz"
            label_path = self.raw_dir / "train-labels-idx1-ubyte.gz"
        else:
            image_path = self.raw_dir / "t10k-images-idx3-ubyte.gz"
            label_path = self.raw_dir / "t10k-labels-idx1-ubyte.gz"

        missing = [str(path) for path in (image_path, label_path) if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "MNIST files are missing. Pass download=True once. Missing: " + ", ".join(missing)
            )

        self.images = _read_idx_images(image_path)
        self.targets = _read_idx_labels(label_path)
        if len(self.images) != len(self.targets):
            raise ValueError("MNIST image/label count mismatch")

        if limit is not None:
            if limit <= 0:
                raise ValueError("limit must be positive")
            limit = min(int(limit), len(self.images))
            self.images = self.images[:limit]
            self.targets = self.targets[:limit]

    def download(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        for filename, remote_name in _RESOURCES.items():
            destination = self.raw_dir / filename
            if destination.exists():
                continue
            _download_file(_MNIST_BASE_URL + remote_name, destination)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        image = self.images[index].astype(np.float32) / 255.0
        if self.normalize:
            image = (image - self.mean) / self.std
        if self.flatten:
            image = image.reshape(-1)
        label = int(self.targets[index])
        return image, label
