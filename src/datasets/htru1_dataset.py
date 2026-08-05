"""PyTorch Dataset for HTRU1 CIFAR-style pickle batches."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class HTRU1Dataset(Dataset):
    """Load one or more HTRU1 batch pickle files.

    Pass only training batch paths or only the test batch path. Do not mix
    train and test files in the same instance, to avoid test-data leakage.

    Each candidate is returned as:
      - image: float32 tensor of shape ``(2, 32, 32)`` (channels 1 and 2,
        values scaled to ``[0, 1]``)
      - label: float32 scalar with ``1.0 = pulsar`` and ``0.0 = non-pulsar``
      - filename: candidate identifier string
    """

    FLAT_SIZE = 3072
    NUM_CHANNELS = 3
    SPATIAL_SIZE = 32
    # Keep Frequency–Phase and Time–Phase; drop Period–DM (channel 0).
    KEEP_CHANNELS = (1, 2)
    REQUIRED_KEYS = ("data", "labels", "filenames")

    def __init__(self, batch_paths: Sequence[str | Path]) -> None:
        if not batch_paths:
            raise ValueError("batch_paths must contain at least one batch file")

        self._reject_mixed_train_test_paths(batch_paths)

        images: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        filenames: list[str] = []

        for batch_path in batch_paths:
            path = Path(batch_path)
            batch_images, batch_labels, batch_filenames = self._load_batch(path)
            images.append(batch_images)
            labels.append(batch_labels)
            filenames.extend(batch_filenames)

        self.images = torch.from_numpy(np.concatenate(images, axis=0))
        self.labels = torch.from_numpy(np.concatenate(labels, axis=0))
        self.filenames = filenames

    @staticmethod
    def _reject_mixed_train_test_paths(batch_paths: Sequence[str | Path]) -> None:
        names = [Path(path).name for path in batch_paths]
        has_train = any(name.startswith("data_batch_") for name in names)
        has_test = any(name == "test_batch" for name in names)
        if has_train and has_test:
            raise ValueError(
                "batch_paths mixes training files (data_batch_*) with "
                "test_batch. Use separate HTRU1Dataset instances for "
                "training and test data to avoid test-data leakage."
            )

    @classmethod
    def _load_batch(
        cls, path: Path
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        with open(path, "rb") as f:
            batch = pickle.load(f, encoding="latin1")

        if not isinstance(batch, dict):
            raise ValueError(
                f"{path}: expected a pickle dictionary, got {type(batch).__name__}"
            )

        missing_keys = [key for key in cls.REQUIRED_KEYS if key not in batch]
        if missing_keys:
            raise ValueError(
                f"{path}: missing required key(s): {', '.join(missing_keys)}"
            )

        data = np.asarray(batch["data"])
        raw_labels = np.asarray(batch["labels"])
        raw_filenames = batch["filenames"]

        if data.ndim != 2:
            raise ValueError(
                f"{path}: expected 2-D data array, got shape {data.shape}"
            )
        if data.shape[1] != cls.FLAT_SIZE:
            raise ValueError(
                f"{path}: expected flattened length {cls.FLAT_SIZE}, "
                f"got {data.shape[1]}"
            )
        if len(raw_labels) != len(data) or len(raw_filenames) != len(data):
            raise ValueError(
                f"{path}: data/labels/filenames length mismatch "
                f"({len(data)}, {len(raw_labels)}, {len(raw_filenames)})"
            )

        # Per-candidate length check (defensive if a row were ever ragged).
        for i, row in enumerate(data):
            if row.shape[0] != cls.FLAT_SIZE:
                raise ValueError(
                    f"{path}: candidate {i} has length {row.shape[0]}, "
                    f"expected {cls.FLAT_SIZE}"
                )

        images = data.reshape(
            -1, cls.NUM_CHANNELS, cls.SPATIAL_SIZE, cls.SPATIAL_SIZE
        )
        images = images[:, cls.KEEP_CHANNELS, :, :].astype(np.float32) / 255.0

        # Original HTRU1 labels: 0 = pulsar, 1 = non-pulsar.
        # Reverse so 1.0 = pulsar, 0.0 = non-pulsar.
        labels = (1.0 - raw_labels.astype(np.float32)).astype(np.float32)

        filenames = [cls._as_filename(name) for name in raw_filenames]
        return images, labels, filenames

    @staticmethod
    def _as_filename(name: object) -> str:
        if isinstance(name, bytes):
            return name.decode("latin1")
        if isinstance(name, np.bytes_):
            return bytes(name).decode("latin1")
        return str(name)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        return self.images[index], self.labels[index], self.filenames[index]
