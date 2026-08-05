#!/usr/bin/env python3
"""Validate HTRU1Dataset loading for train and test splits."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.htru1_dataset import HTRU1Dataset

HTRU1_DIR = REPO_ROOT / "data" / "raw" / "htru1-batches-py"

TRAIN_BATCHES = [
    HTRU1_DIR / f"data_batch_{i}"
    for i in range(1, 6)
]
TEST_BATCH = HTRU1_DIR / "test_batch"


def main() -> None:
    missing = [path for path in [*TRAIN_BATCHES, TEST_BATCH] if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing HTRU1 batch file(s):\n"
            + "\n".join(f"  - {path}" for path in missing)
        )

    train_ds = HTRU1Dataset(TRAIN_BATCHES)
    test_ds = HTRU1Dataset([TEST_BATCH])

    train_pulsar = int((train_ds.labels == 1.0).sum().item())
    train_nonpulsar = int((train_ds.labels == 0.0).sum().item())
    test_pulsar = int((test_ds.labels == 1.0).sum().item())
    test_nonpulsar = int((test_ds.labels == 0.0).sum().item())

    train_image, _, train_filename = train_ds[0]
    train_min = float(train_ds.images.min())
    train_max = float(train_ds.images.max())

    loader = DataLoader(train_ds, batch_size=64, shuffle=False)
    batch_images, batch_labels, batch_filenames = next(iter(loader))

    print(f"Training dataset length: {len(train_ds)}")
    print(f"Test dataset length: {len(test_ds)}")
    print(f"Training pulsar count: {train_pulsar}")
    print(f"Training non-pulsar count: {train_nonpulsar}")
    print(f"Test pulsar count: {test_pulsar}")
    print(f"Test non-pulsar count: {test_nonpulsar}")
    print(f"Training image shape: {tuple(train_image.shape)}")
    print(f"Training image dtype: {train_image.dtype}")
    print(f"One training filename: {train_filename}")
    print(f"Training image min: {train_min}")
    print(f"Training image max: {train_max}")
    print(f"DataLoader batch image shape: {tuple(batch_images.shape)}")

    assert len(train_ds) == 50_000, f"expected 50000 train samples, got {len(train_ds)}"
    assert len(test_ds) == 10_000, f"expected 10000 test samples, got {len(test_ds)}"
    assert tuple(train_image.shape) == (2, 32, 32), train_image.shape
    assert train_image.dtype == torch.float32, train_image.dtype
    assert tuple(batch_images.shape) == (64, 2, 32, 32), batch_images.shape
    assert len(batch_labels) == 64
    assert len(batch_filenames) == 64

    train_names = set(train_ds.filenames)
    test_names = set(test_ds.filenames)
    overlap = train_names & test_names
    assert not overlap, (
        "Training and test filenames overlap "
        f"({len(overlap)} shared names); refusing possible leakage"
    )
    print("Filename overlap check: OK (no shared filenames)")
    print("Validation passed.")


if __name__ == "__main__":
    main()
