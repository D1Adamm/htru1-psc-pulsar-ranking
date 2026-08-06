"""PyTorch Dataset over locally cached PSC candidates, plus source-aware splitting.

Reads the manifest produced by ``scripts/build_psc_manifest.py`` and the JSON
cache written by ``scripts/rank_psc_candidates.py``. Preprocessing goes through
``arrays_to_tensor`` so training sees exactly the tensors the evaluation
pipeline scores — no second code path.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.datasets.psc_preprocessing import arrays_to_tensor
from src.models.three_class_head import CLASS_TO_INDEX


class PSCCachedDataset(Dataset):
    """PSC candidates loaded from the local JSON cache.

    Each item is ``(image, label, plot_id)`` where image is ``(2, 32, 32)``
    float32 and label is a long in ``[0, 3)`` following
    ``three_class_head.CLASS_NAMES``.

    Candidates whose cache file is missing or unusable are skipped and recorded
    in ``self.skipped`` rather than raising, so a partial cache still trains.
    """

    def __init__(self, rows: list[dict[str, str]], cache_dir: Path) -> None:
        self.images: list[torch.Tensor] = []
        self.labels: list[int] = []
        self.plot_ids: list[str] = []
        self.source_groups: list[str] = []
        self.skipped: list[tuple[str, str]] = []

        for row in rows:
            plot_id = row["plot_id"]
            category = row["source_category"]
            if category not in CLASS_TO_INDEX:
                self.skipped.append((plot_id, f"unknown category {category!r}"))
                continue

            path = cache_dir / f"{plot_id}.json"
            if not path.is_file():
                self.skipped.append((plot_id, "not cached"))
                continue

            try:
                with open(path) as f:
                    payload = json.load(f)
                image = arrays_to_tensor(
                    np.asarray(payload["FvP"], dtype=float),
                    np.asarray(payload["TvP"], dtype=float),
                )
            except Exception as exc:  # noqa: BLE001 - a bad cache entry is not fatal
                self.skipped.append((plot_id, f"{type(exc).__name__}: {exc}"))
                continue

            self.images.append(image)
            self.labels.append(CLASS_TO_INDEX[category])
            self.plot_ids.append(plot_id)
            self.source_groups.append(row.get("source_group", plot_id))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        return (
            self.images[index],
            torch.tensor(self.labels[index], dtype=torch.long),
            self.plot_ids[index],
        )

    def class_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for label in self.labels:
            counts[label] = counts.get(label, 0) + 1
        return counts


def read_manifest_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def split_indices_by_source_group(
    source_groups: list[str],
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Split indices into train/val/test without splitting a source group.

    Repeat detections of one pulsar share a ``source_group``. If one detection
    landed in train and another in test, the model would effectively be tested
    on something it had already seen, so groups are assigned whole.

    Returns ``(train_idx, val_idx, test_idx)``.
    """
    if not source_groups:
        raise ValueError("no samples to split")
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1.0, got {fractions}")

    groups = sorted(set(source_groups))
    rng = random.Random(seed)
    rng.shuffle(groups)

    n = len(groups)
    n_train = int(round(n * fractions[0]))
    n_val = int(round(n * fractions[1]))
    assign = {}
    for i, group in enumerate(groups):
        if i < n_train:
            assign[group] = 0
        elif i < n_train + n_val:
            assign[group] = 1
        else:
            assign[group] = 2

    buckets: tuple[list[int], list[int], list[int]] = ([], [], [])
    for index, group in enumerate(source_groups):
        buckets[assign[group]].append(index)
    return buckets


def class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    """Inverse-frequency weights, normalised to mean 1.

    The PSC classes are imbalanced (roughly 500 / 548 / 500 before splitting,
    and more skewed within a split), so an unweighted loss would let the model
    ignore the smallest class.
    """
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for label in labels:
        counts[label] += 1
    if (counts == 0).any():
        missing = [i for i, c in enumerate(counts) if c == 0]
        raise ValueError(f"class(es) {missing} absent from this split; cannot weight")
    weights = counts.sum() / (num_classes * counts)
    return weights / weights.mean()
