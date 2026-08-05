"""Tests for HTRU1Dataset using data_batch_1."""

from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.datasets.htru1_dataset import HTRU1Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
HTRU1_DIR = REPO_ROOT / "data" / "raw" / "htru1-batches-py"
BATCH1_PATH = HTRU1_DIR / "data_batch_1"
TEST_BATCH_PATH = HTRU1_DIR / "test_batch"


class TestHTRU1Dataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not BATCH1_PATH.is_file():
            raise unittest.SkipTest(f"Missing batch file: {BATCH1_PATH}")

        with open(BATCH1_PATH, "rb") as f:
            cls.raw_batch = pickle.load(f, encoding="latin1")

        cls.raw_data = np.asarray(cls.raw_batch["data"])
        cls.raw_labels = np.asarray(cls.raw_batch["labels"])
        cls.dataset = HTRU1Dataset([BATCH1_PATH])

    def test_batch_keys_match_notebook_decoding(self) -> None:
        self.assertEqual(
            set(self.raw_batch.keys()),
            {"data", "labels", "batch_label", "filenames"},
        )
        self.assertEqual(self.raw_data.shape, (10000, 3072))

    def test_dataset_length(self) -> None:
        self.assertEqual(len(self.dataset), len(self.raw_labels))
        self.assertEqual(len(self.dataset), 10000)

    def test_item_types_and_shapes(self) -> None:
        image, label, filename = self.dataset[0]

        self.assertIsInstance(image, torch.Tensor)
        self.assertIsInstance(label, torch.Tensor)
        self.assertIsInstance(filename, str)

        self.assertEqual(image.shape, (2, 32, 32))
        self.assertEqual(image.dtype, torch.float32)
        self.assertEqual(label.dtype, torch.float32)
        self.assertEqual(label.ndim, 0)

    def test_values_scaled_to_unit_interval(self) -> None:
        image, _, _ = self.dataset[0]
        self.assertGreaterEqual(float(image.min()), 0.0)
        self.assertLessEqual(float(image.max()), 1.0)

    def test_channels_one_and_two_selected(self) -> None:
        image, _, _ = self.dataset[0]
        expected = (
            self.raw_data[0]
            .reshape(3, 32, 32)[1:3]
            .astype(np.float32)
            / 255.0
        )
        np.testing.assert_allclose(image.numpy(), expected, rtol=0, atol=0)

    def test_labels_reversed_pulsar_is_one(self) -> None:
        # Original mapping: 0 = pulsar, 1 = non-pulsar.
        pulsar_idx = int(np.where(self.raw_labels == 0)[0][0])
        nonpulsar_idx = int(np.where(self.raw_labels == 1)[0][0])

        _, pulsar_label, _ = self.dataset[pulsar_idx]
        _, nonpulsar_label, _ = self.dataset[nonpulsar_idx]

        self.assertEqual(float(pulsar_label), 1.0)
        self.assertEqual(float(nonpulsar_label), 0.0)

    def test_filename_matches_raw_batch(self) -> None:
        _, _, filename = self.dataset[0]
        raw_name = self.raw_batch["filenames"][0]
        if isinstance(raw_name, (bytes, np.bytes_)):
            raw_name = bytes(raw_name).decode("latin1")
        self.assertEqual(filename, str(raw_name))

    def test_rejects_empty_batch_paths(self) -> None:
        with self.assertRaises(ValueError):
            HTRU1Dataset([])

    def test_rejects_mixed_train_and_test_paths(self) -> None:
        with self.assertRaisesRegex(ValueError, "separate HTRU1Dataset"):
            HTRU1Dataset([BATCH1_PATH, TEST_BATCH_PATH])

    def test_rejects_malformed_pickle_missing_required_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_path = Path(tmp_dir) / "data_batch_bad"
            with open(bad_path, "wb") as f:
                pickle.dump({"data": np.zeros((1, 3072), dtype=np.uint8)}, f)

            with self.assertRaises(ValueError) as ctx:
                HTRU1Dataset([bad_path])

            message = str(ctx.exception)
            self.assertIn(str(bad_path), message)
            self.assertIn("labels", message)
            self.assertIn("filenames", message)


if __name__ == "__main__":
    unittest.main()
