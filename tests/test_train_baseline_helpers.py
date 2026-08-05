"""Unit tests for train_baseline helpers (no full training)."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "train_baseline.py"


def _load_train_baseline():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location("train_baseline", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


train_baseline = _load_train_baseline()


class TestTrainBaselineHelpers(unittest.TestCase):
    def test_compute_pos_weight(self) -> None:
        labels = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0])
        # 3 non-pulsars / 2 pulsars
        weight = train_baseline.compute_pos_weight(labels)
        self.assertEqual(weight.shape, (1,))
        self.assertAlmostEqual(float(weight.item()), 1.5)

    def test_compute_pos_weight_rejects_no_pulsars(self) -> None:
        with self.assertRaises(ValueError):
            train_baseline.compute_pos_weight(torch.zeros(4))

    def test_split_train_val_indices_reproducible_and_sized(self) -> None:
        a_train, a_val = train_baseline.split_train_val_indices(100, 0.2, seed=42)
        b_train, b_val = train_baseline.split_train_val_indices(100, 0.2, seed=42)
        c_train, c_val = train_baseline.split_train_val_indices(100, 0.2, seed=7)

        self.assertEqual(len(a_train), 80)
        self.assertEqual(len(a_val), 20)
        self.assertTrue(torch.equal(a_train, b_train))
        self.assertTrue(torch.equal(a_val, b_val))
        self.assertFalse(torch.equal(a_train, c_train))

        combined = torch.cat([a_train, a_val])
        self.assertEqual(len(torch.unique(combined)), 100)

    def test_binary_classification_metrics(self) -> None:
        # logits -> probs via sigmoid; threshold 0.5
        logits = torch.tensor([4.0, -4.0, 4.0, -4.0])
        labels = torch.tensor([1.0, 0.0, 0.0, 1.0])
        # preds: 1, 0, 1, 0  -> tp=1, fp=1, fn=1
        metrics = train_baseline.binary_classification_metrics(logits, labels)

        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 0.5)

    def test_select_device_returns_torch_device(self) -> None:
        device = train_baseline.select_device()
        self.assertIsInstance(device, torch.device)
        self.assertIn(device.type, {"cpu", "cuda", "mps"})

    def test_save_checkpoint_roundtrip_keys(self) -> None:
        from src.models.baseline_cnn import BaselineCNN

        model = BaselineCNN()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        pos_weight = torch.tensor([10.0])

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "baseline_cnn_best.pt"
            train_baseline.save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                epoch=3,
                val_f1=0.75,
                pos_weight=pos_weight,
                random_seed=42,
            )
            ckpt = torch.load(path, map_location="cpu", weights_only=False)

        self.assertIn("model_state_dict", ckpt)
        self.assertIn("optimizer_state_dict", ckpt)
        self.assertEqual(ckpt["epoch"], 3)
        self.assertAlmostEqual(ckpt["val_f1"], 0.75)
        self.assertAlmostEqual(float(ckpt["pos_weight"].item()), 10.0)
        self.assertEqual(ckpt["random_seed"], 42)


if __name__ == "__main__":
    unittest.main()
