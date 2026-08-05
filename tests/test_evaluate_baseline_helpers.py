"""Unit tests for evaluate_baseline helpers (no full evaluation)."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "evaluate_baseline.py"


def _load_evaluate_baseline():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location("evaluate_baseline", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


evaluate_baseline = _load_evaluate_baseline()


class TestEvaluateBaselineHelpers(unittest.TestCase):
    def test_confusion_counts(self) -> None:
        probs = torch.tensor([0.9, 0.8, 0.2, 0.1])
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        counts = evaluate_baseline.confusion_counts(probs, labels, threshold=0.5)
        self.assertEqual(counts, {"tp": 1, "fp": 1, "tn": 1, "fn": 1})

    def test_classification_metrics_from_probs(self) -> None:
        probs = np.array([0.9, 0.8, 0.2, 0.1], dtype=np.float64)
        labels = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64)
        metrics = evaluate_baseline.classification_metrics_from_probs(
            probs, labels, threshold=0.5
        )

        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 0.5)
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["tn"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertTrue(0.0 <= metrics["roc_auc"] <= 1.0)
        self.assertTrue(0.0 <= metrics["pr_auc"] <= 1.0)

    def test_ranking_metrics_top_n_and_fractions(self) -> None:
        # Highest probs first should recover both pulsars in top 2.
        probs = np.array([0.95, 0.90, 0.40, 0.10, 0.05], dtype=np.float64)
        labels = np.array([1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        metrics = evaluate_baseline.ranking_metrics(
            probs,
            labels,
            top_ns=(1, 2, 10),
            top_fractions=(0.2, 0.4),
        )

        self.assertEqual(metrics["pulsars_in_top_1"], 1.0)
        self.assertEqual(metrics["pulsars_in_top_2"], 2.0)
        self.assertEqual(metrics["pulsars_in_top_10"], 2.0)
        # top 20% of 5 -> 1; top 40% of 5 -> 2
        self.assertAlmostEqual(metrics["recall_at_top_20pct"], 0.5)
        self.assertAlmostEqual(metrics["recall_at_top_40pct"], 1.0)

    def test_probability_ranks(self) -> None:
        probs = np.array([0.2, 0.9, 0.5], dtype=np.float64)
        ranks = evaluate_baseline.probability_ranks(probs)
        np.testing.assert_array_equal(ranks, np.array([3, 1, 2]))

    def test_build_prediction_rows(self) -> None:
        rows = evaluate_baseline.build_prediction_rows(
            filenames=["a.png", "b.png", "c.png"],
            labels=np.array([1.0, 0.0, 1.0]),
            probs=np.array([0.2, 0.9, 0.6]),
            threshold=0.5,
        )
        by_name = {row["filename"]: row for row in rows}
        self.assertEqual(by_name["b.png"]["rank"], 1)
        self.assertEqual(by_name["b.png"]["predicted_label"], 1)
        self.assertEqual(by_name["a.png"]["predicted_label"], 0)
        self.assertEqual(by_name["a.png"]["true_label"], 1)

    def test_save_predictions_and_metrics(self) -> None:
        rows = evaluate_baseline.build_prediction_rows(
            filenames=["x.png"],
            labels=np.array([1.0]),
            probs=np.array([0.7]),
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "preds.csv"
            json_path = Path(tmp_dir) / "metrics.json"
            evaluate_baseline.save_predictions_csv(csv_path, rows)
            evaluate_baseline.save_metrics_json(json_path, {"f1": 1.0, "tp": 1})
            self.assertTrue(csv_path.is_file())
            self.assertTrue(json_path.is_file())
            text = csv_path.read_text()
            self.assertIn("filename,true_label,pulsar_probability,predicted_label,rank", text)
            self.assertIn("x.png", text)


if __name__ == "__main__":
    unittest.main()
