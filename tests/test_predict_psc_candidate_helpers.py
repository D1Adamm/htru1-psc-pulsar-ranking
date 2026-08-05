"""Unit tests for predict_psc_candidate helpers (no live PSC HTTP calls)."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "predict_psc_candidate.py"


def _load_module():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location("predict_psc_candidate", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


predict_psc = _load_module()


def _synthetic_payload() -> dict:
    return {
        "FvP": np.linspace(0.0, 1.0, 12 * 8, dtype=np.float32).reshape(12, 8).tolist(),
        "TvP": np.eye(10, dtype=np.float32).tolist(),
        "metadata": {
            "telescope": "AO327",
            "period": 0.045,
            "sigma": 6.2,
        },
    }


class TestPredictPscHelpers(unittest.TestCase):
    def test_score_image_shapes_and_keys(self) -> None:
        from src.models.baseline_cnn import BaselineCNN

        model = BaselineCNN()
        model.eval()
        image = torch.rand(2, 32, 32, dtype=torch.float32)
        result = predict_psc.score_image(model, image, device=torch.device("cpu"))

        self.assertEqual(result["processed_tensor_shape"], (1, 2, 32, 32))
        self.assertIn("raw_logit", result)
        self.assertIn("pulsar_probability", result)
        self.assertIn(result["predicted_label"], (0, 1))
        self.assertGreaterEqual(result["pulsar_probability"], 0.0)
        self.assertLessEqual(result["pulsar_probability"], 1.0)

    def test_score_psc_payload_includes_metadata(self) -> None:
        from src.models.baseline_cnn import BaselineCNN

        model = BaselineCNN()
        model.eval()
        result = predict_psc.score_psc_payload(
            _synthetic_payload(),
            model,
            torch.device("cpu"),
            plot_id=11732926,
            threshold=0.5,
        )

        self.assertEqual(result["plot_id"], 11732926)
        self.assertEqual(result["telescope"], "AO327")
        self.assertEqual(result["period"], 0.045)
        self.assertEqual(result["sigma"], 6.2)
        self.assertEqual(result["original_fvp_shape"], (12, 8))
        self.assertEqual(result["original_tvp_shape"], (10, 10))
        self.assertEqual(result["processed_tensor_shape"], (1, 2, 32, 32))

    def test_format_prediction_report_is_experimental(self) -> None:
        text = predict_psc.format_prediction_report(
            {
                "plot_id": 1,
                "telescope": "AO327",
                "period": 0.1,
                "sigma": 2.0,
                "original_fvp_shape": (8, 8),
                "original_tvp_shape": (8, 8),
                "processed_tensor_shape": (1, 2, 32, 32),
                "raw_logit": 0.25,
                "pulsar_probability": 0.562176,
                "threshold": 0.5,
                "predicted_label": 1,
            }
        )
        self.assertIn("Experimental cross-survey", text)
        self.assertIn("not a confirmed scientific classification", text)
        self.assertIn("Plot ID: 1", text)
        self.assertIn("Pulsar probability:", text)

    def test_load_baseline_cnn_roundtrip(self) -> None:
        from src.models.baseline_cnn import BaselineCNN

        model = BaselineCNN()
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "baseline_cnn_best.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": {},
                    "epoch": 1,
                    "val_f1": 0.0,
                    "pos_weight": torch.tensor([1.0]),
                    "random_seed": 42,
                },
                path,
            )
            loaded = predict_psc.load_baseline_cnn(path, device=torch.device("cpu"))
            self.assertFalse(loaded.training)

    def test_score_image_rejects_wrong_shape(self) -> None:
        from src.models.baseline_cnn import BaselineCNN

        model = BaselineCNN()
        with self.assertRaisesRegex(ValueError, r"\(2, 32, 32\)"):
            predict_psc.score_image(
                model,
                torch.rand(3, 32, 32),
                device=torch.device("cpu"),
            )

    def test_fetch_is_reusable_and_not_called_by_default(self) -> None:
        # Ensure the prediction script wires through inspect.fetch_plot_json
        # without hitting the network when a session mock is provided.
        session = MagicMock()
        response = MagicMock()
        response.ok = True
        response.json.return_value = _synthetic_payload()
        session.get.return_value = response

        payload = predict_psc.fetch_plot_json(11732926, session=session, timeout_s=5)
        self.assertIn("FvP", payload)
        session.get.assert_called_once()


if __name__ == "__main__":
    unittest.main()
