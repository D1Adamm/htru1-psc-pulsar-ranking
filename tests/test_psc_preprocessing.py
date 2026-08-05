"""Tests for PSC candidate preprocessing (local synthetic arrays only)."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from src.datasets.psc_preprocessing import psc_json_to_tensor


class TestPscPreprocessing(unittest.TestCase):
    def test_output_shape_dtype_range_and_channel_order(self) -> None:
        # Distinct patterns so channel identity survives resize + normalize.
        fvp = np.linspace(0.0, 10.0, 20 * 10, dtype=np.float32).reshape(20, 10)
        tvp = np.eye(16, dtype=np.float32) * 5.0
        payload = {
            "FvP": fvp.tolist(),
            "TvP": tvp.tolist(),
            "metadata": {
                "telescope": "AO327",
                "period": 0.0123,
                "sigma": 7.5,
            },
        }

        image, metadata = psc_json_to_tensor(payload, plot_id=11732926)

        self.assertEqual(tuple(image.shape), (2, 32, 32))
        self.assertEqual(image.dtype, torch.float32)
        self.assertGreaterEqual(float(image.min()), 0.0)
        self.assertLessEqual(float(image.max()), 1.0)

        # Swapping FvP/TvP should swap the output channels.
        swapped, _ = psc_json_to_tensor(
            {"FvP": tvp.tolist(), "TvP": fvp.tolist()}
        )
        self.assertTrue(torch.allclose(image[0], swapped[1], atol=1e-6))
        self.assertTrue(torch.allclose(image[1], swapped[0], atol=1e-6))
        self.assertFalse(torch.allclose(image[0], image[1]))

        self.assertEqual(metadata["plot_id"], 11732926)
        self.assertEqual(metadata["telescope"], "AO327")
        self.assertEqual(metadata["period"], 0.0123)
        self.assertEqual(metadata["sigma"], 7.5)
        self.assertEqual(metadata["original_fvp_shape"], (20, 10))
        self.assertEqual(metadata["original_tvp_shape"], (16, 16))

    def test_rejects_missing_keys(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing required array key"):
            psc_json_to_tensor({"FvP": [[1.0, 2.0], [3.0, 4.0]]})
        with self.assertRaisesRegex(ValueError, "missing required array key"):
            psc_json_to_tensor({"TvP": [[1.0, 2.0], [3.0, 4.0]]})

    def test_rejects_non_2d(self) -> None:
        with self.assertRaisesRegex(ValueError, "FvP must be 2-D"):
            psc_json_to_tensor(
                {
                    "FvP": [1.0, 2.0, 3.0],
                    "TvP": [[1.0, 2.0], [3.0, 4.0]],
                }
            )

    def test_rejects_nan_and_inf(self) -> None:
        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            psc_json_to_tensor(
                {
                    "FvP": [[1.0, float("nan")], [2.0, 3.0]],
                    "TvP": [[1.0, 2.0], [3.0, 4.0]],
                }
            )
        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            psc_json_to_tensor(
                {
                    "FvP": [[1.0, 2.0], [3.0, 4.0]],
                    "TvP": [[1.0, float("inf")], [3.0, 4.0]],
                }
            )

    def test_rejects_ragged_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "ragged"):
            psc_json_to_tensor(
                {
                    "FvP": [[1.0, 2.0], [3.0]],
                    "TvP": [[1.0, 2.0], [3.0, 4.0]],
                }
            )

    def test_constant_array_does_not_produce_nan(self) -> None:
        payload = {
            "FvP": np.full((8, 8), 3.14, dtype=np.float32).tolist(),
            "TvP": np.full((5, 12), -2.0, dtype=np.float32).tolist(),
        }
        image, metadata = psc_json_to_tensor(payload)
        self.assertTrue(torch.isfinite(image).all())
        self.assertTrue(torch.all(image == 0.0))
        self.assertEqual(metadata["original_fvp_shape"], (8, 8))
        self.assertEqual(metadata["original_tvp_shape"], (5, 12))


if __name__ == "__main__":
    unittest.main()
