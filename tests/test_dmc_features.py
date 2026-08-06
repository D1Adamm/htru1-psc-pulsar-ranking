"""Tests for DM-curve resampling. Synthetic curves, no network."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from src.datasets.psc_preprocessing import DM_VECTOR_LENGTH, dmc_to_vector

# Lengths actually observed across the 1,567 cached candidates.
OBSERVED_LENGTHS = (401, 201, 129, 101, 97)


def payload_with_peak(n: int, peak_index: int, lowdm=0.0, highdm=100.0) -> dict:
    curve = np.full(n, 1.0)
    curve[peak_index] = 50.0
    return {
        "DMc": curve.tolist(),
        "meta": {"lowdm": lowdm, "highdm": highdm, "ndms": n},
    }


class TestDmcToVector(unittest.TestCase):
    def test_handles_every_observed_length(self):
        for n in OBSERVED_LENGTHS:
            with self.subTest(length=n):
                vec, feats = dmc_to_vector(payload_with_peak(n, n // 2))
                self.assertEqual(tuple(vec.shape), (DM_VECTOR_LENGTH,))
                self.assertEqual(vec.dtype, torch.float32)
                self.assertTrue(torch.isfinite(vec).all())

    def test_output_is_unit_range(self):
        vec, _ = dmc_to_vector(payload_with_peak(401, 200))
        self.assertGreaterEqual(float(vec.min()), -1e-6)
        self.assertLessEqual(float(vec.max()), 1.0 + 1e-6)

    def test_peak_at_dm_zero_gives_zero_fraction(self):
        # Terrestrial interference: DMc peaks at the first trial DM.
        _vec, feats = dmc_to_vector(payload_with_peak(101, 0))
        self.assertEqual(feats["peak_fraction"], 0.0)
        self.assertAlmostEqual(feats["peak_dm"], 0.0, places=6)

    def test_peak_midway_gives_half_fraction(self):
        _vec, feats = dmc_to_vector(payload_with_peak(101, 50, 0.0, 100.0))
        self.assertAlmostEqual(feats["peak_fraction"], 0.5, places=6)
        self.assertAlmostEqual(feats["peak_dm"], 50.0, places=4)

    def test_peak_dm_respects_a_nonzero_low_edge(self):
        _vec, feats = dmc_to_vector(payload_with_peak(101, 50, 20.0, 120.0))
        self.assertAlmostEqual(feats["dm_range"], 100.0, places=6)
        self.assertAlmostEqual(feats["peak_dm"], 70.0, places=4)

    def test_separates_terrestrial_from_dispersed(self):
        # The physical discriminator EXP-006 identified.
        _v, rfi = dmc_to_vector(payload_with_peak(101, 0))
        _v, pulsar = dmc_to_vector(payload_with_peak(401, 144))
        self.assertLess(rfi["peak_fraction"], pulsar["peak_fraction"])

    def test_resampling_preserves_peak_location(self):
        # A peak 25% along a 401-point curve must stay ~25% along after resampling.
        vec, _ = dmc_to_vector(payload_with_peak(401, 100))
        self.assertAlmostEqual(
            int(vec.argmax()) / (DM_VECTOR_LENGTH - 1), 0.25, delta=0.05
        )

    def test_custom_length(self):
        vec, _ = dmc_to_vector(payload_with_peak(201, 100), length=16)
        self.assertEqual(tuple(vec.shape), (16,))

    def test_flat_curve_does_not_produce_nan(self):
        vec, _ = dmc_to_vector({"DMc": [3.0] * 101, "meta": {}})
        self.assertTrue(torch.isfinite(vec).all())
        self.assertEqual(float(vec.max()), 0.0)

    def test_missing_axis_metadata_yields_nan_scale_not_a_crash(self):
        _vec, feats = dmc_to_vector({"DMc": [1.0, 2.0, 1.0]})
        self.assertEqual(feats["peak_fraction"], 0.5)
        self.assertTrue(np.isnan(feats["peak_dm"]))

    def test_rejects_unusable_curves(self):
        for bad in ({}, {"DMc": []}, {"DMc": [1.0]}, {"DMc": [1.0, float("nan")]}):
            with self.subTest(payload=bad):
                with self.assertRaises(ValueError):
                    dmc_to_vector(bad)


class TestDoesNotDisturbImagePath(unittest.TestCase):
    def test_arrays_to_tensor_is_unchanged(self):
        # EXP-005 reproducibility depends on the image path being untouched.
        from src.datasets.psc_preprocessing import arrays_to_tensor

        rng = np.random.default_rng(0)
        out = arrays_to_tensor(rng.random((64, 200)), rng.random((24, 200)))
        self.assertEqual(tuple(out.shape), (2, 32, 32))
        self.assertEqual(out.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
