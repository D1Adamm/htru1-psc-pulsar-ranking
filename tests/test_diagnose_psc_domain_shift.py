"""Tests for the EXP-003 domain-shift diagnostics. Synthetic arrays, no network."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"_test_{name}", REPO_ROOT / "scripts" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


diag = _load("diagnose_psc_domain_shift")


def fake_folded(n_freq: int, n_phase: int, phase_axis: int = 1, seed: int = 0) -> np.ndarray:
    """A synthetic folded candidate: narrow pulse in phase, extended in frequency."""
    rng = np.random.default_rng(seed)
    prof = rng.normal(0.0, 0.05, n_phase)
    prof[n_phase // 2] += 5.0
    prof[n_phase // 2 - 1] += 2.0
    arr = np.tile(prof, (n_freq, 1))
    arr = arr + rng.normal(0.0, 0.05, arr.shape)
    return arr if phase_axis == 1 else arr.T


class TestPeakiness(unittest.TestCase):
    def test_peaked_beats_flat(self):
        flat = np.ones(64) + np.random.default_rng(0).normal(0, 0.01, 64)
        peaked = flat.copy()
        peaked[32] += 10.0
        self.assertGreater(diag.peakiness(peaked), diag.peakiness(flat))

    def test_constant_vector_is_zero(self):
        self.assertEqual(diag.peakiness(np.full(32, 3.0)), 0.0)


class TestAxisPeakiness(unittest.TestCase):
    def test_rejects_non_2d(self):
        with self.assertRaises(ValueError):
            diag.axis_peakiness(np.zeros(10))

    def test_identifies_phase_along_columns(self):
        cols, rows = diag.axis_peakiness(fake_folded(64, 200, phase_axis=1))
        self.assertGreater(cols, rows)

    def test_identifies_phase_along_rows_when_transposed(self):
        cols, rows = diag.axis_peakiness(fake_folded(64, 200, phase_axis=0))
        self.assertGreater(rows, cols)


class TestInferPhaseAxis(unittest.TestCase):
    def test_detects_column_phase_axis(self):
        arrays = [fake_folded(64, 200, phase_axis=1, seed=i) for i in range(8)]
        axis, cols, rows = diag.infer_phase_axis(arrays)
        self.assertEqual(axis, 1)
        self.assertGreater(cols, rows)

    def test_detects_transposed_input(self):
        arrays = [fake_folded(64, 200, phase_axis=0, seed=i) for i in range(8)]
        axis, _cols, _rows = diag.infer_phase_axis(arrays)
        self.assertEqual(axis, 0, "a transposed survey must be detected, not silently accepted")

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            diag.infer_phase_axis([])


class TestNormalizations(unittest.TestCase):
    def setUp(self):
        self.fvp = fake_folded(128, 200, seed=1)
        self.tvp = fake_folded(24, 200, seed=2)

    def test_all_variants_produce_valid_model_input(self):
        for name, fn in diag.NORMALIZATIONS.items():
            with self.subTest(normalization=name):
                out = fn(self.fvp, self.tvp)
                self.assertEqual(tuple(out.shape), (2, 32, 32), name)
                self.assertTrue(torch.isfinite(out).all(), name)

    def test_current_variant_matches_shipped_preprocessing(self):
        from src.datasets.psc_preprocessing import arrays_to_tensor

        expected = arrays_to_tensor(self.fvp, self.tvp)
        got = diag.NORMALIZATIONS["per_channel_minmax_current"](self.fvp, self.tvp)
        self.assertTrue(
            torch.allclose(expected, got),
            "the 'current' diagnostic variant must be the shipped preprocessing",
        )

    def test_minmax_variants_are_unit_range(self):
        for name in ("per_channel_minmax_current", "global_minmax", "per_row_baseline"):
            out = diag.NORMALIZATIONS[name](self.fvp, self.tvp)
            self.assertGreaterEqual(float(out.min()), -1e-6, name)
            self.assertLessEqual(float(out.max()), 1.0 + 1e-6, name)

    def test_constant_input_does_not_produce_nan(self):
        flat = np.ones((64, 100))
        for name, fn in diag.NORMALIZATIONS.items():
            with self.subTest(normalization=name):
                out = fn(flat, flat)
                self.assertTrue(torch.isfinite(out).all(), name)


if __name__ == "__main__":
    unittest.main()
