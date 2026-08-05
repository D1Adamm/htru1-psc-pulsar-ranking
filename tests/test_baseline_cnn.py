"""Tests for BaselineCNN."""

from __future__ import annotations

import unittest

import torch

from src.models.baseline_cnn import BaselineCNN


class TestBaselineCNN(unittest.TestCase):
    def test_forward_shape_and_dtype(self) -> None:
        model = BaselineCNN()
        model.eval()
        x = torch.randn(64, 2, 32, 32, dtype=torch.float32)

        with torch.no_grad():
            logits = model(x)

        self.assertEqual(tuple(logits.shape), (64, 1))
        self.assertEqual(logits.dtype, torch.float32)

    def test_backward_propagation(self) -> None:
        model = BaselineCNN()
        model.train()
        x = torch.randn(8, 2, 32, 32, dtype=torch.float32, requires_grad=True)
        logits = model(x)
        loss = logits.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(any(p.grad is not None for p in model.parameters()))

    def test_wrong_channel_count_raises(self) -> None:
        model = BaselineCNN()
        x = torch.randn(4, 3, 32, 32, dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, r"\(B, 2, H, W\)"):
            model(x)


if __name__ == "__main__":
    unittest.main()
