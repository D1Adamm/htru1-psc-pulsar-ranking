"""Small CNN baseline for binary pulsar / non-pulsar classification."""

from __future__ import annotations

import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """Two-block CNN that maps ``(B, 2, 32, 32)`` inputs to a single logit.

    Expects Frequency–Phase and Time–Phase channels. Returns raw logits for
    use with ``BCEWithLogitsLoss`` (no sigmoid in ``forward``).
    """

    EXPECTED_CHANNELS = 2

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.EXPECTED_CHANNELS:
            raise ValueError(
                f"Expected input shape (B, {self.EXPECTED_CHANNELS}, H, W), "
                f"got {tuple(x.shape)}"
            )
        return self.classifier(self.features(x))
