"""Three-class head on the HTRU1-pretrained BaselineCNN backbone.

HTRU1 is a binary dataset (pulsar / non-pulsar) with no RFI labels, so there is
no such thing as three-class pretraining. This model keeps the convolutional
features learned on HTRU1, discards the binary output layer, and attaches a
fresh three-way head that is trained entirely on PSC AO327 labels.

Class order is fixed and must not be reordered — the evaluation code collapses
RFI and noise into "not pulsar" by index.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.baseline_cnn import BaselineCNN

# Fixed class order. PULSAR must stay at index 0 so binary collapse is
# `1 - softmax[..., 0]`.
CLASS_NAMES = ("known_pulsar", "rfi", "noise")
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}
PULSAR_INDEX = 0
NUM_CLASSES = len(CLASS_NAMES)


class ThreeClassCNN(nn.Module):
    """``BaselineCNN`` features plus a three-way classifier head.

    Returns raw logits of shape ``(B, 3)`` for use with ``CrossEntropyLoss``.
    """

    EXPECTED_CHANNELS = BaselineCNN.EXPECTED_CHANNELS

    def __init__(self) -> None:
        super().__init__()
        base = BaselineCNN()
        self.features = base.features
        # Reuse the pretrained hidden layer; replace only the output layer.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.EXPECTED_CHANNELS:
            raise ValueError(
                f"Expected input shape (B, {self.EXPECTED_CHANNELS}, H, W), "
                f"got {tuple(x.shape)}"
            )
        return self.classifier(self.features(x))

    def load_pretrained_backbone(
        self, state_dict: dict[str, torch.Tensor], *, strict_features: bool = True
    ) -> dict[str, list[str]]:
        """Copy HTRU1-trained weights for every layer whose shape still matches.

        The final ``classifier.4`` layer changes shape (1 output -> 3), so it is
        always skipped and left randomly initialised. Returns which parameters
        were loaded and which were skipped, so callers can log it rather than
        silently trusting a partial load.
        """
        own = self.state_dict()
        loaded: list[str] = []
        skipped: list[str] = []

        for key, value in state_dict.items():
            if key in own and own[key].shape == value.shape:
                own[key] = value.clone()
                loaded.append(key)
            else:
                skipped.append(key)

        self.load_state_dict(own)

        if strict_features:
            missing = [k for k in own if k.startswith("features.") and k not in loaded]
            if missing:
                raise RuntimeError(
                    "pretrained backbone did not cover the feature layers: "
                    f"{missing}. Refusing to train from a partially loaded backbone."
                )
        return {"loaded": loaded, "skipped": skipped}

    def freeze_backbone(self) -> None:
        """Freeze the convolutional features so only the head trains."""
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.features.parameters():
            param.requires_grad = True


def collapse_to_pulsar_score(logits: torch.Tensor) -> torch.Tensor:
    """Three-class logits -> a scalar pulsar score per candidate.

    Softmax probability of the pulsar class. Equivalent to ``1 - P(rfi) -
    P(noise)``, which is what makes the result directly comparable to the binary
    scores in EXP-001 through EXP-003.
    """
    if logits.ndim != 2 or logits.shape[1] != NUM_CLASSES:
        raise ValueError(
            f"expected logits of shape (B, {NUM_CLASSES}), got {tuple(logits.shape)}"
        )
    return torch.softmax(logits, dim=1)[:, PULSAR_INDEX]
