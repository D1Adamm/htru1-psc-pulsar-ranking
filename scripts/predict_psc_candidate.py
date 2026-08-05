#!/usr/bin/env python3
"""Score one PSC AO327 candidate with a trained HTRU1 BaselineCNN.

This is an experimental cross-survey score, not a confirmed scientific
classification.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.psc_preprocessing import psc_json_to_tensor
from src.models.baseline_cnn import BaselineCNN

DEFAULT_CHECKPOINT = REPO_ROOT / "outputs" / "models" / "baseline_cnn_best.pt"
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_THRESHOLD = 0.5


def _load_inspect_module():
    path = SCRIPTS_DIR / "inspect_psc_candidate.py"
    spec = importlib.util.spec_from_file_location("inspect_psc_candidate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_inspect = _load_inspect_module()
fetch_plot_json = _inspect.fetch_plot_json


def select_device() -> torch.device:
    """Pick CUDA, Apple MPS, or CPU depending on availability."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_baseline_cnn(
    checkpoint_path: Path,
    device: torch.device,
) -> BaselineCNN:
    """Load BaselineCNN weights from a training checkpoint."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"{checkpoint_path} is missing model_state_dict; "
            "expected a train_baseline.py checkpoint"
        )

    model = BaselineCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def score_image(
    model: BaselineCNN,
    image: torch.Tensor,
    device: torch.device,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """Run one ``(2, 32, 32)`` image through the model and return scores."""
    if tuple(image.shape) != (2, 32, 32):
        raise ValueError(f"expected image shape (2, 32, 32), got {tuple(image.shape)}")

    batch = image.unsqueeze(0).to(device=device, dtype=torch.float32)
    if tuple(batch.shape) != (1, 2, 32, 32):
        raise ValueError(f"expected batch shape (1, 2, 32, 32), got {tuple(batch.shape)}")

    model.eval()
    logit = model(batch).view(-1)[0]
    probability = torch.sigmoid(logit)
    predicted_label = int((probability >= threshold).item())

    return {
        "processed_tensor_shape": tuple(batch.shape),
        "raw_logit": float(logit.item()),
        "pulsar_probability": float(probability.item()),
        "threshold": float(threshold),
        "predicted_label": predicted_label,
    }


def score_psc_payload(
    payload: dict[str, Any],
    model: BaselineCNN,
    device: torch.device,
    *,
    plot_id: int | str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """Preprocess a PSC JSON payload and score it with BaselineCNN."""
    image, metadata = psc_json_to_tensor(payload, plot_id=plot_id)
    scores = score_image(model, image, device=device, threshold=threshold)
    return {**metadata, **scores}


def format_prediction_report(result: dict[str, Any]) -> str:
    """Pretty-print an experimental cross-survey prediction."""
    lines = [
        "Experimental cross-survey PSC score",
        "(not a confirmed scientific classification)",
        "",
        f"Plot ID: {result.get('plot_id')}",
        f"Telescope: {result.get('telescope')}",
        f"Period: {result.get('period')}",
        f"Sigma: {result.get('sigma')}",
        f"Original FvP shape: {result.get('original_fvp_shape')}",
        f"Original TvP shape: {result.get('original_tvp_shape')}",
        f"Processed tensor shape: {result.get('processed_tensor_shape')}",
        f"Raw logit: {result.get('raw_logit'):.6f}",
        f"Pulsar probability: {result.get('pulsar_probability'):.6f}",
        (
            f"Threshold prediction (@ {result.get('threshold')}): "
            f"{result.get('predicted_label')} "
            f"({'pulsar' if result.get('predicted_label') == 1 else 'non-pulsar'})"
        ),
    ]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plot_id", help="PSC plot ID, e.g. 11732926")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to baseline_cnn_best.pt",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help=f"HTTP timeout in seconds (default: {DEFAULT_TIMEOUT_S})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Probability threshold for the hard label (default: 0.5)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = select_device()
    print(f"Device: {device}")

    payload = fetch_plot_json(args.plot_id, timeout_s=args.timeout)
    model = load_baseline_cnn(args.checkpoint, device=device)
    result = score_psc_payload(
        payload,
        model,
        device,
        plot_id=args.plot_id,
        threshold=args.threshold,
    )
    print(format_prediction_report(result))


if __name__ == "__main__":
    main()
