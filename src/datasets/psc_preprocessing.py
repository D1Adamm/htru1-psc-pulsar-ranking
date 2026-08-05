"""Convert PSC candidate JSON arrays into BaselineCNN input tensors."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


TARGET_SIZE = (32, 32)
# Larger than float32 bilinear noise on near-constant maps.
EPS = 1e-6


def _is_ragged_2d(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or not value:
        return False
    if not all(isinstance(row, (list, tuple, np.ndarray)) for row in value):
        return False
    lengths = {len(row) for row in value}
    return len(lengths) > 1


def _to_float32_2d(value: Any, *, name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"missing required array key: {name}")
    if _is_ragged_2d(value):
        raise ValueError(f"{name} has ragged rows; expected a rectangular 2-D array")

    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} could not be converted to float32: {exc}") from exc

    if array.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def _extract_field(payload: dict[str, Any], key: str) -> Any:
    if key in payload:
        return payload[key]
    metadata = payload.get("metadata")
    if isinstance(metadata, dict) and key in metadata:
        return metadata[key]
    meta = payload.get("meta")
    if isinstance(meta, dict) and key in meta:
        return meta[key]
    return None


def _minmax_normalize(channel: torch.Tensor) -> torch.Tensor:
    channel_min = float(channel.min().item())
    channel_max = float(channel.max().item())
    denom = channel_max - channel_min
    if denom < EPS:
        return torch.zeros_like(channel)
    return (channel - channel_min) / denom


def _resize_to_32x32(array: np.ndarray) -> torch.Tensor:
    # (1, 1, H, W) -> bilinear resize -> (32, 32)
    tensor = torch.from_numpy(array).to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(
        tensor,
        size=TARGET_SIZE,
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(0).squeeze(0)


def psc_json_to_tensor(
    payload: dict[str, Any],
    plot_id: int | str | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Convert one PSC JSON payload to a ``(2, 32, 32)`` float32 tensor.

    Channel order matches ``BaselineCNN`` / HTRU1 preprocessing:
      - channel 0 = Frequency–Phase (``FvP``)
      - channel 1 = Time–Phase (``TvP``)

    Returns ``(image, metadata)`` where ``metadata`` records plot id, selected
    survey fields, and the original FvP/TvP shapes before resizing.
    """
    if not isinstance(payload, dict):
        raise TypeError(f"payload must be a dict, got {type(payload).__name__}")

    missing = [name for name in ("FvP", "TvP") if name not in payload]
    if missing:
        raise ValueError(f"missing required array key(s): {', '.join(missing)}")

    fvp = _to_float32_2d(payload["FvP"], name="FvP")
    tvp = _to_float32_2d(payload["TvP"], name="TvP")

    fvp_resized = _minmax_normalize(_resize_to_32x32(fvp))
    tvp_resized = _minmax_normalize(_resize_to_32x32(tvp))
    image = torch.stack([fvp_resized, tvp_resized], dim=0).to(dtype=torch.float32)

    if image.shape != (2, 32, 32):
        raise RuntimeError(f"internal error: expected (2, 32, 32), got {tuple(image.shape)}")
    if not torch.isfinite(image).all():
        raise RuntimeError("normalized image contains NaN or infinite values")

    metadata = {
        "plot_id": plot_id,
        "telescope": _extract_field(payload, "telescope"),
        "period": _extract_field(payload, "period"),
        "sigma": _extract_field(payload, "sigma"),
        "original_fvp_shape": tuple(int(x) for x in fvp.shape),
        "original_tvp_shape": tuple(int(x) for x in tvp.shape),
    }
    return image, metadata
