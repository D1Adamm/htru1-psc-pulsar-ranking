#!/usr/bin/env python3
"""Download and inspect one PSC AO327 candidate by plot ID.

Inspection only: no resize, normalize, CNN inference, or local data saves.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import requests

PSC_PLOT_JSON_URL = (
    "https://pulsars.nanograv.org/api/psrsearch/plots/{plot_id}/json"
)
ARRAY_KEYS = ("profile", "FvP", "TvP", "DMc")
DEFAULT_TIMEOUT_S = 30.0

# Print these metadata fields when present (order preserved).
USEFUL_METADATA_KEYS = (
    "id",
    "plot_id",
    "filename",
    "survey",
    "telescope",
    "ra",
    "dec",
    "dm",
    "period",
    "p0",
    "snr",
    "sigma",
    "bary_period",
    "topo_period",
    "candidate_id",
    "pfd_filename",
)


def build_plot_url(plot_id: str | int) -> str:
    """Return the PSC plot JSON endpoint for ``plot_id``."""
    return PSC_PLOT_JSON_URL.format(plot_id=str(plot_id))


def fetch_plot_json(
    plot_id: str | int,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """GET candidate JSON from PSC with timeout and clear HTTP errors."""
    if timeout_s <= 0:
        raise ValueError(f"timeout_s must be positive, got {timeout_s}")

    url = build_plot_url(plot_id)
    http = session or requests
    try:
        response = http.get(
            url,
            headers={"Accept": "application/json"},
            timeout=timeout_s,
        )
    except requests.Timeout as exc:
        raise RuntimeError(
            f"PSC request timed out after {timeout_s}s for plot_id={plot_id} ({url})"
        ) from exc
    except requests.RequestException as exc:
        raise RuntimeError(
            f"PSC network error for plot_id={plot_id}: {exc} ({url})"
        ) from exc

    if not response.ok:
        raise RuntimeError(
            f"PSC HTTP {response.status_code} for plot_id={plot_id}: "
            f"{response.reason} ({url})"
        )

    try:
        payload = response.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"PSC response for plot_id={plot_id} is not valid JSON ({url})"
        ) from exc

    if not isinstance(payload, dict):
        raise RuntimeError(
            f"PSC response for plot_id={plot_id} must be a JSON object, "
            f"got {type(payload).__name__}"
        )
    return payload


def extract_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a metadata mapping from common PSC payload layouts."""
    if isinstance(payload.get("metadata"), dict):
        return dict(payload["metadata"])
    if isinstance(payload.get("meta"), dict):
        return dict(payload["meta"])

    reserved = set(ARRAY_KEYS) | {"metadata", "meta", "data", "arrays"}
    return {key: value for key, value in payload.items() if key not in reserved}


def to_numpy_array(value: Any, *, name: str) -> np.ndarray:
    """Convert a JSON array value to ``np.ndarray`` for inspection only."""
    if value is None:
        raise ValueError(f"{name}: array is missing (null)")
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: could not convert to NumPy array ({exc})") from exc


def is_ragged_sequence(value: Any) -> bool:
    """Return True if ``value`` looks like a nested sequence with uneven lengths."""
    if not isinstance(value, (list, tuple)) or not value:
        return False
    if not all(isinstance(row, (list, tuple)) for row in value):
        return False
    lengths = {len(row) for row in value}
    return len(lengths) > 1


def inspect_array(
    value: Any,
    *,
    name: str,
    expected_ndim: int | None = None,
) -> dict[str, Any]:
    """Validate one candidate array and return an inspection summary."""
    result: dict[str, Any] = {
        "name": name,
        "present": value is not None,
        "shape": None,
        "dtype": None,
        "issues": [],
    }

    if value is None:
        result["issues"].append("missing")
        return result

    if is_ragged_sequence(value):
        result["issues"].append("ragged")
        array = np.asarray(value, dtype=object)
        result["shape"] = tuple(array.shape)
        result["dtype"] = str(array.dtype)
        return result

    try:
        array = to_numpy_array(value, name=name)
    except ValueError as exc:
        result["issues"].append(str(exc))
        return result

    result["shape"] = tuple(array.shape)
    result["dtype"] = str(array.dtype)

    if array.dtype == object:
        result["issues"].append("object_dtype_or_ragged")

    if expected_ndim is not None and array.ndim != expected_ndim:
        result["issues"].append(
            f"unexpected_ndim:{array.ndim}(expected_{expected_ndim})"
        )

    if array.dtype != object:
        if np.isnan(array).any():
            result["issues"].append("nan")
        if np.isinf(array).any():
            result["issues"].append("inf")

    return result


def inspect_candidate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Inspect top-level structure, arrays, and selected metadata."""
    if not isinstance(payload, dict):
        raise TypeError(f"payload must be a dict, got {type(payload).__name__}")

    metadata = extract_metadata(payload)
    useful_metadata = {
        key: metadata[key] for key in USEFUL_METADATA_KEYS if key in metadata
    }

    # Soft expectations: profile usually 1-D; FvP/TvP usually 2-D.
    expected_ndims = {
        "profile": 1,
        "FvP": 2,
        "TvP": 2,
        "DMc": None,
    }

    arrays = {
        name: inspect_array(
            payload.get(name),
            name=name,
            expected_ndim=expected_ndims[name],
        )
        for name in ARRAY_KEYS
    }

    return {
        "top_level_keys": sorted(payload.keys()),
        "metadata_keys": sorted(metadata.keys()),
        "useful_metadata": useful_metadata,
        "arrays": arrays,
    }


def format_inspection_report(report: dict[str, Any]) -> str:
    """Pretty-print an inspection report for the CLI."""
    lines: list[str] = []
    lines.append(f"Top-level JSON keys ({len(report['top_level_keys'])}):")
    lines.append("  " + ", ".join(report["top_level_keys"]))
    lines.append("")

    for name in ARRAY_KEYS:
        info = report["arrays"][name]
        shape = info["shape"]
        issues = info["issues"]
        lines.append(f"{name} shape: {shape}")
        if issues:
            lines.append(f"  issues: {', '.join(issues)}")

    lines.append("")
    lines.append(f"Metadata keys ({len(report['metadata_keys'])}):")
    if report["metadata_keys"]:
        lines.append("  " + ", ".join(report["metadata_keys"]))
    else:
        lines.append("  (none)")

    lines.append("")
    lines.append("Selected useful metadata:")
    useful = report["useful_metadata"]
    if useful:
        for key, value in useful.items():
            lines.append(f"  {key}: {value}")
    else:
        lines.append("  (none of the preferred keys were present)")

    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plot_id", help="PSC plot ID, e.g. 11732926")
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help=f"HTTP timeout in seconds (default: {DEFAULT_TIMEOUT_S})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    payload = fetch_plot_json(args.plot_id, timeout_s=args.timeout)
    report = inspect_candidate_payload(payload)
    print(format_inspection_report(report))


if __name__ == "__main__":
    main()
