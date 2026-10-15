#!/usr/bin/env python3
"""Render poster FvP / TvP images for three real PSC AO327 candidates.

Plots the PSC arrays exactly as served by
https://pulsars.nanograv.org/api/psrsearch/plots/{plot_id}/json -- no resizing,
smoothing, clipping, or renormalisation. The served arrays are already scaled
to [0, 1], so every panel uses one fixed colour scale (vmin=0, vmax=1) and
``interpolation="nearest"`` so each pixel is one real phase bin.

Usage:
    python poster_assets/make_poster_figures.py --json-dir <dir with {id}.json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent
DPI = 300
CMAP = "viridis"

# (plot_id, class label, filename slug) -- column order in the combined figure.
CANDIDATES = [
    ("7914553", "Noise", "noise"),
    ("4502281", "RFI", "rfi"),
    ("9886890", "Pulsar", "pulsar"),
]

PANELS = {
    "FvP": ("Frequency vs Phase", "Frequency sub-band (index)"),
    "TvP": ("Time vs Phase", "Time sub-integration (index)"),
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    }
)


def load(json_dir: Path, plot_id: str) -> dict:
    with open(json_dir / f"{plot_id}.json") as f:
        payload = json.load(f)
    for key in ("FvP", "TvP"):
        arr = np.asarray(payload[key], dtype=float)
        if arr.ndim != 2 or not np.isfinite(arr).all():
            raise ValueError(f"{plot_id} {key}: unusable array {arr.shape}")
    return payload


def draw(ax, array: np.ndarray, ylabel: str):
    """One panel: raw array, phase on x, sub-band/sub-integration on y."""
    n_rows, n_bins = array.shape
    image = ax.imshow(
        array,
        cmap=CMAP,
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        # Pixel edges at phase 0..1 so the x axis reads in rotations.
        extent=(0.0, 1.0, -0.5, n_rows - 0.5),
    )
    ax.set_xlabel("Pulse phase (rotations)")
    ax.set_ylabel(ylabel)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    return image


def caption(payload: dict, key: str) -> str:
    meta = payload["meta"]
    n_rows, n_bins = np.asarray(payload[key]).shape
    return (
        f"P = {meta['period'] * 1e3:.1f} ms   DM = {meta['bestdm']:.1f}   "
        f"{n_rows}×{n_bins}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-dir", type=Path, required=True)
    args = parser.parse_args()

    payloads = {pid: load(args.json_dir, pid) for pid, _, _ in CANDIDATES}

    # Six individual images.
    for pid, label, slug in CANDIDATES:
        for key, (long_name, ylabel) in PANELS.items():
            fig, ax = plt.subplots(figsize=(6, 5))
            image = draw(ax, np.asarray(payloads[pid][key], dtype=float), ylabel)
            ax.set_title(f"{label} — PSC {pid}\n{long_name} ({key})")
            ax.text(
                0.5, -0.2, caption(payloads[pid], key), transform=ax.transAxes,
                ha="center", va="top", fontsize=10, color="0.3",
            )
            cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Normalised intensity (as served by PSC)")
            fig.tight_layout()
            fig.savefig(OUT_DIR / f"psc_{pid}_{slug}_{key}.png", dpi=DPI)
            plt.close(fig)

    # Combined 2 x 3 comparison: rows FvP/TvP, columns Noise/RFI/Pulsar.
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), constrained_layout=True)
    for col, (pid, label, _slug) in enumerate(CANDIDATES):
        for row, (key, (long_name, ylabel)) in enumerate(PANELS.items()):
            ax = axes[row][col]
            image = draw(ax, np.asarray(payloads[pid][key], dtype=float), ylabel)
            if row == 0:
                ax.set_title(f"{label} — PSC {pid}", fontweight="bold")
            if col != 0:
                ax.set_ylabel("")
            if col == 0:
                ax.set_ylabel(f"{long_name} ({key})\n{ylabel}")
            ax.text(
                0.98, 0.97, caption(payloads[pid], key), transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color="white",
                bbox=dict(boxstyle="round,pad=0.25", fc="black", alpha=0.45, lw=0),
            )
    cbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Normalised intensity (as served by PSC)")
    fig.suptitle(
        "Real PSC AO327 (Arecibo, 327 MHz) candidates — native-resolution FvP and TvP",
        fontsize=17,
    )
    fig.savefig(OUT_DIR / "psc_fvp_tvp_comparison.png", dpi=DPI)
    plt.close(fig)
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
