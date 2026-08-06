#!/usr/bin/env python3
"""Diagnose why the HTRU1-trained model cannot separate AO327 pulsars from RFI.

Runs three read-only diagnostics over cached PSC JSON and the HTRU1 test batch:

  D1  orientation  -- is the phase axis the same in HTRU1 and PSC arrays?
  D2  visual       -- render preprocessed FvP/TvP for each category
  D3  normalization-- does the normalization choice change separability?

Writes figures to outputs/figures/ and a stats JSON to outputs/metrics/.
Never fetches from the network, never retrains, never overwrites EXP-002 output.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.htru1_dataset import HTRU1Dataset
from src.datasets.psc_preprocessing import arrays_to_tensor

DEFAULT_CACHE = REPO_ROOT / "data" / "psc" / "raw"
DEFAULT_MANIFEST = REPO_ROOT / "data" / "psc" / "psc_eval_candidates.csv"
DEFAULT_RANKED = REPO_ROOT / "outputs" / "predictions" / "psc_ranked_candidates.csv"
DEFAULT_CHECKPOINT = REPO_ROOT / "outputs" / "models" / "baseline_cnn_best.pt"
DEFAULT_TEST_BATCH = REPO_ROOT / "data" / "raw" / "htru1-batches-py" / "test_batch"
DEFAULT_FIGURE_DIR = REPO_ROOT / "outputs" / "figures"
DEFAULT_STATS_JSON = REPO_ROOT / "outputs" / "metrics" / "exp003_domain_shift.json"

EPS = 1e-6


# --------------------------------------------------------------------------- D1


def peakiness(vector: np.ndarray) -> float:
    """How sharply peaked a 1-D collapse is, in units of its own std.

    A folded pulsar is compact in phase and extended in frequency/time, so the
    collapse along the phase axis is peaked and the other is comparatively flat.
    """
    v = np.asarray(vector, dtype=np.float64).reshape(-1)
    std = v.std()
    if std < EPS:
        return 0.0
    return float((v.max() - np.median(v)) / std)


def axis_peakiness(array: np.ndarray) -> tuple[float, float]:
    """Return ``(peakiness_along_columns, peakiness_along_rows)`` for a 2-D array."""
    a = np.asarray(array, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {a.shape}")
    return peakiness(a.mean(axis=0)), peakiness(a.mean(axis=1))


def infer_phase_axis(arrays: list[np.ndarray]) -> tuple[int, float, float]:
    """Infer which axis carries pulse phase across many pulsar arrays.

    Returns ``(axis, mean_column_peakiness, mean_row_peakiness)`` where axis 1
    means phase runs along columns.
    """
    if not arrays:
        raise ValueError("need at least one array to infer the phase axis")
    cols, rows = zip(*(axis_peakiness(a) for a in arrays))
    mean_cols, mean_rows = float(np.mean(cols)), float(np.mean(rows))
    return (1 if mean_cols > mean_rows else 0), mean_cols, mean_rows


# --------------------------------------------------------------------------- D3


def _resize(a: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(a)).float()[None, None]
    return F.interpolate(t, size=(32, 32), mode="bilinear", align_corners=False)[0, 0]


def _minmax(c: torch.Tensor) -> torch.Tensor:
    lo, hi = float(c.min()), float(c.max())
    if hi - lo < EPS:
        return torch.zeros_like(c)
    return (c - lo) / (hi - lo)


def norm_per_channel_minmax(fvp: np.ndarray, tvp: np.ndarray) -> torch.Tensor:
    """The shipped preprocessing, via the shared core in psc_preprocessing."""
    return arrays_to_tensor(fvp, tvp)


def norm_global_minmax(fvp: np.ndarray, tvp: np.ndarray) -> torch.Tensor:
    """One min-max over both channels jointly, preserving relative channel scale."""
    stacked = torch.stack([_resize(fvp), _resize(tvp)])
    lo, hi = float(stacked.min()), float(stacked.max())
    if hi - lo < EPS:
        return torch.zeros_like(stacked)
    return (stacked - lo) / (hi - lo)


def norm_per_row_baseline(fvp: np.ndarray, tvp: np.ndarray) -> torch.Tensor:
    """Subtract each row's median before scaling, removing per-subband gain offsets."""
    out = []
    for a in (fvp, tvp):
        r = _resize(a)
        out.append(_minmax(r - r.median(dim=1, keepdim=True).values))
    return torch.stack(out)


NORMALIZATIONS: dict[str, Callable[[np.ndarray, np.ndarray], torch.Tensor]] = {
    "per_channel_minmax_current": norm_per_channel_minmax,
    "global_minmax": norm_global_minmax,
    "per_row_baseline": norm_per_row_baseline,
}


# --------------------------------------------------------------------------- IO


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    with open(path, newline="") as f:
        return {r["plot_id"]: r for r in csv.DictReader(f)}


def load_cached(cache_dir: Path, plot_id: str) -> dict:
    with open(cache_dir / f"{plot_id}.json") as f:
        return json.load(f)


def _load_predict_module():
    spec = importlib.util.spec_from_file_location(
        "predict_psc_candidate", SCRIPTS_DIR / "predict_psc_candidate.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> None:
    from sklearn.metrics import roc_auc_score
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--ranked", type=Path, default=DEFAULT_RANKED)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--test-batch", type=Path, default=DEFAULT_TEST_BATCH)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--stats-json", type=Path, default=DEFAULT_STATS_JSON)
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    cache = {pid: load_cached(args.cache_dir, pid) for pid in manifest}
    with open(args.ranked, newline="") as f:
        scores = {r["plot_id"]: float(r["score"]) for r in csv.DictReader(f) if r["score"]}

    stats: dict[str, object] = {}

    # ---- D1 orientation ----------------------------------------------------
    ds = HTRU1Dataset([args.test_batch])
    htru_pulsars = ds.images.numpy()[ds.labels.numpy() == 1.0]
    h_axis, h_cols, h_rows = infer_phase_axis([x[0] for x in htru_pulsars])

    psc_pulsar_ids = [p for p, r in manifest.items() if r["source_category"] == "known_pulsar"]
    p_axis, p_cols, p_rows = infer_phase_axis(
        [np.asarray(cache[p]["FvP"], dtype=float) for p in psc_pulsar_ids]
    )

    stats["d1_orientation"] = {
        "htru1_phase_axis": h_axis,
        "htru1_col_peakiness": h_cols,
        "htru1_row_peakiness": h_rows,
        "psc_phase_axis": p_axis,
        "psc_col_peakiness": p_cols,
        "psc_row_peakiness": p_rows,
        "axes_match": h_axis == p_axis,
    }
    print("D1 orientation:")
    print(f"   HTRU1 phase axis = {h_axis} (cols {h_cols:.2f} vs rows {h_rows:.2f})")
    print(f"   PSC   phase axis = {p_axis} (cols {p_cols:.2f} vs rows {p_rows:.2f})")
    print(f"   -> axes {'MATCH; no transpose needed' if h_axis == p_axis else 'DIFFER; TRANSPOSE BUG'}")

    # ---- D3 normalization --------------------------------------------------
    predict = _load_predict_module()
    device = torch.device("cpu")
    model = predict.load_baseline_cnn(args.checkpoint, device)

    cats = {c: [p for p, r in manifest.items() if r["source_category"] == c]
            for c in ("known_pulsar", "rfi", "noise")}

    d3: dict[str, dict[str, float]] = {}
    print("\nD3 normalization:")
    print(f"   {'variant':28s} {'pulsar-vs-RFI':>14} {'pulsar-vs-noise':>16}")
    for name, fn in NORMALIZATIONS.items():
        s = {}
        for pid, payload in cache.items():
            fvp = np.asarray(payload["FvP"], dtype=float)
            tvp = np.asarray(payload["TvP"], dtype=float)
            with torch.no_grad():
                s[pid] = float(torch.sigmoid(model(fn(fvp, tvp)[None].float())).item())
        P = [s[p] for p in cats["known_pulsar"]]
        R = [s[p] for p in cats["rfi"]]
        N = [s[p] for p in cats["noise"]]
        auc_r = float(roc_auc_score([1] * len(P) + [0] * len(R), P + R))
        auc_n = float(roc_auc_score([1] * len(P) + [0] * len(N), P + N))
        d3[name] = {"pulsar_vs_rfi_auc": auc_r, "pulsar_vs_noise_auc": auc_n}
        print(f"   {name:28s} {auc_r:14.4f} {auc_n:16.4f}")
    stats["d3_normalization"] = d3

    # ---- D2 figure ---------------------------------------------------------
    def top(cat: str, n: int, reverse=True):
        ids = sorted(cats[cat], key=lambda p: scores.get(p, 0.0), reverse=reverse)
        return ids[:n]

    panels = (
        [("pulsar", p) for p in top("known_pulsar", 4)]
        + [("RFI", p) for p in top("rfi", 4)]
        + [("pulsar MISSED", p) for p in top("known_pulsar", 4, reverse=False)]
        + [("noise", p) for p in top("noise", 4)]
    )

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 8, figsize=(20, 11))
    for i, (label, pid) in enumerate(panels):
        payload = cache[pid]
        img = arrays_to_tensor(
            np.asarray(payload["FvP"], dtype=float),
            np.asarray(payload["TvP"], dtype=float),
        ).numpy()
        r, c = divmod(i, 4)
        for ch, chname in ((0, "FvP"), (1, "TvP")):
            ax = axes[r][c * 2 + ch]
            ax.imshow(img[ch], aspect="auto", origin="lower", cmap="magma")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{label} {pid}\n{chname}  score={scores.get(pid, float('nan')):.3f}",
                         fontsize=8)
    fig.suptitle(
        "EXP-003 D2 — preprocessed (2,32,32) inputs by category. "
        "Phase runs along x; frequency (FvP) / time (TvP) along y.",
        fontsize=12,
    )
    fig.tight_layout()
    out_fig = args.figure_dir / "exp003_category_comparison.png"
    fig.savefig(out_fig, dpi=110)
    plt.close(fig)
    print(f"\nD2 figure: {out_fig}")

    args.stats_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.stats_json, "w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Stats: {args.stats_json}")


if __name__ == "__main__":
    main()
