#!/usr/bin/env python3
"""EXP-007 — combine the CNN score with duty cycle and a DM feature.

EXP-006 showed the fine-tuned CNN collapses on an unseen telescope (2/19) while a
duty-cycle threshold chosen on Arecibo transfers perfectly (19/19). This asks a
narrower question: given all three signals, what does each actually contribute?

Fits a logistic regression on the Arecibo *train* split only and evaluates on the
Arecibo held-out split, using the same source-group split as EXP-005 so no
repeat detection of one pulsar straddles train and test.

SELECTION WARNING. The GBNCC RFI set was chosen by DM < 1.0, so any DM-using
model scores near-perfectly on it by rediscovering that filter. GBNCC numbers for
DM-aware variants are printed but labelled selection-confounded; the Arecibo
split is the honest measurement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.psc_dataset import (
    PSCCachedDataset,
    read_manifest_rows,
    split_indices_by_source_group,
)
from src.datasets.psc_preprocessing import dmc_to_vector
from src.models.three_class_head import (
    PULSAR_INDEX,
    ThreeClassCNN,
    collapse_to_pulsar_score,
)

DEFAULT_MANIFEST = REPO_ROOT / "data" / "psc" / "psc_eval_candidates_v3.csv"
DEFAULT_GBNCC = REPO_ROOT / "data" / "psc" / "psc_eval_gbncc_rfi.csv"
DEFAULT_CACHE = REPO_ROOT / "data" / "psc" / "raw"
DEFAULT_CHECKPOINT = REPO_ROOT / "outputs" / "models" / "psc_threeclass_best.pt"
DEFAULT_METRICS = REPO_ROOT / "outputs" / "metrics" / "exp007_hybrid_ranker.json"

FEATURE_NAMES = ("cnn_score", "neg_duty_cycle", "neg_peak_fraction")


def duty_cycle(image: torch.Tensor) -> float:
    """Fraction of phase bins above half maximum. Lower = narrower = pulsar-like."""
    profile = image[0].mean(dim=0)
    lo, hi = float(profile.min()), float(profile.max())
    if hi - lo < 1e-9:
        return 1.0
    return float((((profile - lo) / (hi - lo)) > 0.5).float().mean())


def build_features(
    dataset: PSCCachedDataset, cache_dir: Path, model: ThreeClassCNN
) -> np.ndarray:
    """Return an (N, 3) feature matrix; every column is oriented pulsar-positive."""
    with torch.no_grad():
        scores = collapse_to_pulsar_score(model(torch.stack(dataset.images))).numpy()

    duty = np.array([duty_cycle(img) for img in dataset.images])

    peak = np.empty(len(dataset))
    for i, plot_id in enumerate(dataset.plot_ids):
        with open(cache_dir / f"{plot_id}.json") as f:
            payload = json.load(f)
        _vector, feats = dmc_to_vector(payload)
        peak[i] = feats["peak_fraction"]

    # Negate duty cycle and DM peak fraction so larger always means more
    # pulsar-like, which keeps the fitted coefficients readable.
    return np.column_stack([scores, -duty, -peak])


def fit_and_report(
    X_train,
    y_train,
    X_test,
    y_test,
    subsets: dict[str, tuple],
    used: list[str],
) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(scaler.transform(X_train), y_train)

    if len(used) != X_train.shape[1]:
        raise ValueError(
            f"got {len(used)} feature names for {X_train.shape[1]} columns"
        )
    out = {"features": ", ".join(used), "coefficients": {}}
    for name, coef in zip(used, clf.coef_[0]):
        out["coefficients"][name] = float(coef)

    probs = clf.predict_proba(scaler.transform(X_test))[:, 1]
    out["arecibo_heldout_roc_auc"] = float(roc_auc_score(y_test, probs))

    for subset_name, (Xs, ys) in subsets.items():
        if len(set(ys.tolist())) < 2:
            # Single-class subset: report recall of the negative class instead.
            preds = clf.predict(scaler.transform(Xs))
            correct = int((preds == ys).sum())
            out[subset_name] = f"{correct}/{len(ys)} correct"
        else:
            p = clf.predict_proba(scaler.transform(Xs))[:, 1]
            out[subset_name] = float(roc_auc_score(ys, p))
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--gbncc", type=Path, default=DEFAULT_GBNCC)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--metrics-json", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    rows = read_manifest_rows(args.manifest)
    ds = PSCCachedDataset(rows, args.cache_dir)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ThreeClassCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Arecibo: {len(ds)} candidates")
    X = build_features(ds, args.cache_dir, model)
    y = np.array([1 if lab == PULSAR_INDEX else 0 for lab in ds.labels])

    train_idx, val_idx, test_idx = split_indices_by_source_group(
        ds.source_groups, seed=args.seed
    )
    fit_idx = train_idx + val_idx  # no early stopping needed for logistic regression

    # Pulsar-vs-RFI only, the comparison EXP-003/005/006 all report.
    rfi_mask = np.array([lab == 1 for lab in ds.labels])
    pul_mask = np.array([lab == PULSAR_INDEX for lab in ds.labels])
    pr_test = np.array([i for i in test_idx if rfi_mask[i] or pul_mask[i]])

    gb_rows = read_manifest_rows(args.gbncc)
    gb = PSCCachedDataset(gb_rows, args.cache_dir)
    Xg = build_features(gb, args.cache_dir, model)
    yg = np.zeros(len(gb), dtype=int)  # all RFI, all negatives
    print(f"GBNCC:   {len(gb)} candidates (all RFI, DM-selected)\n")

    combos = {
        "cnn only": [0],
        "duty only": [1],
        "dm only": [2],
        "cnn + duty": [0, 1],
        "duty + dm": [1, 2],
        "cnn + duty + dm": [0, 1, 2],
    }

    results = []
    print(f"{'features':18s} {'Arecibo P-vs-RFI':>17} {'GBNCC 19 RFI':>14}")
    print("-" * 52)
    for label, cols in combos.items():
        used = [FEATURE_NAMES[c] for c in cols]
        res = fit_and_report(
            X[np.ix_(fit_idx, cols)],
            y[fit_idx],
            X[np.ix_(pr_test, cols)],
            y[pr_test],
            {"gbncc": (Xg[:, cols], yg)},
            used,
        )
        flag = "  (confounded)" if 2 in cols else ""
        print(
            f"{label:18s} {res['arecibo_heldout_roc_auc']:17.4f} "
            f"{res['gbncc']:>14}{flag}"
        )
        results.append(res)

    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_json, "w") as f:
        json.dump(
            {
                "results": results,
                "note": "GBNCC RFI was selected by DM < 1.0; any dm feature makes "
                "that column selection-confounded.",
                "arecibo_test_n": int(len(pr_test)),
                "gbncc_n": int(len(gb)),
                "seed": args.seed,
            },
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")
    print(f"\nSaved: {args.metrics_json}")


if __name__ == "__main__":
    main()
