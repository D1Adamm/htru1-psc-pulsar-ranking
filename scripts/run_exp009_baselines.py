#!/usr/bin/env python3
"""Refit the EXP-009 forest baselines on the frozen folds, within and across surveys.

Forests train only on training folds and tune leaf size on the validation fold.
Label-uncertain rows are excluded from fitting and tuning, and are scored. Rows and
sources are down-weighted by unit size, then balanced by class. Statistics that need
no fitting (duty cycle, sigma, existing HTRU1 scores) are computed at evaluation.

Writes only under ``outputs/exp009/baselines/`` and refuses to overwrite.
``--dry-run`` prints the plan and writes nothing.
"""

from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.psc_dataset import read_manifest_rows
from src.datasets.psc_folds import (
    BACKENDS,
    assert_under,
    attach_identity,
    fold_vector,
    load_candidate_records,
    load_folds,
    read_identity,
    rotation,
    sha256_file,
    shared_sources,
    write_exclusive,
)
from src.datasets.psc_preprocessing import arrays_to_tensor
from src.models.three_class_head import CLASS_TO_INDEX
from src.psc_metrics import duty_cycle

DEFAULT_MANIFEST = REPO_ROOT / "data" / "psc" / "psc_eval_multisurvey.csv"
DEFAULT_CACHE = REPO_ROOT / "data" / "psc" / "raw"
DEFAULT_IDENTITY = REPO_ROOT / "data" / "psc" / "identity" / "source_identity_v1.csv"
DEFAULT_FOLDS = REPO_ROOT / "data" / "psc" / "folds" / "folds_v1.json"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "exp009" / "baselines"
LEAF_GRID = (1, 2, 5, 10, 25)
DIRECTIONS = (("ao327", "gbncc"), ("gbncc", "ao327"))


def image_features(records: list[dict], cache_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Duty cycle on the CNN input, and where the DM curve peaks in its searched range."""
    duty, dm_peak = [], []
    for rec in records:
        with open(cache_dir / f"{rec['plot_id']}.json") as f:
            payload = json.load(f)
        image = arrays_to_tensor(np.asarray(payload["FvP"], dtype=float), np.asarray(payload["TvP"], dtype=float))
        duty.append(duty_cycle(image))
        curve = np.asarray(payload["DMc"], dtype=float)
        dm_peak.append(curve.argmax() / (len(curve) - 1))
    return np.array(duty), np.array(dm_peak)


def feature_sets(records: list[dict], duty: np.ndarray, dm_peak: np.ndarray) -> dict[str, np.ndarray]:
    get = lambda key: np.array([r[key] for r in records], dtype=float)  # noqa: E731
    period = np.log10(get("period_ms"))
    dm = np.log10(get("dm") + 0.1)
    geometry = np.c_[get("phase_bins"), get("subbands"), get("subints")]
    acquisition = np.c_[get("t_sample"), get("t_obs")]
    backend = np.array([[r["backend"] == b for b in BACKENDS] for r in records], dtype=float)
    significance = np.c_[get("sigma"), np.log10(get("chi2"))]
    return {
        "period_dm": np.c_[period, dm],
        "geometry_acquisition": np.c_[geometry, acquisition],
        "sigma_chi2": significance,
        "all_metadata": np.c_[period, dm, geometry, acquisition, backend, significance],
        "duty_forest": duty[:, None],
        "duty_dmc": np.c_[duty, dm_peak],
    }


def unit_of(record: dict) -> tuple:
    if record["category"] == "known_pulsar":
        return ("source", record["source_id"])
    if record["category"] == "rfi":
        return ("rfi", record["survey"], record["night"])
    return ("noise", record["obs"])


def balanced_weights(rows: np.ndarray, records: list[dict], labels: np.ndarray) -> np.ndarray:
    """Each unit counts once, then each class gets equal total weight."""
    units = [unit_of(records[i]) for i in rows]
    sizes = collections.Counter(units)
    weights = np.array([1 / sizes[u] for u in units])
    for c in range(3):
        mask = labels[rows] == c
        if mask.any():
            weights[mask] /= weights[mask].sum()
    return weights * len(rows) / 3


def fit_forest(X, labels, records, train, val):
    """Choose leaf size by weighted validation log-loss, then refit on training rows."""
    train_weights = balanced_weights(train, records, labels)
    val_weights = balanced_weights(val, records, labels)
    best_loss, best_leaf = None, None
    for leaf in LEAF_GRID:
        forest = RandomForestClassifier(150, min_samples_leaf=leaf, random_state=0, n_jobs=-1)
        forest.fit(X[train], labels[train], sample_weight=train_weights)
        loss = log_loss(labels[val], forest.predict_proba(X[val]), labels=[0, 1, 2], sample_weight=val_weights)
        if best_loss is None or loss < best_loss:
            best_loss, best_leaf = loss, leaf
    forest = RandomForestClassifier(300, min_samples_leaf=best_leaf, random_state=0, n_jobs=-1)
    forest.fit(X[train], labels[train], sample_weight=train_weights)
    return forest, best_leaf


def within_splits(fold: np.ndarray, uncertain: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parts = rotation(k)
    return (
        np.where(np.isin(fold, parts["train"]) & ~uncertain)[0],
        np.where(np.isin(fold, parts["val"]) & ~uncertain)[0],
        np.where(fold == k)[0],
    )


def cross_splits(fold, uncertain, survey, keep, source, target):
    in_source = (survey == source) & keep & ~uncertain
    return (
        np.where(in_source & (fold != 0))[0],
        np.where(in_source & (fold == 0))[0],
        np.where((survey == target) & keep)[0],
    )


def render_scores(rows: list[list]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["plot_id", "model", "assignment", "evaluation", "p_pulsar", "p_rfi", "p_noise"])
    writer.writerows(rows)
    return buffer.getvalue()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print the plan; fit and write nothing")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--identity", type=Path, default=DEFAULT_IDENTITY)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    records = load_candidate_records(args.manifest, args.cache_dir)
    attach_identity(records, read_identity(args.identity))
    folds = load_folds(args.folds)
    plot_ids = [r["plot_id"] for r in records]
    labels = np.array([CLASS_TO_INDEX[r["category"]] for r in records])
    uncertain = np.array([r["label_uncertain"] for r in records])
    survey = np.array([r["survey"] for r in records])
    shared = shared_sources(records)
    keep = np.array([r["source_id"] not in shared for r in records])

    print(f"{len(records)} rows, {int(uncertain.sum())} excluded from fitting, shared sources removed from cross-survey: {sorted(shared)}")
    if args.dry_run:
        for a in range(len(folds["assignments"])):
            fold = np.array(fold_vector(folds, a, plot_ids))
            for k in range(5):
                train, val, test = within_splits(fold, uncertain, k)
                print(f"  a{a} fold {k}: train {len(train)}, val {len(val)}, test {len(test)}")
            for source, target in DIRECTIONS:
                train, val, test = cross_splits(fold, uncertain, survey, keep, source, target)
                print(f"  a{a} {source}->{target}: train {len(train)}, val {len(val)}, test {len(test)}")
        print("6 feature sets; dry run passed. Nothing was fitted or written.")
        return 0

    assert_under(OUTPUT_ROOT / "scores.csv", OUTPUT_ROOT)
    if OUTPUT_ROOT.exists():
        print(f"Refusing to overwrite {OUTPUT_ROOT}. Inspect or move it before rerunning.")
        return 1
    duty, dm_peak = image_features(records, args.cache_dir)
    features = feature_sets(records, duty, dm_peak)
    rows, chosen = [], {}
    for name, X in features.items():
        for a in range(len(folds["assignments"])):
            fold = np.array(fold_vector(folds, a, plot_ids))
            for k in range(5):
                train, val, test = within_splits(fold, uncertain, k)
                forest, leaf = fit_forest(X, labels, records, train, val)
                chosen[f"{name}|a{a}|f{k}"] = leaf
                for i, p in zip(test.tolist(), forest.predict_proba(X[test]).tolist()):
                    rows.append([plot_ids[i], name, a, "within", *[f"{x:.6f}" for x in p]])
            for source, target in DIRECTIONS:
                train, val, test = cross_splits(fold, uncertain, survey, keep, source, target)
                forest, leaf = fit_forest(X, labels, records, train, val)
                chosen[f"{name}|a{a}|{source}->{target}"] = leaf
                for i, p in zip(test.tolist(), forest.predict_proba(X[test]).tolist()):
                    rows.append([plot_ids[i], name, a, f"{source}->{target}", *[f"{x:.6f}" for x in p]])
        print(f"{name}: done", flush=True)
    write_exclusive(OUTPUT_ROOT / "scores.csv", render_scores(rows))
    write_exclusive(
        OUTPUT_ROOT / "metadata.json",
        json.dumps(
            {
                "leaf_grid": list(LEAF_GRID),
                "chosen_leaf": chosen,
                "identity_sha256": sha256_file(args.identity),
                "folds_sha256": sha256_file(args.folds),
                "manifest_sha256": sha256_file(args.manifest),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    print(f"Wrote {OUTPUT_ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
