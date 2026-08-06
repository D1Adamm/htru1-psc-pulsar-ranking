#!/usr/bin/env python3
"""Score and rank a manifest of PSC candidates with a trained HTRU1 BaselineCNN.

Produces experimental cross-survey pulsar-likelihood scores for reviewer
triage. The sigmoid output is NOT a calibrated probability: it comes from a
model trained on a different survey, instrument, and frequency band.

Fetched JSON is cached under ``data/psc/raw/`` so each candidate is retrieved
at most once. Requests are spaced by ``--delay`` seconds to stay polite.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.psc_preprocessing import psc_json_to_tensor

DEFAULT_MANIFEST = REPO_ROOT / "data" / "psc" / "psc_eval_candidates.csv"
DEFAULT_CHECKPOINT = REPO_ROOT / "outputs" / "models" / "baseline_cnn_best.pt"
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "psc" / "raw"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "predictions" / "psc_ranked_candidates.csv"
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_THRESHOLD = 0.5
DEFAULT_DELAY_S = 0.5

OUTPUT_FIELDNAMES = [
    "rank",
    "plot_id",
    "score",
    "logit",
    "threshold_label",
    "label",
    "source_category",
    "source_group",
    "label_source",
    "period_ms",
    "dm",
    "chi2",
    "fetch_status",
    "preprocessing_status",
    "error",
]


def _load_script_module(name: str):
    path = SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_predict = _load_script_module("predict_psc_candidate")
load_baseline_cnn = _predict.load_baseline_cnn
score_image = _predict.score_image
select_device = _predict.select_device
fetch_plot_json = _predict.fetch_plot_json


def read_manifest(path: Path) -> list[dict[str, str]]:
    """Read the evaluation manifest CSV. ``plot_id`` is the only required column."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing manifest: {path}")

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"{path}: manifest is empty")
    if "plot_id" not in rows[0]:
        raise ValueError(f"{path}: manifest must have a 'plot_id' column")

    seen: set[str] = set()
    for i, row in enumerate(rows, start=2):
        plot_id = (row.get("plot_id") or "").strip()
        if not plot_id:
            raise ValueError(f"{path}:{i}: blank plot_id")
        if plot_id in seen:
            raise ValueError(f"{path}:{i}: duplicate plot_id {plot_id}")
        seen.add(plot_id)
    return rows


def cache_path_for(cache_dir: Path, plot_id: str) -> Path:
    return cache_dir / f"{plot_id}.json"


def load_cached_payload(cache_dir: Path, plot_id: str) -> dict[str, Any] | None:
    """Return a cached payload, or None when absent or unreadable."""
    path = cache_path_for(cache_dir, plot_id)
    if not path.is_file():
        return None
    try:
        with open(path) as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def save_cached_payload(cache_dir: Path, plot_id: str, payload: dict[str, Any]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_path_for(cache_dir, plot_id), "w") as f:
        json.dump(payload, f)


def get_payload(
    plot_id: str,
    *,
    cache_dir: Path,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    use_cache: bool = True,
    fetch: Callable[..., dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], str]:
    """Return ``(payload, fetch_status)``, preferring the local cache."""
    if use_cache:
        cached = load_cached_payload(cache_dir, plot_id)
        if cached is not None:
            return cached, "cached"

    fetcher = fetch or fetch_plot_json
    payload = fetcher(plot_id, timeout_s=timeout_s)
    if use_cache:
        save_cached_payload(cache_dir, plot_id, payload)
    return payload, "fetched"


def score_manifest_rows(
    rows: list[dict[str, str]],
    model,
    device,
    *,
    cache_dir: Path,
    threshold: float = DEFAULT_THRESHOLD,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    use_cache: bool = True,
    delay_s: float = 0.0,
    fetch: Callable[..., dict[str, Any]] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Score every manifest row, recording failures instead of aborting."""
    results: list[dict[str, Any]] = []

    for row in rows:
        plot_id = (row.get("plot_id") or "").strip()
        result: dict[str, Any] = {
            "plot_id": plot_id,
            "score": None,
            "logit": None,
            "threshold_label": None,
            "label": row.get("label", ""),
            "source_category": row.get("source_category", ""),
            "source_group": row.get("source_group", ""),
            "label_source": row.get("label_source", ""),
            "period_ms": row.get("period_ms", ""),
            "dm": row.get("dm", ""),
            "chi2": row.get("chi2", ""),
            "fetch_status": "",
            "preprocessing_status": "",
            "error": "",
        }

        try:
            payload, fetch_status = get_payload(
                plot_id,
                cache_dir=cache_dir,
                timeout_s=timeout_s,
                use_cache=use_cache,
                fetch=fetch,
            )
            result["fetch_status"] = fetch_status
        except Exception as exc:  # network/HTTP/JSON failures are recoverable
            result["fetch_status"] = "failed"
            result["preprocessing_status"] = "skipped"
            result["error"] = f"{type(exc).__name__}: {exc}"
            results.append(result)
            if progress:
                progress(f"{plot_id}: fetch failed ({exc})")
            continue

        if result["fetch_status"] == "fetched" and delay_s > 0:
            sleep(delay_s)

        try:
            image, _metadata = psc_json_to_tensor(payload, plot_id=plot_id)
            scores = score_image(model, image, device=device, threshold=threshold)
        except Exception as exc:
            result["preprocessing_status"] = "failed"
            result["error"] = f"{type(exc).__name__}: {exc}"
            results.append(result)
            if progress:
                progress(f"{plot_id}: preprocessing/scoring failed ({exc})")
            continue

        result["preprocessing_status"] = "ok"
        result["logit"] = scores["raw_logit"]
        result["score"] = scores["pulsar_probability"]
        result["threshold_label"] = scores["predicted_label"]
        results.append(result)
        if progress:
            progress(f"{plot_id}: score={scores['pulsar_probability']:.6f}")

    return results


def rank_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort by descending score; failed rows sink to the bottom with blank rank."""
    scored = [r for r in results if r.get("score") is not None]
    failed = [r for r in results if r.get("score") is None]
    scored.sort(key=lambda r: -float(r["score"]))
    for i, row in enumerate(scored, start=1):
        row["rank"] = i
    for row in failed:
        row["rank"] = ""
    return scored + failed


def save_ranked_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY_S,
        help="Seconds to wait after each network fetch (cached reads are free)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Always refetch; do not read or write the local JSON cache",
    )
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N rows")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    rows = read_manifest(args.manifest)
    if args.limit is not None:
        rows = rows[: args.limit]

    device = select_device()
    print(f"Device: {device}")
    print(f"Manifest: {args.manifest} ({len(rows)} candidates)")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Cache: {'disabled' if args.no_cache else args.cache_dir}")

    model = load_baseline_cnn(args.checkpoint, device=device)

    results = score_manifest_rows(
        rows,
        model,
        device,
        cache_dir=args.cache_dir,
        threshold=args.threshold,
        timeout_s=args.timeout,
        use_cache=not args.no_cache,
        delay_s=args.delay,
        progress=lambda msg: print(f"  {msg}"),
    )

    ranked = rank_results(results)
    save_ranked_csv(args.output, ranked)

    n_ok = sum(1 for r in ranked if r.get("score") is not None)
    n_failed = len(ranked) - n_ok
    print(f"\nScored {n_ok}/{len(ranked)} candidates ({n_failed} failed)")
    print(f"Wrote ranked results: {args.output}")
    print(
        "\nScores are experimental cross-survey pulsar-likelihood values, "
        "not calibrated probabilities."
    )


if __name__ == "__main__":
    main()
