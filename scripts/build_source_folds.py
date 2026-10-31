#!/usr/bin/env python3
"""Build the frozen EXP-009 fold assignments and fail on any leakage.

Seeds 5, 44, and 52 were chosen for row balance, without reference to any model
result; ``--verify-selection`` re-runs that search over seeds 0-199. Writes refuse
to overwrite. ``--check`` compares against the existing file byte for byte.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.psc_folds import (
    attach_identity,
    assign_folds,
    balance_deviation,
    leak_report,
    linked_components,
    load_candidate_records,
    read_identity,
    sha256_file,
    write_exclusive,
)

DEFAULT_MANIFEST = REPO_ROOT / "data" / "psc" / "psc_eval_multisurvey.csv"
DEFAULT_CACHE = REPO_ROOT / "data" / "psc" / "raw"
DEFAULT_IDENTITY = REPO_ROOT / "data" / "psc" / "identity" / "source_identity_v1.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "psc" / "folds" / "folds_v1.json"
SEEDS = (5, 44, 52)


def select_seeds(records: list[dict], components: list[int], candidates: range, count: int = 3) -> list[int]:
    """Most balanced seeds whose fold labels agree on fewer than 90% of rows."""
    ranked = sorted((balance_deviation(records, assign_folds(records, components, s)), s) for s in candidates)
    chosen: list[int] = []
    vectors: list[list[int]] = []
    for _, seed in ranked:
        vector = assign_folds(records, components, seed)
        agreement_ok = all(
            sum(a == b for a, b in zip(vector, other)) / len(vector) < 0.9 for other in vectors
        )
        if agreement_ok and vector not in vectors:
            chosen.append(seed)
            vectors.append(vector)
        if len(chosen) == count:
            break
    return chosen


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--identity", type=Path, default=DEFAULT_IDENTITY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="Compare against the existing output; write nothing")
    parser.add_argument("--verify-selection", action="store_true", help="Re-run the seed search over 0-199")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    records = load_candidate_records(args.manifest, args.cache_dir)
    attach_identity(records, read_identity(args.identity))
    components = linked_components(records)
    print(f"{len(records)} candidates in {len(set(components))} linked components")

    assignments = []
    for seed in SEEDS:
        vector = assign_folds(records, components, seed)
        leaks = leak_report(records, vector)
        print(f"seed {seed}: balance {balance_deviation(records, vector):.3f}, leaks {leaks}")
        if any(leaks.values()):
            print(f"FAILED: seed {seed} leaks across folds")
            return 1
        assignments.append({"seed": seed, "folds": vector})

    if args.verify_selection:
        selected = select_seeds(records, components, range(200))
        print(f"seed search selects {selected}")
        if tuple(selected) != SEEDS:
            print(f"FAILED: expected {SEEDS}")
            return 1

    payload = {
        "version": "folds_v1",
        "n_folds": 5,
        "manifest_sha256": sha256_file(args.manifest),
        "identity_sha256": sha256_file(args.identity),
        "plot_ids": [r["plot_id"] for r in records],
        "assignments": assignments,
    }
    text = json.dumps(payload, sort_keys=True) + "\n"
    if args.check:
        ok = args.output.read_text() == text
        print("OK: folds reproduce byte for byte" if ok else "FAILED: folds differ")
        return 0 if ok else 1
    write_exclusive(args.output, text)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
