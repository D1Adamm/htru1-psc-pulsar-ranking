#!/usr/bin/env python3
"""Build a PSC AO327 evaluation manifest from pasted psrsearch tab exports.

Reads per-tab exports under a source directory and writes a single labeled
manifest CSV. Adds a ``source_group`` column so downstream ranking metrics can
be computed group-aware: many rows in these tabs are repeat detections or
harmonics of the same underlying source, and treating them as independent
samples overstates performance.

Accepts raw copy-pasted rows. The PSC tables carry a variable number of
columns (the four Subplot columns and Subplot Score are often blank), so the
parser locates the grade by vocabulary rather than by fixed position.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = REPO_ROOT / "data" / "psc" / "source"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "psc" / "psc_eval_candidates.csv"

# Every grade string the PSC tables are known to use. Used to find the grade
# column when the row's width varies.
KNOWN_GRADES = {"Known", "RFI", "Noise", "New", "Maybe", "Best"}

# tab file stem -> (source_category, label, allowed grades, label_source)
TABS: dict[str, tuple[str, str, set[str], str]] = {
    "known_pulsars": (
        "known_pulsar",
        "1",
        {"Known"},
        "PSC psrsearch AO327 tab=Known Pulsars, consensus grade=Known",
    ),
    "new_pulsars": (
        "ao327_discovery",
        "1",
        {"New", "Known"},
        "PSC psrsearch AO327 tab=New Pulsars (survey discovery)",
    ),
    "rfi": (
        "rfi",
        "0",
        {"RFI"},
        "PSC psrsearch AO327 tab=RFI, consensus grade=RFI",
    ),
    "noise": (
        "noise",
        "0",
        {"Noise"},
        "PSC psrsearch AO327 tab=Noise, consensus grade=Noise",
    ),
}

FIELDNAMES = [
    "plot_id",
    "survey",
    "source_category",
    "label",
    "label_confidence",
    "label_source",
    "source_group",
    "ra",
    "dec",
    "mjd",
    "period_ms",
    "dm",
    "chi2",
    "psc_count",
    "notes",
]


def _is_header(fields: list[str]) -> bool:
    first = fields[0].strip().lower()
    return first in {"id", "pulsar name", "plot_id"} or first.startswith("id\t")


def parse_row(fields: list[str], *, path: Path, lineno: int) -> dict[str, str] | None:
    """Parse one exported table row of variable width.

    Layout is ``ID RA DEC MJD Period DM X2 [Subplot1..4 SubplotScore] Grade Count``
    where the bracketed columns are frequently blank. The grade is located by
    matching against ``KNOWN_GRADES`` rather than by index.
    """
    if len(fields) < 8:
        raise ValueError(
            f"{path.name}:{lineno}: expected at least 8 tab-separated columns, "
            f"got {len(fields)}"
        )

    grade_indices = [
        i for i, value in enumerate(fields) if value.strip() in KNOWN_GRADES
    ]
    if not grade_indices:
        raise ValueError(
            f"{path.name}:{lineno}: no recognizable grade column "
            f"(looked for one of {sorted(KNOWN_GRADES)}). Row: {fields[:3]}..."
        )
    grade_idx = grade_indices[-1]

    remainder = [f.strip() for f in fields[grade_idx + 1 :] if f.strip()]
    return {
        "plot_id": fields[0].strip(),
        "ra": fields[1].strip(),
        "dec": fields[2].strip(),
        "mjd": fields[3].strip(),
        "period_ms": fields[4].strip(),
        "dm": fields[5].strip(),
        "chi2": fields[6].strip(),
        "grade": fields[grade_idx].strip(),
        "psc_count": remainder[0] if remainder else "",
    }


def read_tab(path: Path) -> list[dict[str, str]]:
    """Parse one exported tab, skipping headers, blanks, and exact duplicates."""
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    duplicates = 0

    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split("\t")
        if _is_header(fields):
            continue

        row = parse_row(fields, path=path, lineno=lineno)
        if row is None:
            continue
        if not row["plot_id"].isdigit():
            raise ValueError(
                f"{path.name}:{lineno}: plot_id {row['plot_id']!r} is not numeric"
            )
        # Overlapping page grabs can repeat a candidate; that is benign.
        if row["plot_id"] in seen:
            duplicates += 1
            continue
        seen.add(row["plot_id"])
        rows.append(row)

    if duplicates:
        print(f"  {path.name}: skipped {duplicates} duplicate row(s)")
    return rows


def assign_source_groups(rows: list[dict[str, str]], category: str) -> None:
    """Tag rows that plausibly share one underlying source.

    Conservative and positional: same RA to ~10 minutes and same Dec degree,
    with periods agreeing to 0.2 percent. A heuristic for de-weighting repeat
    detections, not a source identification.
    """
    groups: list[list[dict[str, str]]] = []
    for row in rows:
        pos = (row["ra"][:4], row["dec"][:3])
        try:
            period = float(row["period_ms"])
        except ValueError:
            period = float("nan")
        for group in groups:
            head = group[0]
            same_pos = (head["ra"][:4], head["dec"][:3]) == pos
            try:
                head_p = float(head["period_ms"])
            except ValueError:
                continue
            if same_pos and head_p and abs(head_p - period) / head_p < 0.002:
                group.append(row)
                break
        else:
            groups.append([row])

    for i, group in enumerate(groups):
        for row in group:
            row["source_group"] = f"{category}_g{i:04d}"


def build_rows(source_dir: Path) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    found_any = False

    for tab, (category, label, allowed, label_source) in TABS.items():
        path = source_dir / f"{tab}.tsv"
        if not path.is_file():
            print(f"  {tab}.tsv: not present, skipping")
            continue
        found_any = True

        rows = read_tab(path)
        bad = [r for r in rows if r["grade"] not in allowed]
        if bad:
            observed = sorted({r["grade"] for r in rows})
            raise ValueError(
                f"{path.name}: {len(bad)} row(s) carry a grade outside "
                f"{sorted(allowed)} (first: plot_id={bad[0]['plot_id']} "
                f"grade={bad[0]['grade']!r}). Grades observed in this file: "
                f"{observed}. Refusing to mislabel."
            )

        assign_source_groups(rows, category)
        for row in rows:
            out.append(
                {
                    "plot_id": row["plot_id"],
                    "survey": "AO327",
                    "source_category": category,
                    "label": label,
                    "label_confidence": "psc_consensus_grade",
                    "label_source": label_source,
                    "source_group": row["source_group"],
                    "ra": row["ra"],
                    "dec": row["dec"],
                    "mjd": row["mjd"],
                    "period_ms": row["period_ms"],
                    "dm": row["dm"],
                    "chi2": row["chi2"],
                    "psc_count": row["psc_count"],
                    "notes": "",
                }
            )
        print(f"  {tab}.tsv: {len(rows)} rows")

    if not found_any:
        raise FileNotFoundError(f"No tab exports found in {source_dir}")

    # A candidate appearing under two different grades is a real label conflict.
    seen: dict[str, str] = {}
    for row in out:
        prior = seen.get(row["plot_id"])
        if prior is not None and prior != row["source_category"]:
            raise ValueError(
                f"plot_id {row['plot_id']} appears as both {prior!r} and "
                f"{row['source_category']!r}. Resolve before building."
            )
        seen[row["plot_id"]] = row["source_category"]
    return out


def summarize(rows: list[dict[str, str]]) -> None:
    """Report the diversity that actually matters, per class."""
    from collections import defaultdict

    by_cat: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_cat[row["source_category"]].append(row)

    print("\nDiversity by class:")
    print(f"  {'category':18s} {'rows':>6} {'sources':>8} {'largest':>8} {'MJDs':>6}")
    for cat, group in sorted(by_cat.items()):
        counts: dict[str, int] = defaultdict(int)
        for r in group:
            counts[r["source_group"]] += 1
        largest = max(counts.values()) if counts else 0
        mjds = len({r["mjd"] for r in group})
        print(f"  {cat:18s} {len(group):6d} {len(counts):8d} {largest:8d} {mjds:6d}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and report diversity without writing the manifest",
    )
    args = parser.parse_args(argv)

    print(f"Source: {args.source_dir}")
    rows = build_rows(args.source_dir)
    summarize(rows)

    n_pos = sum(1 for r in rows if r["label"] == "1")
    groups = {r["source_group"] for r in rows}
    print(f"\nTotal: {len(rows)} candidates, {len(groups)} distinct source groups")
    print(f"  positives: {n_pos}   negatives: {len(rows) - n_pos}")

    if args.dry_run:
        print("\n(dry run: manifest not written)")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
