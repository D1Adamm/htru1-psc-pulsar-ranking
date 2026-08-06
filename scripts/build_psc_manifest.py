#!/usr/bin/env python3
"""Build the PSC AO327 evaluation manifest from pasted psrsearch tab exports.

Reads the raw per-tab TSV exports under ``data/psc/source/`` and writes a single
labeled manifest CSV. Adds a ``source_group`` column so downstream ranking
metrics can be computed group-aware: several rows in these tabs are repeat
detections or harmonics of the same underlying source, and treating them as
independent samples would overstate performance.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "data" / "psc" / "source"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "psc" / "psc_eval_candidates.csv"

# tab file -> (source_category, label, label_source)
TABS = {
    "known_pulsars": (
        "known_pulsar",
        "1",
        "PSC psrsearch AO327 tab=Known Pulsars, consensus grade=Known",
    ),
    "rfi": (
        "rfi",
        "0",
        "PSC psrsearch AO327 tab=RFI, consensus grade=RFI",
    ),
    "noise": (
        "noise",
        "0",
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


def read_tab(path: Path) -> list[dict[str, str]]:
    """Parse one exported tab. Columns are ID RA DEC MJD Period DM X2 Grade Count."""
    rows: list[dict[str, str]] = []
    for lineno, line in enumerate(path.read_text().strip().splitlines(), start=1):
        if not line.strip():
            continue
        fields = [f.strip() for f in line.split("\t")]
        if len(fields) != 9:
            raise ValueError(
                f"{path.name}:{lineno}: expected 9 tab-separated fields, got {len(fields)}"
            )
        rows.append(
            {
                "plot_id": fields[0],
                "ra": fields[1],
                "dec": fields[2],
                "mjd": fields[3],
                "period_ms": fields[4],
                "dm": fields[5],
                "chi2": fields[6],
                "grade": fields[7],
                "psc_count": fields[8],
            }
        )
    return rows


def assign_source_groups(rows: list[dict[str, str]], category: str) -> None:
    """Tag rows that plausibly share one underlying source.

    Grouping is deliberately conservative and positional: same RA to ~10 minutes
    and same Dec degree, with periods agreeing to 0.2 percent. It is a heuristic
    for de-weighting repeat detections, not a source identification.
    """
    if category == "noise":
        # Every noise row here is from one pointing and one MJD.
        for row in rows:
            row["source_group"] = f"noise_{rows[0]['mjd']}_{rows[0]['ra']}"
        return

    groups: list[list[dict[str, str]]] = []
    for row in rows:
        pos = (row["ra"][:4], row["dec"][:3])
        period = float(row["period_ms"])
        for group in groups:
            head = group[0]
            same_pos = (head["ra"][:4], head["dec"][:3]) == pos
            head_p = float(head["period_ms"])
            if same_pos and abs(head_p - period) / head_p < 0.002:
                group.append(row)
                break
        else:
            groups.append([row])

    for i, group in enumerate(groups):
        for row in group:
            row["source_group"] = f"{category}_g{i:03d}"


def build_rows() -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for tab, (category, label, label_source) in TABS.items():
        path = SOURCE_DIR / f"{tab}.tsv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing source export: {path}")

        rows = read_tab(path)
        expected_grade = {"known_pulsar": "Known", "rfi": "RFI", "noise": "Noise"}[category]
        bad = [r for r in rows if r["grade"] != expected_grade]
        if bad:
            raise ValueError(
                f"{path.name}: {len(bad)} row(s) have a grade other than "
                f"{expected_grade!r} (first: plot_id={bad[0]['plot_id']} "
                f"grade={bad[0]['grade']!r}). Refusing to mislabel."
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

    seen: dict[str, str] = {}
    for row in out:
        if row["plot_id"] in seen:
            raise ValueError(
                f"duplicate plot_id {row['plot_id']} in "
                f"{seen[row['plot_id']]} and {row['source_category']}"
            )
        seen[row["plot_id"]] = row["source_category"]
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    rows = build_rows()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    n_pos = sum(1 for r in rows if r["label"] == "1")
    groups = {r["source_group"] for r in rows}
    pos_groups = {r["source_group"] for r in rows if r["label"] == "1"}
    print(f"Wrote {len(rows)} candidates -> {args.output}")
    print(f"  positives: {n_pos}  negatives: {len(rows) - n_pos}")
    print(f"  distinct source groups: {len(groups)} ({len(pos_groups)} among positives)")


if __name__ == "__main__":
    main()
