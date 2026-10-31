#!/usr/bin/env python3
"""Build EXP-009 source identities for PSC pulsar candidates from the ATNF catalogue.

Cross-matches every ``known_pulsar`` row in the multi-survey manifest against an
ATNF ``psrcat.db`` on sky position, DM, and period (allowing harmonics), then
applies the recorded manual decisions. The database is read as plain text; no
catalogue code is compiled or run. It is not committed: its version and hash are
checked and recorded instead.

Two groupings are written, because they answer different questions:

  * ``fold_group_id`` is what the frozen folds were generated from. It merges
    conservatively and must not change, or the folds would change with it.
  * ``source_id`` applies the manual decisions and is the unit for metrics and
    resampling. Every source_id stays inside one fold (checked by the fold builder).

Writes refuse to overwrite. ``--check`` rebuilds in memory and compares the output
byte for byte against the existing files.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.psc_folds import load_candidate_records, sha256_file, write_exclusive

DEFAULT_MANIFEST = REPO_ROOT / "data" / "psc" / "psc_eval_multisurvey.csv"
DEFAULT_CACHE = REPO_ROOT / "data" / "psc" / "raw"
DEFAULT_CATALOGUE = REPO_ROOT / "data" / "raw" / "atnf" / "psrcat_v2.9.0.db"
DEFAULT_DECISIONS = REPO_ROOT / "data" / "psc" / "identity" / "manual_decisions.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "psc" / "identity" / "source_identity_v1.csv"
CATALOGUE_SHA256 = "bac32471fef3e60cfd734f1db85503a9bd0a565018cb98edff54475f87eb642f"
CATALOGUE_VERSION = "2.9.0"

ACCEPTED_CLASSES = ("A", "B1", "B2", "B3", "C1")
SIMPLE_HARMONIC = 8
OBLIQUITY_RAD = math.radians(23.4392911)
HARMONIC_ORDER = sorted(
    (
        (a, b)
        for a in range(1, 33)
        for b in range(1, 33)
        if math.gcd(a, b) == 1
    ),
    key=lambda t: (max(t), t),
)
FIELDNAMES = [
    "plot_id",
    "survey",
    "source_id",
    "fold_group_id",
    "catalog_name",
    "label_uncertain",
    "fundamental",
    "match_class",
    "confidence",
    "matched_pulsar",
    "alternatives",
    "sep_deg",
    "delta_dm",
    "harmonic_ratio",
    "note",
]


def _number(text: str) -> float | None:
    try:
        return float(text.replace("D", "E"))
    except ValueError:
        return None


def _ra_degrees(text: str) -> float:
    parts = [float(x) for x in text.split(":")]
    parts += [0.0] * (3 - len(parts))
    return 15 * (parts[0] + parts[1] / 60 + parts[2] / 3600)


def _dec_degrees(text: str) -> float:
    sign = -1 if text.startswith("-") else 1
    parts = [float(x) for x in text.lstrip("+-").split(":")]
    parts += [0.0] * (3 - len(parts))
    return sign * (parts[0] + parts[1] / 60 + parts[2] / 3600)


def ecliptic_to_equatorial(longitude: float, latitude: float) -> tuple[float, float]:
    """J2000 ecliptic to equatorial degrees. Many timing solutions list only these."""
    lon, lat = math.radians(longitude), math.radians(latitude)
    dec = math.asin(
        math.sin(lat) * math.cos(OBLIQUITY_RAD)
        + math.cos(lat) * math.sin(OBLIQUITY_RAD) * math.sin(lon)
    )
    ra = math.atan2(
        math.sin(lon) * math.cos(OBLIQUITY_RAD) - math.tan(lat) * math.sin(OBLIQUITY_RAD),
        math.cos(lon),
    )
    return math.degrees(ra) % 360, math.degrees(dec)


def parse_catalogue(text: str) -> list[dict]:
    """Parse ``psrcat.db`` text into records usable for matching.

    Records with no position in either coordinate system are dropped.
    """
    records: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        if line.startswith("@"):
            if current:
                records.append(current)
            current = {}
            continue
        parts = line.split()
        if len(parts) >= 2:
            current.setdefault(parts[0], parts[1])
    if current:
        records.append(current)

    out = []
    for rec in records:
        period = None
        if "P0" in rec:
            period = _number(rec["P0"])
        elif "F0" in rec:
            frequency = _number(rec["F0"])
            period = 1 / frequency if frequency else None
        ra = dec = None
        try:
            if "RAJ" in rec and "DECJ" in rec:
                ra, dec = _ra_degrees(rec["RAJ"]), _dec_degrees(rec["DECJ"])
            elif "ELONG" in rec and "ELAT" in rec:
                ra, dec = ecliptic_to_equatorial(_number(rec["ELONG"]), _number(rec["ELAT"]))
        except (TypeError, ValueError):
            ra = dec = None
        if ra is None:
            continue
        out.append(
            {
                "jname": rec.get("PSRJ"),
                "binary": rec.get("BINARY"),
                "period_ms": period * 1000 if period else None,
                "dm": _number(rec["DM"]) if "DM" in rec else None,
                "ra": ra,
                "dec": dec,
            }
        )
    return out


def harmonic(ratio: float, tolerance: float, max_order: int) -> tuple[int, int, float] | None:
    """Lowest-order ``a/b`` within ``tolerance`` of ``ratio``, or None."""
    for a, b in HARMONIC_ORDER:
        if max(a, b) > max_order:
            break
        residual = abs(ratio / (a / b) - 1)
        if residual < tolerance:
            return a, b, residual
    return None


def separations(row: dict, ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    cos_dec = np.cos(np.radians((dec + row["dec_deg"]) / 2))
    return np.hypot(((ra - row["ra_deg"] + 180) % 360 - 180) * cos_dec, dec - row["dec_deg"])


def _pair_score(row: dict, entry: dict, sep: float) -> tuple[int, tuple | None] | None:
    if entry["dm"] is None or entry["period_ms"] is None:
        return None
    delta = abs(row["dm"] - entry["dm"])
    dm_strict = delta <= max(1, 0.05 * entry["dm"])
    dm_loose = delta <= max(3, 0.15 * entry["dm"])
    ratio = row["period_ms"] / entry["period_ms"]
    h_tight = harmonic(ratio, 1e-3, SIMPLE_HARMONIC)
    h_wide = harmonic(ratio, 2e-3, 32)
    h_binary = harmonic(ratio, 1e-2, SIMPLE_HARMONIC) if entry["binary"] else None
    if sep <= 1 and dm_strict and h_tight:
        return 1, h_tight
    if sep <= 2 and dm_loose and h_wide:
        return 2, h_wide
    if sep <= 2 and dm_loose and h_binary:
        return 2, h_binary
    if sep <= 1 and dm_strict:
        return 3, None
    if sep <= 3 and dm_strict and h_tight:
        return 3, h_tight
    return None


def _loose_harmonic(row, entry, sep, max_order=32, tolerance=2e-3):
    if entry["dm"] is None or entry["period_ms"] is None:
        return None
    if sep > 2 or abs(row["dm"] - entry["dm"]) > max(3, 0.15 * entry["dm"]):
        return None
    ratio = row["period_ms"] / entry["period_ms"]
    found = harmonic(ratio, tolerance, max_order)
    if not found and entry["binary"]:
        found = harmonic(ratio, 1e-2, 16)
    return found


def _order_key(hit: dict) -> tuple:
    return (max(hit["harmonic"][:2]) if hit["harmonic"] else 99, hit["sep"])


def direct_hits(row: dict, catalogue: list[dict], ra: np.ndarray, dec: np.ndarray) -> list[dict]:
    seps = separations(row, ra, dec)
    hits = []
    for i in np.where(seps <= 3)[0]:
        score = _pair_score(row, catalogue[i], float(seps[i]))
        if score:
            hits.append({"i": int(i), "tier": score[0], "harmonic": score[1], "sep": float(seps[i])})
    hits.sort(key=lambda h: (h["tier"], *_order_key(h)))
    return hits


def classify(row, hits, catalogue, anchors, ra, dec) -> tuple[str, str, dict | None, set[int]]:
    """Assign one row to a match class, strongest evidence first."""
    seps = separations(row, ra, dec)
    pack = lambda i, h: {"i": i, "harmonic": h, "sep": float(seps[i])}  # noqa: E731

    direct = [h for h in hits if h["tier"] == 1]
    if direct:
        return "A direct", "high", direct[0], {h["i"] for h in direct}

    found = []
    for i in anchors:
        h = _loose_harmonic(row, catalogue[i], float(seps[i]))
        if h:
            found.append(pack(i, h))
    found.sort(key=_order_key)
    if found:
        return "B1 anchored harmonic", "high", found[0], {x["i"] for x in found}

    found = [
        pack(i, None)
        for i in anchors
        if seps[i] <= 1
        and catalogue[i]["dm"] is not None
        and abs(row["dm"] - catalogue[i]["dm"]) <= max(1, 0.05 * catalogue[i]["dm"])
    ]
    found.sort(key=lambda x: x["sep"])
    if found:
        return "B2 anchored sky+DM, non-harmonic fold", "medium-high", found[0], {x["i"] for x in found}

    found = []
    for i in anchors:
        if seps[i] <= 1 and catalogue[i]["period_ms"]:
            h = harmonic(row["period_ms"] / catalogue[i]["period_ms"], 5e-4, SIMPLE_HARMONIC)
            if h:
                found.append(pack(i, h))
    found.sort(key=_order_key)
    if found:
        return "B3 anchored sky+period, DM off", "medium", found[0], {x["i"] for x in found}

    found = []
    for i in np.where(seps <= 3)[0]:
        i = int(i)
        h = _loose_harmonic(row, catalogue[i], float(seps[i]), SIMPLE_HARMONIC, 2e-3)
        if h:
            found.append(pack(i, h))
    found.sort(key=_order_key)
    if found:
        return "C1 unanchored simple harmonic", "medium", found[0], {x["i"] for x in found}

    if hits:
        tier = hits[0]["tier"]
        return "C2 weak (high-order or sky+DM only)", "low", hits[0], {h["i"] for h in hits if h["tier"] == tier}
    return "D no catalogue match", "none", None, set()


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)


def _sky_separation(a: dict, b: dict) -> float:
    cos_dec = np.cos(np.radians((a["dec_deg"] + b["dec_deg"]) / 2))
    return float(np.hypot(((a["ra_deg"] - b["ra_deg"] + 180) % 360 - 180) * cos_dec, a["dec_deg"] - b["dec_deg"]))


def internal_links(rows: list[dict]) -> list[int]:
    """Group rows on sky + DM, or tight period + DM within 5 degrees.

    Over-merging only costs sample size; missing a repeat causes leakage.
    """
    uf = _UnionFind(len(rows))
    near_ratios = np.array(sorted({a / b for a in range(1, 3) for b in range(1, 3)}))
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            if abs(a["dm"] - b["dm"]) > max(3, 0.1 * max(a["dm"], b["dm"])):
                continue
            sep = _sky_separation(a, b)
            close_period = (np.abs((a["period_ms"] / b["period_ms"]) / near_ratios - 1) < 5e-4).any()
            if sep <= 1 or (sep <= 5 and close_period):
                uf.union(i, j)
    return [uf.find(i) for i in range(len(rows))]


def _relabel(values: list, prefix: str) -> list[str]:
    """Name groups by order of first appearance so ids are stable across runs."""
    names: dict = {}
    out = []
    for value in values:
        if value not in names:
            names[value] = f"{prefix}{len(names):03d}"
        out.append(names[value])
    return out


def load_decisions(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def match_rows(rows: list[dict], catalogue: list[dict]) -> list[dict]:
    """Classify every row. Pulsars with a direct match become anchors for the rest."""
    ra = np.array([c["ra"] for c in catalogue])
    dec = np.array([c["dec"] for c in catalogue])
    all_hits = [direct_hits(row, catalogue, ra, dec) for row in rows]
    anchors = sorted({h["i"] for hits in all_hits for h in hits if h["tier"] == 1})

    matched = []
    for row, hits in zip(rows, all_hits):
        cls, confidence, best, alts = classify(row, hits, catalogue, anchors, ra, dec)
        rec = {"cls": cls, "confidence": confidence, "jname": None, "alts": [], "sep": None, "ddm": None, "ratio": None}
        if best:
            entry = catalogue[best["i"]]
            rec.update(
                jname=entry["jname"],
                sep=best["sep"],
                ddm=row["dm"] - entry["dm"] if entry["dm"] is not None else None,
                ratio=list(best["harmonic"][:2]) if best.get("harmonic") else None,
                alts=sorted({catalogue[i]["jname"] for i in alts} - {entry["jname"]}),
            )
        matched.append(rec)
    return matched


def assign_groups(rows: list[dict], matched: list[dict], decisions: list[dict[str, str]]) -> list[dict]:
    """Combine evidence and decisions into the final per-row identity table."""
    links = internal_links(rows)
    uf = _UnionFind(len(rows))
    first: dict = {}

    def link(key, i):
        if key in first:
            uf.union(i, first[key])
        else:
            first[key] = i

    for i, m in enumerate(matched):
        link(("internal", links[i]), i)
        if m["jname"] and m["cls"].split()[0] in ACCEPTED_CLASSES:
            link(("name", m["jname"]), i)
            if m["cls"][0] in "AB":
                for alt in m["alts"]:
                    link(("name", alt), i)
    fold_groups = [uf.find(i) for i in range(len(rows))]

    by_action = collections.defaultdict(dict)
    for d in decisions:
        by_action[d["action"]][d["key"]] = d
    name_group: dict[str, int] = {}
    for g, m in zip(fold_groups, matched):
        if m["jname"]:
            name_group.setdefault(m["jname"], g)

    table = []
    groups: list = []
    for i, (row, m) in enumerate(zip(rows, matched)):
        pid = row["plot_id"]
        group = fold_groups[i]
        accepted = m["cls"].split()[0] in ACCEPTED_CLASSES
        name = m["jname"] if accepted else None
        note = ""
        if pid in by_action["split_source"]:
            d = by_action["split_source"][pid]
            group, name, note = f"split:{pid}", d["target"], d["note"]
        if pid in by_action["merge_into"]:
            d = by_action["merge_into"][pid]
            group, name, note = name_group[d["target"]], d["target"], d["note"]
        if m["jname"] in by_action["accept_name"]:
            name = m["jname"]
            note = note or by_action["accept_name"][m["jname"]]["note"]
        if m["jname"] in by_action["withhold_name"]:
            name = None
            note = by_action["withhold_name"][m["jname"]]["note"]
        uncertain = pid in by_action["label_uncertain"]
        if uncertain:
            note = by_action["label_uncertain"][pid]["note"]
        fundamental = m["ratio"] == [1, 1] and m["cls"].split()[0] in ("A", "B1", "C1") and not uncertain
        groups.append(group)
        table.append(
            {
                "plot_id": pid,
                "survey": row["survey"],
                "catalog_name": name or "",
                "label_uncertain": int(uncertain),
                "fundamental": int(fundamental),
                "match_class": m["cls"],
                "confidence": m["confidence"],
                "matched_pulsar": m["jname"] or "",
                "alternatives": ";".join(m["alts"]),
                "sep_deg": "" if m["sep"] is None else f"{m['sep']:.4f}",
                "delta_dm": "" if m["ddm"] is None else f"{m['ddm']:.4f}",
                "harmonic_ratio": "" if not m["ratio"] else f"{m['ratio'][0]}/{m['ratio'][1]}",
                "note": note,
            }
        )
    for rec, source, fold_group in zip(table, _relabel(groups, "S"), _relabel(fold_groups, "G")):
        rec["source_id"] = source
        rec["fold_group_id"] = fold_group
    return table


def render_csv(table: list[dict]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(table)
    return buffer.getvalue()


def summarize(table: list[dict]) -> dict:
    counts = collections.Counter(r["match_class"].split()[0] for r in table)
    by_survey = {
        survey: {
            "rows": sum(r["survey"] == survey for r in table),
            "source_groups": len({r["source_id"] for r in table if r["survey"] == survey}),
            "fundamental_sources": len({r["source_id"] for r in table if r["survey"] == survey and r["fundamental"]}),
        }
        for survey in sorted({r["survey"] for r in table})
    }
    spans = collections.defaultdict(set)
    for r in table:
        spans[r["source_id"]].add(r["survey"])
    return {
        "rows": len(table),
        "source_groups": len(spans),
        "shared_sources": sum(len(v) > 1 for v in spans.values()),
        "named_pulsars": len({r["catalog_name"] for r in table if r["catalog_name"]}),
        "label_uncertain_rows": sum(r["label_uncertain"] for r in table),
        "fundamental_rows": sum(r["fundamental"] for r in table),
        "by_survey": by_survey,
        "match_classes": dict(sorted(counts.items())),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--catalogue-db", type=Path, default=DEFAULT_CATALOGUE)
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="Compare against the existing output; write nothing")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.catalogue_db.is_file():
        raise FileNotFoundError(
            f"{args.catalogue_db} not found. Get the ATNF Pulsar Catalogue package from "
            "https://www.atnf.csiro.au/research/pulsar/psrcat/download.html and extract psrcat.db."
        )
    catalogue_hash = sha256_file(args.catalogue_db)
    if catalogue_hash != CATALOGUE_SHA256:
        raise ValueError(f"catalogue hash {catalogue_hash} is not the recorded {CATALOGUE_SHA256}")

    catalogue = parse_catalogue(args.catalogue_db.read_text(errors="replace"))
    records = load_candidate_records(args.manifest, args.cache_dir)
    rows = [r for r in records if r["category"] == "known_pulsar"]
    table = assign_groups(rows, match_rows(rows, catalogue), load_decisions(args.decisions))

    text = render_csv(table)
    summary = summarize(table)
    meta = {
        "catalogue_version": CATALOGUE_VERSION,
        "catalogue_sha256": catalogue_hash,
        "catalogue_citation": "Manchester et al. 2005, AJ 129, 1993; ATNF Pulsar Catalogue",
        "manifest": args.manifest.relative_to(REPO_ROOT).as_posix(),
        "manifest_sha256": sha256_file(args.manifest),
        "decisions_sha256": sha256_file(args.decisions),
        "summary": summary,
    }
    meta_text = json.dumps(meta, indent=2, sort_keys=True) + "\n"
    meta_path = args.output.with_suffix(".meta.json")
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.check:
        ok = args.output.read_text() == text and meta_path.read_text() == meta_text
        print("OK: identity output reproduces byte for byte" if ok else "FAILED: identity output differs")
        return 0 if ok else 1

    write_exclusive(args.output, text)
    write_exclusive(meta_path, meta_text)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
