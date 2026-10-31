"""EXP-009 source-aware folds, candidate records, and safe-write helpers.

Folds keep together rows from one source, rows sharing an observation, and RFI
from one night. Noise comes from a single night per survey, so it can only be
split by observation.
"""

from __future__ import annotations

import collections
import csv
import hashlib
import json
import random
import re
from pathlib import Path

N_FOLDS = 5
CLASS_KEYS = ("known_pulsar", "rfi", "noise")
BACKENDS = ("p1693", "puppi", "guppi", "vegas")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_exclusive(path: Path, text: str) -> None:
    """Create ``path``, failing if anything already exists there."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x") as f:
        f.write(text)


def assert_under(path: Path, root: Path) -> None:
    """Refuse to write outside ``root``."""
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"refusing to write outside {root}: {path}")


def _ra_degrees(text: str) -> float:
    h, m, s = (float(x) for x in text.split(":"))
    return 15 * (h + m / 60 + s / 3600)


def _dec_degrees(text: str) -> float:
    sign = -1 if text.startswith("-") else 1
    d, m, s = (float(x) for x in text.lstrip("+-").split(":"))
    return sign * (d + m / 60 + s / 3600)


def observation_key(input_file: str) -> str:
    """The pointing and scan: everything before a candidate's trial DM."""
    return re.split(r"_DM", input_file)[0]


def backend_of(input_file: str) -> str:
    if "puppi" in input_file:
        return "puppi"
    if input_file.startswith("p1693"):
        return "p1693"
    if input_file.startswith("guppi"):
        return "guppi"
    if input_file.startswith("vegas"):
        return "vegas"
    return "other"


def load_candidate_records(manifest: Path, cache_dir: Path) -> list[dict]:
    """One metadata record per manifest row, read from the JSON cache."""
    with open(manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    records = []
    for row in rows:
        with open(cache_dir / f"{row['plot_id']}.json") as f:
            payload = json.load(f)
        meta = payload["meta"]
        mjd = float(meta["MJD"])
        records.append(
            {
                "plot_id": row["plot_id"],
                "survey": row["survey"],
                "category": row["source_category"],
                "period_ms": float(meta["period"]) * 1000,
                "dm": float(meta["bestdm"]),
                "sigma": float(meta["sigma"]),
                "chi2": float(row["chi2"]),
                "night": int(mjd),
                "obs": observation_key(meta["input_file"]),
                "backend": backend_of(meta["input_file"]),
                "phase_bins": len(payload["FvP"][0]),
                "subbands": len(payload["FvP"]),
                "subints": len(payload["TvP"]),
                "t_sample": float(meta["T_sample"]),
                "t_obs": float(meta["observation_length"]),
                "ra_deg": _ra_degrees(row["ra"]),
                "dec_deg": _dec_degrees(row["dec"]),
            }
        )
    return records


def read_identity(path: Path) -> dict[str, dict[str, str]]:
    with open(path, newline="") as f:
        return {row["plot_id"]: row for row in csv.DictReader(f)}


def attach_identity(records: list[dict], identity: dict[str, dict[str, str]]) -> None:
    """Add identity fields to records in place. Negatives get empty values."""
    for rec in records:
        ident = identity.get(rec["plot_id"])
        rec["source_id"] = ident["source_id"] if ident else None
        rec["fold_group_id"] = ident["fold_group_id"] if ident else None
        rec["label_uncertain"] = bool(int(ident["label_uncertain"])) if ident else False
        rec["fundamental"] = bool(int(ident["fundamental"])) if ident else False


def linked_components(records: list[dict]) -> list[int]:
    """Component id per record: rows that must share a fold."""
    parent = list(range(len(records)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    first: dict = {}

    def link(key, i: int) -> None:
        if key in first:
            parent[find(i)] = find(first[key])
        else:
            first[key] = i

    for i, rec in enumerate(records):
        link(("obs", rec["obs"]), i)
        if rec["category"] == "known_pulsar":
            link(("group", rec["fold_group_id"]), i)
        elif rec["category"] == "rfi":
            link(("rfi-night", rec["survey"], rec["night"]), i)
    return [find(i) for i in range(len(records))]


def assign_folds(records: list[dict], components: list[int], seed: int, n_folds: int = N_FOLDS) -> list[int]:
    """Greedily balance rows and pulsar groups per survey and class across folds."""
    rng = random.Random(seed)
    grouped: dict[int, list[int]] = collections.defaultdict(list)
    for i, component in enumerate(components):
        grouped[component].append(i)
    items = list(grouped.values())
    rng.shuffle(items)
    items.sort(key=lambda v: -len(v))

    totals = collections.Counter((r["survey"], r["category"]) for r in records)
    group_totals = collections.Counter(
        survey
        for survey, _ in {
            (r["survey"], r["fold_group_id"]) for r in records if r["category"] == "known_pulsar"
        }
    )
    row_counts = [collections.Counter() for _ in range(n_folds)]
    group_counts = [collections.Counter() for _ in range(n_folds)]
    assignment = [0] * len(records)
    for item in items:
        need = collections.Counter((records[i]["survey"], records[i]["category"]) for i in item)
        groups = collections.Counter(
            survey
            for survey, _ in sorted(
                {
                    (records[i]["survey"], records[i]["fold_group_id"])
                    for i in item
                    if records[i]["category"] == "known_pulsar"
                }
            )
        )
        best, best_score = None, None
        for k in range(n_folds):
            score = sum(v * (1 / n_folds - row_counts[k][q] / totals[q]) for q, v in need.items())
            score += sum(v * 2 * (1 / n_folds - group_counts[k][q] / group_totals[q]) for q, v in groups.items())
            score += rng.random() * 1e-3
            if best_score is None or score > best_score:
                best, best_score = k, score
        for i in item:
            assignment[i] = best
        row_counts[best].update(need)
        group_counts[best].update(groups)
    return assignment


def balance_deviation(records: list[dict], assignment: list[int], n_folds: int = N_FOLDS) -> float:
    """Total departure of each survey-class pair from equal fold shares."""
    totals = collections.Counter((r["survey"], r["category"]) for r in records)
    deviation = 0.0
    for key, total in totals.items():
        counts = collections.Counter(a for a, r in zip(assignment, records) if (r["survey"], r["category"]) == key)
        deviation += sum(abs(counts[k] / total - 1 / n_folds) for k in range(n_folds))
    return deviation


def leak_report(records: list[dict], assignment: list[int]) -> dict[str, int]:
    """Number of units appearing in more than one fold, per rule."""
    report = {}
    rules = {
        "source_id": lambda r: r["source_id"],
        "fold_group_id": lambda r: r["fold_group_id"],
        "observation": lambda r: r["obs"],
        "rfi_night": lambda r: (r["survey"], r["night"]) if r["category"] == "rfi" else None,
    }
    for name, key in rules.items():
        spans = collections.defaultdict(set)
        for rec, fold in zip(records, assignment):
            value = key(rec)
            if value:
                spans[value].add(fold)
        report[name] = sum(len(v) > 1 for v in spans.values())
    return report


def rotation(test_fold: int, n_folds: int = N_FOLDS) -> dict[str, list[int]]:
    """Test fold k, validation fold k+1, training on the other three."""
    validation = (test_fold + 1) % n_folds
    return {
        "test": [test_fold],
        "val": [validation],
        "train": [k for k in range(n_folds) if k not in (test_fold, validation)],
    }


def load_folds(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def fold_vector(folds: dict, assignment: int, plot_ids: list[str]) -> list[int]:
    """Fold per plot id for one assignment. Raises if any id is absent."""
    lookup = dict(zip(folds["plot_ids"], folds["assignments"][assignment]["folds"]))
    return [lookup[p] for p in plot_ids]


def shared_sources(records: list[dict]) -> set[str]:
    """Source ids seen in more than one survey."""
    spans = collections.defaultdict(set)
    for rec in records:
        if rec["source_id"]:
            spans[rec["source_id"]].add(rec["survey"])
    return {k for k, v in spans.items() if len(v) > 1}
