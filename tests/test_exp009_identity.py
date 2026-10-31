"""Tests for EXP-009 catalogue parsing, matching, and identity decisions."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


identity = load_script("build_source_identity")

CATALOGUE = """#CATALOGUE test
#
PSRJ     J0100+0000                    ref
RAJ      01:00:00                 2    ref
DECJ     +00:00:00                1    ref
P0       1.0                      5    ref
DM       20.0                     1    ref
@-----------------------------------------------------------------
PSRJ     J0200+0000                    ref
ELONG    30.0                     2    ref
ELAT     0.0                      1    ref
F0       2.0                      5    ref
DM       40.0                     1    ref
@-----------------------------------------------------------------
PSRJ     J0300+0000                    ref
P0       1.0                      5    ref
DM       10.0                     1    ref
@-----------------------------------------------------------------
"""


def row(plot_id, ra, dec, period, dm, survey="ao327"):
    return {
        "plot_id": plot_id,
        "survey": survey,
        "ra_deg": ra,
        "dec_deg": dec,
        "period_ms": period,
        "dm": dm,
    }


def decision(key, action, target="", note="note"):
    return {"key": key, "action": action, "target": target, "note": note}


class CatalogueParsingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalogue = identity.parse_catalogue(CATALOGUE)

    def test_positionless_records_are_dropped(self) -> None:
        self.assertEqual([c["jname"] for c in self.catalogue], ["J0100+0000", "J0200+0000"])

    def test_period_from_frequency(self) -> None:
        self.assertAlmostEqual(self.catalogue[1]["period_ms"], 500.0)

    def test_ecliptic_conversion_on_the_ecliptic(self) -> None:
        ra, dec = identity.ecliptic_to_equatorial(0.0, 0.0)
        self.assertAlmostEqual(ra, 0.0, places=6)
        self.assertAlmostEqual(dec, 0.0, places=6)
        ra, dec = identity.ecliptic_to_equatorial(90.0, 0.0)
        self.assertAlmostEqual(ra, 90.0, places=6)
        self.assertAlmostEqual(dec, 23.4392911, places=6)


class MatchingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalogue = identity.parse_catalogue(CATALOGUE)

    def test_lowest_order_harmonic_wins(self) -> None:
        self.assertEqual(identity.harmonic(0.5, 1e-3, 8)[:2], (1, 2))
        self.assertIsNone(identity.harmonic(0.5123, 1e-3, 8))

    def test_classes_follow_evidence(self) -> None:
        rows = [
            row("direct", 15.0, 0.0, 1000.0, 20.0),
            row("harmonic", 15.0, 0.0, 1000.0 * 13 / 3, 20.0),
            row("nonharmonic", 15.0, 0.0, 1010.0, 20.0),
            row("wrong_dm", 15.0, 0.0, 500.0, 60.0),
            row("nothing", 200.0, 60.0, 777.0, 90.0),
        ]
        matched = identity.match_rows(rows, self.catalogue)
        self.assertEqual([m["cls"][:2] for m in matched], ["A ", "B1", "B2", "B3", "D "])
        self.assertEqual(matched[0]["ratio"], [1, 1])
        self.assertIsNone(matched[4]["jname"])

    def test_no_anchor_gives_weak_or_medium_class_only(self) -> None:
        rows = [row("alone", 15.0, 0.0, 1234.5, 20.0)]
        matched = identity.match_rows(rows, self.catalogue)
        self.assertEqual(matched[0]["cls"][:2], "C2")


class DecisionTests(unittest.TestCase):
    """Built around the real 3013353 case: one row chained into a neighbour's group."""

    def setUp(self) -> None:
        self.catalogue = identity.parse_catalogue(CATALOGUE)
        self.rows = [
            row("near1", 15.0, 0.0, 1000.0, 20.0),
            row("near2", 15.1, 0.0, 2000.0, 20.0),
            row("3013353", 15.5, 0.0, 777.7, 21.0, survey="gbncc"),
            row("uncertain", 90.0, 40.0, 321.0, 77.0),
        ]
        self.matched = identity.match_rows(self.rows, self.catalogue)

    def build(self, decisions):
        return {r["plot_id"]: r for r in identity.assign_groups(self.rows, self.matched, decisions)}

    def test_chained_row_starts_in_the_neighbour_group(self) -> None:
        table = self.build([])
        self.assertEqual(table["3013353"]["source_id"], table["near1"]["source_id"])

    def test_split_changes_source_but_not_fold_group(self) -> None:
        table = self.build([decision("3013353", "split_source", "J9999+0000")])
        self.assertNotEqual(table["3013353"]["source_id"], table["near1"]["source_id"])
        self.assertEqual(table["3013353"]["fold_group_id"], table["near1"]["fold_group_id"])
        self.assertEqual(table["3013353"]["catalog_name"], "J9999+0000")

    def test_merge_joins_the_target_source(self) -> None:
        table = self.build([decision("uncertain", "merge_into", "J0100+0000")])
        self.assertEqual(table["uncertain"]["source_id"], table["near1"]["source_id"])

    def test_label_uncertain_is_flagged_and_never_fundamental(self) -> None:
        table = self.build([decision("near1", "label_uncertain")])
        self.assertEqual(table["near1"]["label_uncertain"], 1)
        self.assertEqual(table["near1"]["fundamental"], 0)

    def test_fundamental_requires_true_period(self) -> None:
        table = self.build([])
        self.assertEqual(table["near1"]["fundamental"], 1)
        self.assertEqual(table["near2"]["fundamental"], 0)

    def test_withheld_name_keeps_the_group_without_a_claim(self) -> None:
        table = self.build([decision("J0100+0000", "withhold_name")])
        self.assertEqual(table["near1"]["catalog_name"], "")
        self.assertEqual(table["near1"]["match_class"][:1], "A")

    def test_output_is_deterministic(self) -> None:
        first = identity.render_csv(identity.assign_groups(self.rows, self.matched, []))
        second = identity.render_csv(identity.assign_groups(self.rows, self.matched, []))
        self.assertEqual(first, second)


class FrozenTableTests(unittest.TestCase):
    """Checks on the committed derived tables; no raw data or network needed."""

    @classmethod
    def setUpClass(cls) -> None:
        with open(REPO_ROOT / "data/psc/identity/source_identity_v1.csv", newline="") as f:
            cls.rows = {r["plot_id"]: r for r in csv.DictReader(f)}
        cls.folds = json.loads((REPO_ROOT / "data/psc/folds/folds_v1.json").read_text())

    def test_recorded_counts(self) -> None:
        self.assertEqual(len(self.rows), 800)
        self.assertEqual(len({r["source_id"] for r in self.rows.values()}), 102)
        self.assertEqual(sum(int(r["label_uncertain"]) for r in self.rows.values()), 9)
        self.assertEqual(sum(int(r["fundamental"]) for r in self.rows.values()), 242)

    def test_row_3013353_is_its_own_source(self) -> None:
        split = self.rows["3013353"]
        self.assertEqual(split["catalog_name"], "J1926+1648")
        self.assertEqual(sum(r["source_id"] == split["source_id"] for r in self.rows.values()), 1)
        neighbours = [r for r in self.rows.values() if r["catalog_name"] == "J1935+1616"]
        self.assertTrue(neighbours)
        self.assertEqual(split["fold_group_id"], neighbours[0]["fold_group_id"])

    def test_merged_rows_join_their_source(self) -> None:
        for plot_id, name in (("8880312", "J1136+1551"), ("6454363", "J2317+2149")):
            target = next(r for r in self.rows.values() if r["catalog_name"] == name and r["plot_id"] != plot_id)
            self.assertEqual(self.rows[plot_id]["source_id"], target["source_id"])

    def test_no_source_crosses_folds(self) -> None:
        for assignment in self.folds["assignments"]:
            fold_of = dict(zip(self.folds["plot_ids"], assignment["folds"]))
            spans: dict[str, set[int]] = {}
            for plot_id, rec in self.rows.items():
                spans.setdefault(rec["source_id"], set()).add(fold_of[plot_id])
            self.assertTrue(all(len(v) == 1 for v in spans.values()))

    def test_fold_seeds_are_frozen(self) -> None:
        self.assertEqual([a["seed"] for a in self.folds["assignments"]], [5, 44, 52])


if __name__ == "__main__":
    unittest.main()
