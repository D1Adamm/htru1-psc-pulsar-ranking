"""Tests for the PSC manifest builder. No network, synthetic rows only."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"_test_{name}", REPO_ROOT / "scripts" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


build = _load("build_psc_manifest")


# A raw PSC paste row: ID RA DEC MJD Period DM X2 S1 S2 S3 S4 Score Grade Count
RAW_WITH_SUBPLOTS = "9214140\t17:39:54\t+13:08:15\t57278\t2939.411\t43.4\t24.648\t3\t3\t3\t3\t12\tKnown\t1"
# The same shape but with the Subplot columns blank, as PSC often emits.
RAW_BLANK_SUBPLOTS = "7241578\t01:58:24\t+09:06:32\t55585\t5807.635\t14.7\t634.936\t\t\t\t\t\tKnown\t2"
# The narrow form used by the v1 hand-stripped exports.
NARROW = "9063165\t17:13:51\t+34:29:52\t57963\t2490.438\t23.7\t3.01\tKnown\t1"


class TestParseRow(unittest.TestCase):
    def _parse(self, line: str):
        return build.parse_row(line.split("\t"), path=Path("x.tsv"), lineno=1)

    def test_parses_full_width_row_with_subplots(self):
        row = self._parse(RAW_WITH_SUBPLOTS)
        self.assertEqual(row["plot_id"], "9214140")
        self.assertEqual(row["period_ms"], "2939.411")
        self.assertEqual(row["dm"], "43.4")
        self.assertEqual(row["grade"], "Known")
        self.assertEqual(row["psc_count"], "1")

    def test_parses_row_with_blank_subplot_columns(self):
        row = self._parse(RAW_BLANK_SUBPLOTS)
        self.assertEqual(row["plot_id"], "7241578")
        self.assertEqual(row["chi2"], "634.936")
        self.assertEqual(row["grade"], "Known")
        self.assertEqual(row["psc_count"], "2")

    def test_parses_narrow_v1_style_row(self):
        row = self._parse(NARROW)
        self.assertEqual(row["plot_id"], "9063165")
        self.assertEqual(row["grade"], "Known")
        self.assertEqual(row["psc_count"], "1")

    def test_all_widths_agree_on_the_leading_columns(self):
        for line in (RAW_WITH_SUBPLOTS, RAW_BLANK_SUBPLOTS, NARROW):
            row = self._parse(line)
            self.assertTrue(row["ra"].count(":") == 2, line)
            self.assertTrue(row["dec"].startswith(("+", "-")), line)
            self.assertTrue(row["mjd"].isdigit(), line)

    def test_rejects_row_with_no_recognizable_grade(self):
        bad = "123\t01:02:03\t+04:05:06\t55555\t100.0\t1.0\t2.0\tWobble\t1"
        with self.assertRaises(ValueError) as ctx:
            self._parse(bad)
        self.assertIn("grade", str(ctx.exception).lower())

    def test_rejects_too_few_columns(self):
        with self.assertRaises(ValueError):
            self._parse("123\t01:02:03\tKnown")


class TestReadTab(unittest.TestCase):
    def _write(self, tmp: str, name: str, lines: list[str]) -> Path:
        path = Path(tmp) / name
        path.write_text("\n".join(lines) + "\n")
        return path

    def test_skips_header_and_blank_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, "rfi.tsv", [
                "ID\tRA\tDEC\tMJD\tPeriod\tDM\tX2\tGrade\tCount",
                "",
                "10457372\t18:33:20\t+35:52:59\t58031\t671.054\t0.0\t75466.5\tRFI\t1",
            ])
            rows = build.read_tab(path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["plot_id"], "10457372")

    def test_drops_duplicate_plot_ids_from_overlapping_pages(self):
        line = "10457372\t18:33:20\t+35:52:59\t58031\t671.054\t0.0\t75466.5\tRFI\t1"
        other = "3121594\t15:27:19\t+36:12:32\t58032\t671.089\t0.0\t75043.1\tRFI\t1"
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, "rfi.tsv", [line, other, line])
            rows = build.read_tab(path)
            self.assertEqual(len(rows), 2, "duplicate page rows must collapse, not error")

    def test_rejects_non_numeric_plot_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, "rfi.tsv", [
                "notanid\t18:33:20\t+35:52:59\t58031\t671.054\t0.0\t75466.5\tRFI\t1",
            ])
            with self.assertRaises(ValueError):
                build.read_tab(path)


class TestBuildRows(unittest.TestCase):
    def _make_dir(self, tmp: str, files: dict[str, list[str]]) -> Path:
        d = Path(tmp) / "src"
        d.mkdir()
        for name, lines in files.items():
            (d / name).write_text("\n".join(lines) + "\n")
        return d

    def test_new_pulsars_tab_becomes_labeled_positive(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = self._make_dir(tmp, {"new_pulsars.tsv": [
                "9999999\t01:02:03\t+04:05:06\t58000\t500.0\t20.0\t50.0\tNew\t1",
            ]})
            rows = build.build_rows(d)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["source_category"], "ao327_discovery")
            self.assertEqual(rows[0]["label"], "1")

    def test_grade_mismatch_is_fatal_and_lists_observed_grades(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = self._make_dir(tmp, {"rfi.tsv": [
                "10457372\t18:33:20\t+35:52:59\t58031\t671.054\t0.0\t75466.5\tRFI\t1",
                "6004822\t00:14:08\t+29:20:05\t57966\t33.321\t0.0\t47.535\tKnown\t1",
            ]})
            with self.assertRaises(ValueError) as ctx:
                build.build_rows(d)
            msg = str(ctx.exception)
            self.assertIn("6004822", msg)
            self.assertIn("Known", msg)

    def test_cross_tab_label_conflict_is_fatal(self):
        shared = "5555555\t01:02:03\t+04:05:06\t58000\t500.0\t20.0\t50.0"
        with tempfile.TemporaryDirectory() as tmp:
            d = self._make_dir(tmp, {
                "known_pulsars.tsv": [f"{shared}\tKnown\t1"],
                "rfi.tsv": [f"{shared}\tRFI\t1"],
            })
            with self.assertRaises(ValueError) as ctx:
                build.build_rows(d)
            self.assertIn("5555555", str(ctx.exception))

    def test_missing_tab_files_are_skipped_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = self._make_dir(tmp, {"noise.tsv": [
                "1871325\t02:25:52\t+07:01:40\t57167\t3.174\t46.5\t1.562\tNoise\t23",
            ]})
            rows = build.build_rows(d)
            self.assertEqual(len(rows), 1)

    def test_empty_source_dir_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "src"
            d.mkdir()
            with self.assertRaises(FileNotFoundError):
                build.build_rows(d)


class TestSourceGroups(unittest.TestCase):
    def _rows(self, specs):
        return [
            {"plot_id": str(i), "ra": ra, "dec": dec, "period_ms": p}
            for i, (ra, dec, p) in enumerate(specs)
        ]

    def test_same_position_and_period_group_together(self):
        rows = self._rows([
            ("03:05:44", "+19:18:25", "1387.637"),
            ("03:05:45", "+19:18:26", "1387.640"),
        ])
        build.assign_source_groups(rows, "known_pulsar")
        self.assertEqual(rows[0]["source_group"], rows[1]["source_group"])

    def test_same_position_different_period_stay_separate(self):
        rows = self._rows([
            ("17:39:54", "+13:08:15", "2939.411"),
            ("17:39:54", "+13:08:15", "1619.011"),
        ])
        build.assign_source_groups(rows, "known_pulsar")
        self.assertNotEqual(
            rows[0]["source_group"], rows[1]["source_group"],
            "harmonics at one position are different candidates",
        )

    def test_different_positions_stay_separate(self):
        rows = self._rows([
            ("03:05:44", "+19:18:25", "1387.637"),
            ("12:38:19", "+25:04:43", "1387.637"),
        ])
        build.assign_source_groups(rows, "known_pulsar")
        self.assertNotEqual(rows[0]["source_group"], rows[1]["source_group"])

    def test_noise_is_grouped_positionally_like_every_other_class(self):
        # Regression guard: an earlier version forced all noise into one group,
        # which distorted the per-source metric in EXP-002.
        rows = self._rows([
            ("02:25:52", "+07:01:40", "3.174"),
            ("02:25:52", "+07:01:40", "15.749"),
            ("02:25:52", "+07:01:40", "319.202"),
        ])
        build.assign_source_groups(rows, "noise")
        self.assertEqual(
            len({r["source_group"] for r in rows}), 3,
            "distinct trial periods must not collapse into a single noise group",
        )


if __name__ == "__main__":
    unittest.main()
