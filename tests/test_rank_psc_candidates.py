"""Tests for the batch PSC ranking script. No live network access."""

from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _load_script_module(name: str):
    path = SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_test_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


rank_psc = _load_script_module("rank_psc_candidates")

from src.models.baseline_cnn import BaselineCNN


def make_payload(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "FvP": rng.random((128, 50)).tolist(),
        "TvP": rng.random((48, 50)).tolist(),
        "profile": rng.random(50).tolist(),
        "meta": {"telescope": "Arecibo", "period": 0.5, "sigma": 7.0},
    }


def write_manifest(path: Path, plot_ids: list[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["plot_id", "label", "source_category", "source_group"]
        )
        writer.writeheader()
        for i, plot_id in enumerate(plot_ids):
            writer.writerow(
                {
                    "plot_id": plot_id,
                    "label": "1" if i % 2 == 0 else "0",
                    "source_category": "known_pulsar" if i % 2 == 0 else "rfi",
                    "source_group": f"g{i:03d}",
                }
            )


class TestReadManifest(unittest.TestCase):
    def test_reads_rows_and_requires_plot_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.csv"
            write_manifest(path, ["111", "222", "333"])
            rows = rank_psc.read_manifest(path)
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["plot_id"], "111")

            bad = Path(tmp) / "bad.csv"
            bad.write_text("id,label\n1,0\n")
            with self.assertRaises(ValueError):
                rank_psc.read_manifest(bad)

    def test_rejects_duplicate_and_blank_plot_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            dup = Path(tmp) / "dup.csv"
            dup.write_text("plot_id,label\n111,1\n111,0\n")
            with self.assertRaises(ValueError) as ctx:
                rank_psc.read_manifest(dup)
            self.assertIn("duplicate", str(ctx.exception))

            blank = Path(tmp) / "blank.csv"
            blank.write_text("plot_id,label\n111,1\n,0\n")
            with self.assertRaises(ValueError) as ctx:
                rank_psc.read_manifest(blank)
            self.assertIn("blank", str(ctx.exception))

    def test_missing_manifest_raises(self):
        with self.assertRaises(FileNotFoundError):
            rank_psc.read_manifest(Path("/nonexistent/manifest.csv"))


class TestCache(unittest.TestCase):
    def test_roundtrip_and_prefers_cache_over_fetch(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            payload = make_payload(1)
            rank_psc.save_cached_payload(cache, "555", payload)

            calls: list[str] = []

            def fetch(plot_id, timeout_s=30.0):
                calls.append(plot_id)
                return make_payload(2)

            got, status = rank_psc.get_payload(
                "555", cache_dir=cache, fetch=fetch
            )
            self.assertEqual(status, "cached")
            self.assertEqual(calls, [], "cached candidate must not hit the network")
            self.assertEqual(len(got["FvP"]), 128)

    def test_fetch_writes_cache_and_no_cache_bypasses_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            calls: list[str] = []

            def fetch(plot_id, timeout_s=30.0):
                calls.append(plot_id)
                return make_payload(3)

            _, status = rank_psc.get_payload("777", cache_dir=cache, fetch=fetch)
            self.assertEqual(status, "fetched")
            self.assertTrue(rank_psc.cache_path_for(cache, "777").is_file())

            # Second call is served from the cache written above.
            _, status2 = rank_psc.get_payload("777", cache_dir=cache, fetch=fetch)
            self.assertEqual(status2, "cached")
            self.assertEqual(len(calls), 1)

            # use_cache=False always refetches.
            _, status3 = rank_psc.get_payload(
                "777", cache_dir=cache, use_cache=False, fetch=fetch
            )
            self.assertEqual(status3, "fetched")
            self.assertEqual(len(calls), 2)

    def test_corrupt_cache_falls_back_to_fetch(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            rank_psc.cache_path_for(cache, "888").write_text("{not json")

            def fetch(plot_id, timeout_s=30.0):
                return make_payload(4)

            _, status = rank_psc.get_payload("888", cache_dir=cache, fetch=fetch)
            self.assertEqual(status, "fetched")


class TestScoreManifestRows(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = BaselineCNN()
        self.model.eval()
        self.device = torch.device("cpu")

    def test_scores_all_rows_with_mocked_fetch(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "m.csv"
            write_manifest(manifest, ["101", "102", "103"])
            rows = rank_psc.read_manifest(manifest)

            def fetch(plot_id, timeout_s=30.0):
                return make_payload(int(plot_id))

            results = rank_psc.score_manifest_rows(
                rows,
                self.model,
                self.device,
                cache_dir=Path(tmp) / "cache",
                fetch=fetch,
            )

            self.assertEqual(len(results), 3)
            for r in results:
                self.assertEqual(r["fetch_status"], "fetched")
                self.assertEqual(r["preprocessing_status"], "ok")
                self.assertEqual(r["error"], "")
                self.assertIsInstance(r["score"], float)
                self.assertGreaterEqual(r["score"], 0.0)
                self.assertLessEqual(r["score"], 1.0)

    def test_continues_past_fetch_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "m.csv"
            write_manifest(manifest, ["201", "202", "203"])
            rows = rank_psc.read_manifest(manifest)

            def fetch(plot_id, timeout_s=30.0):
                if plot_id == "202":
                    raise RuntimeError("PSC HTTP 404")
                return make_payload(int(plot_id))

            results = rank_psc.score_manifest_rows(
                rows,
                self.model,
                self.device,
                cache_dir=Path(tmp) / "cache",
                fetch=fetch,
            )

            self.assertEqual(len(results), 3, "a failure must not drop other rows")
            failed = [r for r in results if r["plot_id"] == "202"][0]
            self.assertEqual(failed["fetch_status"], "failed")
            self.assertEqual(failed["preprocessing_status"], "skipped")
            self.assertIn("404", failed["error"])
            self.assertIsNone(failed["score"])
            ok = [r for r in results if r["plot_id"] != "202"]
            self.assertTrue(all(r["score"] is not None for r in ok))

    def test_continues_past_malformed_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "m.csv"
            write_manifest(manifest, ["301", "302"])
            rows = rank_psc.read_manifest(manifest)

            def fetch(plot_id, timeout_s=30.0):
                if plot_id == "302":
                    return {"FvP": [[1.0, 2.0], [3.0]], "TvP": [[1.0]]}  # ragged
                return make_payload(1)

            results = rank_psc.score_manifest_rows(
                rows,
                self.model,
                self.device,
                cache_dir=Path(tmp) / "cache",
                fetch=fetch,
            )

            bad = [r for r in results if r["plot_id"] == "302"][0]
            self.assertEqual(bad["fetch_status"], "fetched")
            self.assertEqual(bad["preprocessing_status"], "failed")
            self.assertIsNone(bad["score"])
            self.assertTrue(bad["error"])

    def test_delay_applied_only_on_network_fetch(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "m.csv"
            write_manifest(manifest, ["401", "402"])
            rows = rank_psc.read_manifest(manifest)
            cache = Path(tmp) / "cache"
            rank_psc.save_cached_payload(cache, "401", make_payload(1))

            slept: list[float] = []

            def fetch(plot_id, timeout_s=30.0):
                return make_payload(2)

            rank_psc.score_manifest_rows(
                rows,
                self.model,
                self.device,
                cache_dir=cache,
                fetch=fetch,
                delay_s=0.25,
                sleep=slept.append,
            )

            self.assertEqual(
                slept, [0.25], "only the uncached candidate should incur a delay"
            )


class TestRankAndSave(unittest.TestCase):
    def test_ranks_descending_and_sinks_failures(self):
        results = [
            {"plot_id": "a", "score": 0.10},
            {"plot_id": "b", "score": 0.99},
            {"plot_id": "c", "score": None, "error": "boom"},
            {"plot_id": "d", "score": 0.50},
        ]
        ranked = rank_psc.rank_results(results)

        self.assertEqual([r["plot_id"] for r in ranked], ["b", "d", "a", "c"])
        self.assertEqual([r["rank"] for r in ranked[:3]], [1, 2, 3])
        self.assertEqual(ranked[-1]["rank"], "", "failed rows get no rank")

    def test_save_ranked_csv_headers_and_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "ranked.csv"
            rows = rank_psc.rank_results(
                [
                    {"plot_id": "a", "score": 0.9, "logit": 2.2, "label": "1"},
                    {"plot_id": "b", "score": 0.1, "logit": -2.2, "label": "0"},
                ]
            )
            rank_psc.save_ranked_csv(out, rows)

            with open(out, newline="") as f:
                read_back = list(csv.DictReader(f))

            self.assertEqual(len(read_back), 2)
            self.assertEqual(read_back[0]["plot_id"], "a")
            self.assertEqual(read_back[0]["rank"], "1")
            for field in rank_psc.OUTPUT_FIELDNAMES:
                self.assertIn(field, read_back[0])


if __name__ == "__main__":
    unittest.main()
