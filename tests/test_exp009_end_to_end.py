"""End-to-end checks of the EXP-009 scripts on small synthetic data.

Nothing here reads real candidates or writes to the repository. These exist so a
bug in evaluation cannot first appear after real training.
"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier

from src.datasets.psc_folds import (
    assign_folds,
    attach_identity,
    linked_components,
    load_candidate_records,
    read_identity,
    sha256_file,
)
from src.models.three_class_head import ThreeClassCNN
from tests.test_exp009_identity import load_script

train = load_script("train_exp009")
baselines = load_script("run_exp009_baselines")
evaluate = load_script("evaluate_exp009")
build_folds = load_script("build_source_folds")


def payload(plot_id, category, obs, rng):
    fvp, tvp = rng.random((8, 40)) * 0.2, rng.random((6, 40)) * 0.2
    if category == "known_pulsar":
        fvp[:, 10:13] += 1.0
        tvp[:, 10:13] += 1.0
    elif category == "rfi":
        fvp[:, 5:25] += 1.0
        tvp[:, 5:25] += 1.0
    return {
        "FvP": fvp.tolist(),
        "TvP": tvp.tolist(),
        "DMc": rng.random(21).tolist(),
        "meta": {
            "MJD": "57000.5",
            "period": str(0.1 + rng.random()),
            "bestdm": str(5 + 50 * rng.random()),
            "sigma": str(5 + 20 * rng.random()),
            "input_file": f"{obs}_DM10.00_Z0_ACCEL_Cand_1.pfd",
            "T_sample": "0.001",
            "observation_length": "60.0",
        },
    }


class Fixture:
    """A manifest, JSON cache, identity table, and folds for two surveys."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.cache = root / "cache"
        self.cache.mkdir()
        rng = np.random.default_rng(0)
        rows, identity, night = [], [], 1000
        for survey in ("ao327", "gbncc"):
            for s in range(14):
                source = "SHARED" if s == 0 else f"{survey}-S{s}"
                for d in range(2):
                    rows.append((f"{survey}p{s}d{d}", survey, "known_pulsar", f"{survey}_pobs{s}", night + s))
                    identity.append(
                        {
                            "plot_id": f"{survey}p{s}d{d}",
                            "source_id": source,
                            "fold_group_id": source,
                            "label_uncertain": int(s == 1 and d == 1),
                            "fundamental": int(d == 0),
                        }
                    )
            for i in range(14):
                rows.append((f"{survey}r{i}", survey, "rfi", f"{survey}_robs{i}", night + 100 + i))
                rows.append((f"{survey}n{i}", survey, "noise", f"{survey}_nobs{i}", night + 200))
            night += 1000
        self.manifest = root / "manifest.csv"
        with open(self.manifest, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["plot_id", "survey", "source_category", "chi2", "ra", "dec"])
            for plot_id, survey, category, obs, n in rows:
                writer.writerow([plot_id, survey, category, "3.0", "01:00:00", "+01:00:00"])
                data = payload(plot_id, category, obs, rng)
                data["meta"]["MJD"] = str(n + 0.5)
                (self.cache / f"{plot_id}.json").write_text(json.dumps(data))
        self.identity = root / "identity.csv"
        with open(self.identity, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(identity[0]))
            writer.writeheader()
            writer.writerows(identity)
        self.records = load_candidate_records(self.manifest, self.cache)
        attach_identity(self.records, read_identity(self.identity))
        components = linked_components(self.records)
        self.folds_data = {
            "manifest_sha256": sha256_file(self.manifest),
            "identity_sha256": sha256_file(self.identity),
            "plot_ids": [r["plot_id"] for r in self.records],
            "assignments": [{"seed": s, "folds": assign_folds(self.records, components, s)} for s in (1, 2, 3)],
        }
        self.folds = root / "folds.json"
        self.folds.write_text(json.dumps(self.folds_data))

    def arguments(self) -> list[str]:
        return [
            "--manifest", str(self.manifest),
            "--cache-dir", str(self.cache),
            "--identity", str(self.identity),
            "--folds", str(self.folds),
        ]  # fmt: skip


class ScriptEntryPointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.fixture = Fixture(Path(cls.tmp.name))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def clean_git(self):
        return mock.patch.object(train, "git_state", return_value={"commit": "abc", "dirty": False, "dirty_entries": []})

    def test_training_refuses_uncommitted_code(self) -> None:
        with tempfile.TemporaryDirectory() as out, mock.patch.object(train, "OUTPUT_ROOT", Path(out) / "runs"):
            with mock.patch.object(train, "git_state", return_value={"commit": "abc", "dirty": True, "dirty_entries": ["x"]}):
                code = train.main(["--arm", "E2", "--assignment", "0", "--fold", "0", *self.fixture.arguments()])
            self.assertEqual(code, 1)
            self.assertFalse((Path(out) / "runs").exists())

    def test_dry_run_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as out, mock.patch.object(train, "OUTPUT_ROOT", Path(out) / "runs"), self.clean_git():
            code = train.main(["--arm", "E2", "--assignment", "0", "--fold", "0", "--dry-run", *self.fixture.arguments()])
            self.assertEqual(code, 0)
            self.assertEqual(list(Path(out).iterdir()), [])

    def test_training_writes_a_run_and_refuses_to_repeat_it(self) -> None:
        with tempfile.TemporaryDirectory() as out, mock.patch.object(train, "OUTPUT_ROOT", Path(out) / "runs"), self.clean_git():
            command = ["--arm", "E2", "--assignment", "0", "--fold", "0", "--threads", "1", *self.fixture.arguments()]
            self.assertEqual(train.main(command), 0)
            run_dir = Path(out) / "runs" / "E2_a0_f0_s42"
            metadata = json.loads((run_dir / "metadata.json").read_text())
            self.assertEqual(metadata["git"]["commit"], "abc")
            self.assertEqual(metadata["identity_sha256"], sha256_file(self.fixture.identity))
            before = (run_dir / "metadata.json").read_bytes()
            self.assertEqual(train.main(command), 1)
            self.assertEqual(train.main([*command, "--skip-completed"]), 0)
            self.assertEqual((run_dir / "metadata.json").read_bytes(), before)

    def test_stale_folds_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as out:
            stale = Path(out) / "folds.json"
            stale.write_text(json.dumps({**self.fixture.folds_data, "identity_sha256": "other"}))
            arguments = self.fixture.arguments()
            arguments[arguments.index("--folds") + 1] = str(stale)
            with self.clean_git(), self.assertRaises(ValueError):
                train.main(["--arm", "E2", "--assignment", "0", "--fold", "0", "--dry-run", *arguments])

    def test_seed_verification_alone_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as out:
            output = Path(out) / "folds.json"
            arguments = ["--manifest", str(self.fixture.manifest), "--cache-dir", str(self.fixture.cache),
                         "--identity", str(self.fixture.identity), "--output", str(output), "--verify-selection"]
            with mock.patch.object(build_folds, "select_seeds", return_value=list(build_folds.SEEDS)):
                self.assertEqual(build_folds.main(arguments), 0)
            with mock.patch.object(build_folds, "select_seeds", return_value=[1, 2, 3]):
                self.assertEqual(build_folds.main(arguments), 1)
            self.assertFalse(output.exists())

    def test_baselines_write_every_row_and_refuse_to_repeat(self) -> None:
        small_forest = lambda n, **kw: RandomForestClassifier(5, **kw)  # noqa: E731
        clean = {"commit": "abc", "dirty": False, "dirty_entries": []}
        with tempfile.TemporaryDirectory() as out, mock.patch.object(baselines, "OUTPUT_ROOT", Path(out) / "baselines"):
            with mock.patch.object(baselines, "RandomForestClassifier", small_forest), mock.patch.object(baselines, "LEAF_GRID", (2,)), \
                    mock.patch.object(baselines, "git_state", return_value=clean):
                self.assertEqual(baselines.main(self.fixture.arguments()), 0)
            self.assertEqual(json.loads((Path(out) / "baselines" / "metadata.json").read_text())["code"], clean)
            with open(Path(out) / "baselines" / "scores.csv", newline="") as f:
                rows = list(csv.DictReader(f))
            within = [r for r in rows if r["evaluation"] == "within" and r["model"] == "period_dm" and r["assignment"] == "0"]
            self.assertEqual(sorted(r["plot_id"] for r in within), sorted(r["plot_id"] for r in self.fixture.records))
            self.assertEqual({r["model"] for r in rows}, {"period_dm", "geometry_acquisition", "sigma_chi2", "all_metadata", "duty_forest", "duty_dmc"})
            self.assertEqual(baselines.main(self.fixture.arguments()), 1)


class EvaluationTests(unittest.TestCase):
    """Runs synthetic scores through the real evaluation, without any training."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.fixture = Fixture(Path(cls.tmp.name))
        cls.records = cls.fixture.records

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def fake_runs(self, assignments) -> dict:
        """Arms get scores of decreasing quality, with C1 weakest."""
        quality = {"E1": 0.3, "E2": 0.4, "E3": 0.5, "C1": 3.0, "L1": 0.5, "L2": 0.6}
        rng = np.random.default_rng(5)
        truth = {r["plot_id"]: float(r["category"] == "known_pulsar") for r in self.records}
        ids = [r["plot_id"] for r in self.records]

        def score(plot_id, arm):
            return float(np.clip(0.5 * truth[plot_id] + 0.25 + rng.normal(0, 0.1 * quality[arm]), 0.001, 0.999))

        runs = {}
        for a in assignments:
            folds = self.fixture.folds_data["assignments"][a]["folds"]
            for arm in ("E1", "E2", "E3", "C1"):
                for k in range(5):
                    spec = train.RunSpec(arm, a, fold=k)
                    splits = train.split_rows(spec, self.records, folds)
                    runs[spec.run_id] = {
                        "metadata": {"arm": arm, "assignment": a, "target_survey": None},
                        "test": {ids[i]: score(ids[i], arm) for i in splits["test"]},
                        "val": {ids[i]: score(ids[i], arm) for i in splits["val_scored"]},
                    }
            for arm in ("L1", "L2"):
                for src, dst in train.DIRECTIONS:
                    spec = train.RunSpec(arm, a, source_survey=src, target_survey=dst)
                    splits = train.split_rows(spec, self.records, folds)
                    runs[spec.run_id] = {
                        "metadata": {"arm": arm, "assignment": a, "target_survey": dst},
                        "test": {ids[i]: score(ids[i], arm) for i in splits["test"]},
                        "val": {ids[i]: score(ids[i], arm) for i in splits["val_scored"]},
                    }
        return runs

    def static(self) -> dict:
        rng = np.random.default_rng(9)
        truth = np.array([r["category"] == "known_pulsar" for r in self.records], dtype=float)
        n = len(truth)
        return {
            "duty_cnn_input": truth + rng.normal(0, 0.8, n),
            "duty_native": truth + rng.normal(0, 0.6, n),
            "htru1": rng.random(n),
            "sigma": np.array([r["sigma"] for r in self.records]),
        }

    def run_stage(self, stage, assignments):
        with tempfile.TemporaryDirectory() as out:
            with mock.patch.object(evaluate, "BASELINE_SCORES", Path(out) / "absent.csv"):
                return evaluate.evaluate(stage, self.records, self.static(), self.fake_runs(assignments), 20, self.fixture.cache, False)

    def test_pilot_report_is_complete_and_serializable(self) -> None:
        report = self.run_stage("pilot", [0])
        self.assertIn(report["pilot_decision"]["outcome"], {"clear_signal", "inconclusive", "provisional_negative"})
        self.assertNotIn("claim_a", report)
        section = report["populations"]["GBNCC RFI"]
        self.assertIn("E1 - C1, typical AUC", section["contrasts"])
        self.assertIn("E1 - duty_cnn_input, FPR at 90", section["contrasts"])
        self.assertIn("one_detection_strongest_auc", section["models"]["E1"])
        self.assertIn("E1 - C1, typical AUC", section["component_resampling"])
        self.assertIn("L1 - duty_cnn_input, typical AUC", report["cross_survey"]["gbncc"]["contrasts"])
        self.assertIn("E1 + duty_cnn_input, GBNCC RFI", report["claim_c"])
        self.assertEqual(report["interval_level"], 0.95)
        json.dumps(report)
        self.assertIn("Pilot decision", evaluate.render_report(report))

    def test_full_report_applies_the_claims(self) -> None:
        report = self.run_stage("full", [0, 1, 2])
        self.assertIn(report["claim_a"], {"success", "failure", "inconclusive"})
        self.assertIn(report["claim_b"], {"success", "failure", "inconclusive"})
        self.assertNotIn("pilot_decision", report)
        self.assertAlmostEqual(report["interval_level"], evaluate.FINAL_LEVEL)
        values = report["populations"]["GBNCC RFI"]["models"]["E1"]["typical_auc_by_assignment"]
        self.assertEqual(len(values), 3)
        json.dumps(report)

    def test_cross_survey_population_drops_shared_sources(self) -> None:
        report = self.run_stage("pilot", [0])
        self.assertEqual(report["populations"]["GBNCC RFI"]["sources"] - report["cross_survey"]["gbncc"]["sources"], 1)

    def test_double_scored_row_is_rejected(self) -> None:
        runs = self.fake_runs([0])
        first = next(r for r in runs.values() if r["metadata"]["arm"] == "E1")
        other = [r for r in runs.values() if r["metadata"]["arm"] == "E1" and r is not first][0]
        other["test"].update({next(iter(first["test"])): 0.5})
        with self.assertRaises(ValueError):
            evaluate.arm_scores("E1", runs, [r["plot_id"] for r in self.records], [0])

    def test_missing_runs_block_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as out:
            with mock.patch.object(evaluate, "RUN_ROOT", Path(out)):
                runs, missing = evaluate.read_run_scores(["E1_a0_f0_s42"], self.records, self.fixture.folds_data)
        self.assertEqual((runs, missing), ({}, ["E1_a0_f0_s42"]))

    def test_run_from_other_identity_table_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as out:
            run_dir = Path(out) / "E1_a0_f0_s42"
            run_dir.mkdir()
            (run_dir / "metadata.json").write_text(json.dumps({"identity_sha256": "other"}))
            with mock.patch.object(evaluate, "RUN_ROOT", Path(out)), self.assertRaises(ValueError):
                evaluate.read_run_scores(["E1_a0_f0_s42"], self.records, self.fixture.folds_data)

    def test_saturated_probabilities_are_ranked_from_logits(self) -> None:
        identity = self.fixture.folds_data["identity_sha256"]
        with tempfile.TemporaryDirectory() as out:
            run_dir = Path(out) / "E1_a0_f0_s42"
            run_dir.mkdir()
            (run_dir / "metadata.json").write_text(json.dumps({"identity_sha256": identity}))
            (run_dir / "scores.csv").write_text(
                "plot_id,split,logit_known_pulsar,logit_rfi,logit_noise,p_pulsar\n"
                "a,test,-12.0,9.0,1.0,0.00000000\n"
                "b,test,-15.0,9.0,1.0,0.00000000\n"
                "c,val,2.0,1.0,0.0,0.66524096\n"
            )
            with mock.patch.object(evaluate, "RUN_ROOT", Path(out)):
                runs, _ = evaluate.read_run_scores(["E1_a0_f0_s42"], self.records, self.fixture.folds_data)
        scores = runs["E1_a0_f0_s42"]
        self.assertGreater(scores["test"]["a"], scores["test"]["b"])
        logits = np.array([2.0, 1.0, 0.0])
        p = np.exp(logits[0]) / np.exp(logits).sum()
        self.assertAlmostEqual(scores["val"]["c"], np.log(p / (1 - p)), places=12)

    def test_output_name_cannot_leave_evaluation_directory(self) -> None:
        args = ["--stage", "pilot", "--name", "../escaped", "--manifest", str(self.fixture.manifest),
                "--cache-dir", str(self.fixture.cache), "--identity", str(self.fixture.identity), "--folds", str(self.fixture.folds)]
        with tempfile.TemporaryDirectory() as out, \
                mock.patch.object(evaluate, "OUTPUT_ROOT", Path(out) / "evaluation"), \
                mock.patch.object(evaluate, "load_folds", side_effect=AssertionError("loaded data")):
            with self.assertRaisesRegex(ValueError, "refusing to write outside"):
                evaluate.main(args)
            self.assertFalse((Path(out) / "escaped").exists())

    def test_interventions_rescore_without_changing_the_checkpoint(self) -> None:
        torch.manual_seed(0)
        ids = [r["plot_id"] for r in self.records]
        with tempfile.TemporaryDirectory() as out:
            run_dir = Path(out) / "E1_a0_f0_s42"
            run_dir.mkdir()
            torch.save({"model_state_dict": ThreeClassCNN().state_dict()}, run_dir / "best.pt")
            before = (run_dir / "best.pt").read_bytes()
            runs = {"E1_a0_f0_s42": {"metadata": {"arm": "E1", "assignment": 0}, "test": {p: 0.5 for p in ids[:40]}, "val": {}}}
            with mock.patch.object(evaluate, "RUN_ROOT", Path(out)):
                effects = evaluate.intervention_changes(["E1_a0_f0_s42"], runs, self.records, self.fixture.cache)
            self.assertEqual((run_dir / "best.pt").read_bytes(), before)
        self.assertEqual(set(effects["E1_a0_f0_s42"]), set(evaluate.INTERVENTIONS))


if __name__ == "__main__":
    unittest.main()
