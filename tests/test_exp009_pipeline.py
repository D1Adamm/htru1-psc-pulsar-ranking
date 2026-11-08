"""Tests for EXP-009 run plans, data rules, safe outputs, and decision logic.

All data here is synthetic. The short training calls only exercise code paths.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.datasets.psc_dataset import PSCCachedDataset
from src.models.baseline_cnn import BaselineCNN
from tests.test_exp009_identity import load_script

train = load_script("train_exp009")
baselines = load_script("run_exp009_baselines")
evaluate = load_script("evaluate_exp009")
verify = load_script("verify_exp009_protected")


def make_records():
    """Three folds' worth of each class per survey, with one shared source and one uncertain row."""
    records = []
    for survey in ("ao327", "gbncc"):
        for k in range(5):
            for category in ("known_pulsar", "rfi", "noise"):
                plot_id = f"{survey}-{category}-{k}"
                is_pulsar = category == "known_pulsar"
                records.append(
                    {
                        "plot_id": plot_id,
                        "survey": survey,
                        "category": category,
                        "source_id": f"src-{survey}-{k}" if is_pulsar else None,
                        "obs": plot_id,
                        "night": 100 + k,
                        "label_uncertain": False,
                    }
                )
    records[0]["source_id"] = records[15]["source_id"] = "shared"
    records[3]["label_uncertain"] = True
    return records


FOLDS = [int(r["plot_id"].rsplit("-", 1)[1]) for r in make_records()]
UNCERTAIN_INDEX = 3


class PlanTests(unittest.TestCase):
    def test_plan_sizes(self) -> None:
        self.assertEqual(len(train.plan_specs("pilot")), 24)
        self.assertEqual(len(train.plan_specs("extension")), 38)
        self.assertEqual(len(train.plan_specs("e3-completion")), 10)

    def test_pilot_arms_and_assignment(self) -> None:
        specs = train.plan_specs("pilot")
        self.assertEqual({s.arm for s in specs}, {"E1", "E2", "E3", "C1", "L1", "L2"})
        self.assertEqual({s.assignment for s in specs}, {0})

    def test_run_ids_and_seeds(self) -> None:
        specs = [s for name in ("pilot", "extension", "e3-completion") for s in train.plan_specs(name)]
        self.assertEqual(len({s.run_id for s in specs}), 72)
        self.assertEqual(train.RunSpec("E1", 1, fold=2).model_seed, 144)

    def test_compared_arms_share_a_seed(self) -> None:
        seeds = {train.RunSpec(arm, 0, fold=3).model_seed for arm in ("E1", "E2", "E3", "C1")}
        self.assertEqual(len(seeds), 1)


class SplitRuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = make_records()

    def test_uncertain_rows_are_excluded_from_training_and_selection(self) -> None:
        for k in range(5):
            splits = train.split_rows(train.RunSpec("E1", 0, fold=k), self.records, FOLDS)
            self.assertNotIn(UNCERTAIN_INDEX, splits["train"].tolist())
            self.assertNotIn(UNCERTAIN_INDEX, splits["val"].tolist())

    def test_uncertain_rows_are_kept_for_evaluation(self) -> None:
        fold = FOLDS[UNCERTAIN_INDEX]
        as_test = train.split_rows(train.RunSpec("E1", 0, fold=fold), self.records, FOLDS)
        self.assertIn(UNCERTAIN_INDEX, as_test["test"].tolist())
        as_val = train.split_rows(train.RunSpec("E1", 0, fold=(fold - 1) % 5), self.records, FOLDS)
        self.assertIn(UNCERTAIN_INDEX, as_val["val_scored"].tolist())

    def test_every_row_is_tested_exactly_once(self) -> None:
        tested = np.concatenate([train.split_rows(train.RunSpec("E1", 0, fold=k), self.records, FOLDS)["test"] for k in range(5)])
        self.assertEqual(sorted(tested.tolist()), list(range(len(self.records))))

    def test_cross_survey_removes_shared_sources_from_both_sides(self) -> None:
        splits = train.split_rows(train.RunSpec("L1", 0, source_survey="ao327", target_survey="gbncc"), self.records, FOLDS)
        for name in ("train", "val_scored", "test"):
            self.assertFalse({self.records[i]["source_id"] for i in splits[name]} & {"shared"})
        self.assertTrue(all(self.records[i]["survey"] == "gbncc" for i in splits["test"]))
        self.assertTrue(all(self.records[i]["survey"] == "ao327" for i in splits["train"]))

    def test_check_split_rejects_overlap_and_missing_classes(self) -> None:
        good = train.split_rows(train.RunSpec("E1", 0, fold=0), self.records, FOLDS)
        train.check_split(good, self.records)
        crossed = {**good, "test": np.r_[good["test"], good["train"][:1]]}
        with self.assertRaises(RuntimeError):
            train.check_split(crossed, self.records)
        missing = {**good, "train": np.array([i for i in good["train"] if self.records[i]["category"] != "noise"])}
        with self.assertRaises(RuntimeError):
            train.check_split(missing, self.records)


class ImageLoadingTests(unittest.TestCase):
    def test_real_inputs_match_the_shared_dataset(self) -> None:
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            rows, records = [], []
            for i in range(3):
                payload = {"FvP": rng.random((8, 20)).tolist(), "TvP": rng.random((6, 20)).tolist()}
                (cache / f"{i}.json").write_text(json.dumps(payload))
                rows.append({"plot_id": str(i), "source_category": "rfi"})
                records.append({"plot_id": str(i)})
            ours = train.load_images(records, cache, control=False)
            shared = PSCCachedDataset(rows, cache)
            self.assertTrue(torch.equal(ours, torch.stack(shared.images)))
            self.assertFalse(torch.equal(ours, train.load_images(records, cache, control=True)))


class TrainingPathTests(unittest.TestCase):
    """One or two epochs on noise, only to check initialization and bookkeeping."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.records = make_records()
        self.images = torch.rand(len(self.records), 2, 32, 32)
        names = ("known_pulsar", "rfi", "noise")
        self.labels = torch.tensor([names.index(r["category"]) for r in self.records])
        self.pretrained = BaselineCNN().state_dict()
        self.device = torch.device("cpu")

    def run_arm(self, arm, epochs=1):
        spec = train.RunSpec(arm, 0, fold=0)
        splits = train.split_rows(spec, self.records, FOLDS)
        state = self.pretrained if train.ARMS[arm]["init"] == "htru1" else None
        return spec, splits, train.train_run(spec, self.images, self.labels, splits, state, self.device, epochs=epochs)

    def test_frozen_arm_keeps_convolutions_fixed_and_trains_the_head(self) -> None:
        _, _, result = self.run_arm("E3")
        for key in ("features.0.weight", "features.3.weight"):
            self.assertTrue(torch.equal(result["state"][key], self.pretrained[key]))
        self.assertFalse(torch.equal(result["state"]["classifier.1.weight"], self.pretrained["classifier.1.weight"]))

    def test_fine_tuned_arm_updates_convolutions(self) -> None:
        _, _, result = self.run_arm("E1")
        self.assertFalse(torch.equal(result["state"]["features.0.weight"], self.pretrained["features.0.weight"]))

    def test_scratch_arm_ignores_the_checkpoint(self) -> None:
        _, _, result = self.run_arm("E2")
        self.assertEqual(result["load_info"]["loaded"], [])

    def test_htru1_arm_requires_the_checkpoint(self) -> None:
        with self.assertRaises(ValueError):
            train.build_model(train.RunSpec("E1", 0, fold=0), None)

    def test_same_seed_reproduces_on_cpu(self) -> None:
        _, _, first = self.run_arm("E2")
        _, _, second = self.run_arm("E2")
        self.assertTrue(torch.equal(first["state"]["classifier.4.weight"], second["state"]["classifier.4.weight"]))

    def test_run_output_is_complete_and_never_overwritten(self) -> None:
        spec, splits, result = self.run_arm("E2")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            run_dir = root / spec.run_id
            train.write_run(run_dir, root, spec, self.records, splits, result, {"identity_sha256": "x"})
            metadata = json.loads((run_dir / "metadata.json").read_text())
            self.assertEqual(metadata["run_id"], spec.run_id)
            self.assertEqual(metadata["recipe"]["learning_rate"], 1e-3)
            self.assertEqual(metadata["split_sizes"]["test"], len(splits["test"]))
            scored = (run_dir / "scores.csv").read_text().splitlines()
            self.assertEqual(len(scored) - 1, len(splits["val_scored"]) + len(splits["test"]))
            before = (run_dir / "best.pt").read_bytes()
            with self.assertRaises(FileExistsError):
                train.write_run(run_dir, root, spec, self.records, splits, result, {})
            self.assertEqual((run_dir / "best.pt").read_bytes(), before)
            with self.assertRaises(ValueError):
                train.write_run(Path(tmp) / "elsewhere" / "r", root, spec, self.records, splits, result, {})

    def test_wrong_checkpoint_hash_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "other.pt"
            torch.save({"model_state_dict": self.pretrained}, path)
            with self.assertRaises(ValueError):
                train.load_pretrained(path, self.device)


class BaselineRuleTests(unittest.TestCase):
    def test_units_count_once_then_classes_balance(self) -> None:
        records = [
            {"category": "known_pulsar", "source_id": "big", "survey": "a", "night": 1, "obs": "x"},
            {"category": "known_pulsar", "source_id": "big", "survey": "a", "night": 1, "obs": "x"},
            {"category": "known_pulsar", "source_id": "small", "survey": "a", "night": 1, "obs": "y"},
            {"category": "rfi", "source_id": None, "survey": "a", "night": 1, "obs": "z"},
            {"category": "noise", "source_id": None, "survey": "a", "night": 1, "obs": "w"},
        ]
        labels = np.array([0, 0, 0, 1, 2])
        weights = baselines.balanced_weights(np.arange(5), records, labels)
        self.assertAlmostEqual(weights[0], weights[1])
        self.assertAlmostEqual(weights[0] * 2, weights[2])
        self.assertAlmostEqual(weights[:3].sum(), weights[3])
        self.assertAlmostEqual(weights[3], weights[4])

    def test_forest_splits_exclude_uncertain_from_fitting_only(self) -> None:
        fold = np.array([0, 1, 2, 3, 4, 0])
        uncertain = np.array([False, False, False, False, False, True])
        fit, val, test = baselines.within_splits(fold, uncertain, 0)
        self.assertNotIn(5, fit.tolist())
        self.assertNotIn(5, val.tolist())
        self.assertIn(5, test.tolist())


def interval(low, high):
    return {"difference": (low + high) / 2, "low": low, "high": high}


class DecisionLogicTests(unittest.TestCase):
    def test_pilot_clear_signal_needs_the_allowance(self) -> None:
        self.assertEqual(evaluate.pilot_decision({"E1": interval(0.05, 0.2)})["outcome"], "clear_signal")
        self.assertEqual(evaluate.pilot_decision({"E1": interval(0.01, 0.2)})["outcome"], "inconclusive")

    def test_pilot_cannot_declare_failure(self) -> None:
        decision = evaluate.pilot_decision({"E1": interval(-0.1, 0.03), "E2": interval(-0.2, 0.0)})
        self.assertEqual(decision["outcome"], "provisional_negative")
        self.assertIn("extension", decision["action"])
        self.assertNotIn("failure", json.dumps(decision))

    def test_one_clear_arm_is_enough_to_proceed(self) -> None:
        decision = evaluate.pilot_decision({"E1": interval(0.05, 0.2), "E2": interval(-0.2, 0.0)})
        self.assertEqual(decision["outcome"], "clear_signal")

    def claim_a_inputs(self, control, duty, ao, drop=0.1, fpr_high=0.0):
        fundamental = interval(0.01, 0.1)
        return (
            {"E1": control},
            {"E1": duty},
            {"E1": ao},
            {"E1": fundamental},
            {"E1": fundamental},
            {"E1": drop},
            {"E1": interval(-0.2, fpr_high)},
        )

    def test_claim_a_outcomes(self) -> None:
        good = interval(0.02, 0.1)
        self.assertEqual(evaluate.claim_a(*self.claim_a_inputs(good, good, interval(-0.02, 0.01))), "success")
        self.assertEqual(evaluate.claim_a(*self.claim_a_inputs(interval(-0.05, 0.01), good, interval(-0.02, 0.01))), "failure")
        self.assertEqual(evaluate.claim_a(*self.claim_a_inputs(interval(-0.01, 0.1), good, interval(-0.02, 0.01))), "inconclusive")

    def test_claim_a_needs_duty_and_ao327_and_coherence(self) -> None:
        good = interval(0.02, 0.1)
        ao = interval(-0.02, 0.01)
        self.assertEqual(evaluate.claim_a(*self.claim_a_inputs(good, interval(-0.01, 0.1), ao)), "inconclusive")
        self.assertEqual(evaluate.claim_a(*self.claim_a_inputs(good, good, interval(-0.05, 0.0))), "inconclusive")
        self.assertEqual(evaluate.claim_a(*self.claim_a_inputs(good, good, ao, drop=-0.01)), "inconclusive")
        self.assertEqual(evaluate.claim_a(*self.claim_a_inputs(good, good, ao, fpr_high=0.2)), "inconclusive")

    def test_claim_b_outcomes(self) -> None:
        beats, matches, below = interval(0.01, 0.1), interval(-0.02, 0.05), interval(-0.2, -0.01)
        self.assertEqual(evaluate.claim_b({"gbncc": {"L1": beats}, "ao327": {"L1": matches}}), "success")
        self.assertEqual(evaluate.claim_b({"gbncc": {"L1": matches}, "ao327": {"L1": matches}}), "inconclusive")
        self.assertEqual(evaluate.claim_b({"gbncc": {"L1": below}, "ao327": {"L1": below}}), "failure")

    def test_claim_c_outcomes(self) -> None:
        wide_fpr = interval(-0.1, 0.1)
        self.assertEqual(evaluate.claim_c(interval(0.01, 0.05), wide_fpr, 0.8), "success")
        self.assertEqual(evaluate.claim_c(interval(-0.01, 0.005), wide_fpr, 0.8), "failure")
        self.assertEqual(evaluate.claim_c(interval(-0.1, 0.1), wide_fpr, 0.8), "inconclusive")
        self.assertIn("ceiling", evaluate.claim_c(interval(-0.001, 0.005), wide_fpr, 0.995))

    def test_frozen_bars_are_the_protocol_values(self) -> None:
        self.assertEqual(evaluate.FROZEN_BARS["GBNCC RFI"], (0.791, 0.858, 0.375))
        self.assertEqual(evaluate.PILOT_ALLOWANCE, 0.04)
        self.assertAlmostEqual(evaluate.FINAL_LEVEL, 0.98333, places=4)


class ProtectedHashTests(unittest.TestCase):
    def test_compare_reports_changes_and_missing_files(self) -> None:
        recorded = {"a": "1", "b": "2", "c": "3"}
        self.assertEqual(verify.compare(recorded, {"a": "1", "b": "9"}), ["changed: b", "missing: c"])
        self.assertEqual(verify.compare(recorded, recorded), [])

    def test_directory_digest_sees_edits_and_renames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "x").write_text("one")
            first = verify.digest_directory(root)
            (root / "x").write_text("two")
            second = verify.digest_directory(root)
            (root / "x").rename(root / "y")
            third = verify.digest_directory(root)
            self.assertEqual(len({first, second, third}), 3)


if __name__ == "__main__":
    unittest.main()
