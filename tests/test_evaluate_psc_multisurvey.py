"""Tests for survey-aware evaluation of a saved three-class checkpoint."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "evaluate_psc_multisurvey.py"


def _load_script():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location("evaluate_psc_multisurvey", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


evaluate_psc_multisurvey = _load_script()

from src.datasets.psc_dataset import PSCCachedDataset  # noqa: E402
from src.psc_metrics import (  # noqa: E402
    binary_collapsed_report,
    class_report,
    duty_cycle,
    duty_cycle_roc_auc,
    group_indices_by_survey,
)


def one_hot_logits(predictions: list[int], confidence: float = 8.0) -> torch.Tensor:
    logits = torch.zeros(len(predictions), 3)
    for row, cls in enumerate(predictions):
        logits[row, cls] = confidence
    return logits


class TestClassReport(unittest.TestCase):
    def test_perfect_predictions(self):
        labels = torch.tensor([0, 1, 2, 0])
        report = class_report(one_hot_logits([0, 1, 2, 0]), labels)
        self.assertEqual(report["accuracy"], 1.0)
        self.assertEqual(report["n"], 4)
        for cls in report["class_order"]:
            self.assertEqual(report["per_class"][cls]["f1"], 1.0)
        self.assertEqual(report["per_class"]["known_pulsar"]["support"], 2)

    def test_confusion_matrix_orientation_is_true_by_predicted(self):
        # Two true pulsars, both predicted rfi. That must land at [0][1].
        report = class_report(one_hot_logits([1, 1]), torch.tensor([0, 0]))
        self.assertEqual(report["confusion_matrix"][0][1], 2)
        self.assertEqual(report["confusion_matrix"][1][0], 0)

    def test_f1_is_the_harmonic_mean_of_precision_and_recall(self):
        # 2 of 3 true pulsars found, 1 false positive -> P=2/3, R=2/3, F1=2/3.
        labels = torch.tensor([0, 0, 0, 1])
        report = class_report(one_hot_logits([0, 0, 1, 0]), labels)
        pulsar = report["per_class"]["known_pulsar"]
        self.assertAlmostEqual(pulsar["precision"], 2 / 3)
        self.assertAlmostEqual(pulsar["recall"], 2 / 3)
        self.assertAlmostEqual(pulsar["f1"], 2 / 3)

    def test_undefined_metrics_are_none_not_nan(self):
        # A class that is never predicted and never true has no precision,
        # recall, or F1. NaN would make the JSON non-standard and would read as
        # a score rather than as an absence.
        report = class_report(one_hot_logits([0, 0]), torch.tensor([0, 0]))
        noise = report["per_class"]["noise"]
        self.assertIsNone(noise["precision"])
        self.assertIsNone(noise["recall"])
        self.assertIsNone(noise["f1"])
        self.assertEqual(noise["support"], 0)
        json.dumps(report, allow_nan=False)  # must not raise

    def test_zero_f1_when_precision_and_recall_are_both_zero(self):
        report = class_report(one_hot_logits([1, 0]), torch.tensor([0, 1]))
        self.assertEqual(report["per_class"]["known_pulsar"]["f1"], 0.0)

    def test_rejects_mismatched_shapes(self):
        with self.assertRaises(ValueError):
            class_report(torch.randn(4, 2), torch.tensor([0, 1, 2, 0]))
        with self.assertRaises(ValueError):
            class_report(torch.randn(4, 3), torch.tensor([0, 1]))


class TestBinaryCollapsedReport(unittest.TestCase):
    def test_separable_scores_give_auc_one(self):
        logits = one_hot_logits([0, 0, 1, 2])
        report = binary_collapsed_report(logits, torch.tensor([0, 0, 1, 2]))
        self.assertEqual(report["n"], 4)
        self.assertEqual(report["n_pulsar"], 2)
        self.assertAlmostEqual(report["roc_auc"], 1.0)
        self.assertAlmostEqual(report["pr_auc"], 1.0)

    def test_single_class_subset_reports_none_not_a_crash(self):
        # A per-survey subset can easily contain no pulsars at all; AUC is
        # undefined there and must be reported as absent, not as 0.5.
        report = binary_collapsed_report(one_hot_logits([1, 2]), torch.tensor([1, 2]))
        self.assertIsNone(report["roc_auc"])
        self.assertIsNone(report["pr_auc"])
        self.assertEqual(report["n_pulsar"], 0)


class TestDutyCycle(unittest.TestCase):
    def test_narrow_profile_scores_lower_than_broad(self):
        narrow = torch.zeros(2, 32, 32)
        narrow[0, :, 16] = 1.0
        broad = torch.ones(2, 32, 32)
        broad[0, :, :16] = 0.0
        self.assertLess(duty_cycle(narrow), duty_cycle(broad))

    def test_flat_profile_is_maximally_broad(self):
        self.assertEqual(duty_cycle(torch.ones(2, 32, 32)), 1.0)

    def test_roc_auc_prefers_narrow_pulsars(self):
        narrow = torch.zeros(2, 32, 32)
        narrow[0, :, 16] = 1.0
        broad = torch.ones(2, 32, 32)
        broad[0, :, :16] = 0.0
        auc = duty_cycle_roc_auc([narrow, narrow, broad, broad], torch.tensor([0, 0, 1, 2]))
        self.assertAlmostEqual(auc, 1.0)

    def test_single_class_subset_returns_none(self):
        self.assertIsNone(duty_cycle_roc_auc([torch.ones(2, 32, 32)], torch.tensor([1])))


class TestGroupIndicesBySurvey(unittest.TestCase):
    def test_partitions_indices_and_sorts_survey_names(self):
        surveys = ["gbncc", "ao327", "gbncc", "ao327"]
        grouped = group_indices_by_survey(surveys, [0, 1, 2, 3])
        self.assertEqual(list(grouped), ["ao327", "gbncc"])
        self.assertEqual(grouped["ao327"], [1, 3])
        self.assertEqual(grouped["gbncc"], [0, 2])

    def test_only_the_given_indices_are_grouped(self):
        # Callers pass a split, not the whole dataset; nothing outside it may
        # leak into a per-survey number.
        surveys = ["ao327", "gbncc", "ao327", "gbncc"]
        grouped = group_indices_by_survey(surveys, [1, 2])
        self.assertEqual(grouped, {"ao327": [2], "gbncc": [1]})


class TestEvaluateBySurvey(unittest.TestCase):
    def _fixture(self):
        # ao327 perfect on 3; gbncc gets one of 3 wrong.
        labels = torch.tensor([0, 1, 2, 0, 1, 2])
        logits = one_hot_logits([0, 1, 2, 0, 1, 0])
        surveys = ["ao327", "ao327", "ao327", "gbncc", "gbncc", "gbncc"]
        images = [torch.rand(2, 32, 32) for _ in range(6)]
        return logits, labels, surveys, images

    def test_splits_metrics_by_survey(self):
        result = evaluate_psc_multisurvey.evaluate_by_survey(*self._fixture())
        self.assertEqual(sorted(result["by_survey"]), ["ao327", "gbncc"])
        self.assertEqual(result["by_survey"]["ao327"]["three_class"]["accuracy"], 1.0)
        self.assertAlmostEqual(result["by_survey"]["gbncc"]["three_class"]["accuracy"], 2 / 3)
        self.assertAlmostEqual(result["combined"]["three_class"]["accuracy"], 5 / 6)

    def test_subset_sizes_sum_to_the_combined_total(self):
        result = evaluate_psc_multisurvey.evaluate_by_survey(*self._fixture())
        per_survey = sum(b["three_class"]["n"] for b in result["by_survey"].values())
        self.assertEqual(per_survey, result["combined"]["three_class"]["n"])

    def test_combined_accuracy_is_the_support_weighted_mean_of_the_subsets(self):
        # The property that makes a per-survey breakdown checkable against a
        # previously recorded combined number.
        result = evaluate_psc_multisurvey.evaluate_by_survey(*self._fixture())
        correct = sum(
            b["three_class"]["accuracy"] * b["three_class"]["n"]
            for b in result["by_survey"].values()
        )
        combined = result["combined"]["three_class"]
        self.assertAlmostEqual(correct / combined["n"], combined["accuracy"])

    def test_per_survey_confusion_matrices_sum_to_the_combined_one(self):
        result = evaluate_psc_multisurvey.evaluate_by_survey(*self._fixture())
        total = np.zeros((3, 3), dtype=int)
        for block in result["by_survey"].values():
            total += np.array(block["three_class"]["confusion_matrix"])
        np.testing.assert_array_equal(
            total, np.array(result["combined"]["three_class"]["confusion_matrix"])
        )

    def test_single_survey_subset_equals_combined(self):
        logits, labels, _surveys, images = self._fixture()
        result = evaluate_psc_multisurvey.evaluate_by_survey(
            logits, labels, ["ao327"] * 6, images
        )
        self.assertEqual(list(result["by_survey"]), ["ao327"])
        self.assertEqual(
            result["by_survey"]["ao327"]["three_class"], result["combined"]["three_class"]
        )

    def test_rejects_non_parallel_inputs(self):
        logits, labels, surveys, images = self._fixture()
        with self.assertRaises(ValueError):
            evaluate_psc_multisurvey.evaluate_by_survey(logits, labels, surveys[:-1], images)

    def test_result_is_strict_json(self):
        result = evaluate_psc_multisurvey.evaluate_by_survey(*self._fixture())
        json.dumps(result, allow_nan=False)

    def test_format_report_mentions_every_block(self):
        result = evaluate_psc_multisurvey.evaluate_by_survey(*self._fixture())
        text = evaluate_psc_multisurvey.format_report(result)
        for name in ("combined", "ao327", "gbncc", "known_pulsar", "duty-cycle"):
            self.assertIn(name, text)


class TestScoreIndices(unittest.TestCase):
    def _dataset(self, tmp: Path, surveys: list[str]) -> PSCCachedDataset:
        rng = np.random.default_rng(0)
        rows = []
        categories = ["known_pulsar", "rfi", "noise"]
        for i, survey in enumerate(surveys):
            (tmp / f"{i}.json").write_text(
                json.dumps(
                    {
                        "FvP": rng.random((64, 200)).tolist(),
                        "TvP": rng.random((24, 200)).tolist(),
                    }
                )
            )
            rows.append(
                {
                    "plot_id": str(i),
                    "source_category": categories[i % 3],
                    "source_group": f"g{i}",
                    "survey": survey,
                }
            )
        return PSCCachedDataset(rows, tmp)

    def test_scores_are_returned_in_the_order_of_the_given_indices(self):
        # The survey list is built by indexing the dataset in the same order,
        # so any reordering here silently mislabels every per-survey number.
        from src.models.three_class_head import ThreeClassCNN

        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._dataset(Path(tmp), ["ao327", "gbncc"] * 3)
            model = ThreeClassCNN()
            indices = [4, 1, 3]
            logits, labels = evaluate_psc_multisurvey.score_indices(
                model, dataset, indices, torch.device("cpu"), batch_size=2
            )
            self.assertEqual(len(logits), 3)
            self.assertEqual(labels.tolist(), [dataset.labels[i] for i in indices])
            with torch.no_grad():
                expected = model(torch.stack([dataset.images[i] for i in indices]))
            self.assertTrue(torch.allclose(logits, expected, atol=1e-5))


class TestDatasetCarriesSurvey(unittest.TestCase):
    def test_survey_is_recorded_per_usable_item_not_per_manifest_row(self):
        rng = np.random.default_rng(1)
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            (cache / "1.json").write_text(
                json.dumps(
                    {"FvP": rng.random((64, 200)).tolist(), "TvP": rng.random((24, 200)).tolist()}
                )
            )
            (cache / "3.json").write_text(
                json.dumps(
                    {"FvP": rng.random((64, 200)).tolist(), "TvP": rng.random((24, 200)).tolist()}
                )
            )
            rows = [
                {"plot_id": "1", "source_category": "known_pulsar", "source_group": "a", "survey": "ao327"},
                # Uncached: must not shift the surveys of the rows after it.
                {"plot_id": "2", "source_category": "rfi", "source_group": "b", "survey": "ao327"},
                {"plot_id": "3", "source_category": "noise", "source_group": "c", "survey": "gbncc"},
            ]
            dataset = PSCCachedDataset(rows, cache)
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset.surveys, ["ao327", "gbncc"])

    def test_missing_survey_column_becomes_unknown(self):
        rng = np.random.default_rng(2)
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            (cache / "1.json").write_text(
                json.dumps(
                    {"FvP": rng.random((64, 200)).tolist(), "TvP": rng.random((24, 200)).tolist()}
                )
            )
            dataset = PSCCachedDataset(
                [{"plot_id": "1", "source_category": "rfi", "source_group": "a"}], cache
            )
            self.assertEqual(dataset.surveys, ["unknown"])


class TestSplitVerification(unittest.TestCase):
    def test_matching_split_passes(self):
        self.assertTrue(
            evaluate_psc_multisurvey.verify_split_matches_checkpoint([1, 2, 3], [1, 2, 3])
        )

    def test_absent_record_is_unverified_not_a_pass(self):
        self.assertIsNone(
            evaluate_psc_multisurvey.verify_split_matches_checkpoint([1, 2, 3], None)
        )

    def test_reordered_or_changed_split_is_fatal(self):
        # Same candidates in a different order still means the split logic moved.
        for recorded in ([3, 2, 1], [1, 2], [1, 2, 4]):
            with self.subTest(recorded=recorded):
                with self.assertRaises(RuntimeError):
                    evaluate_psc_multisurvey.verify_split_matches_checkpoint(
                        [1, 2, 3], recorded
                    )


class TestArgumentGuards(unittest.TestCase):
    def test_defaults_point_at_the_multisurvey_artifacts(self):
        args = evaluate_psc_multisurvey.parse_args([])
        self.assertEqual(args.checkpoint.name, "psc_multisurvey_best.pt")
        self.assertEqual(args.manifest.name, "psc_eval_multisurvey.csv")
        self.assertEqual(args.split, "test")
        self.assertEqual(args.seed, 42)

    def test_refuses_the_binary_htru1_baseline_checkpoint(self):
        with self.assertRaises(ValueError):
            evaluate_psc_multisurvey.main(
                ["--checkpoint", str(evaluate_psc_multisurvey.BASELINE_CHECKPOINT)]
            )


if __name__ == "__main__":
    unittest.main()
