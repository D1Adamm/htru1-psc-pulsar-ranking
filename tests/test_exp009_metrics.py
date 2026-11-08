"""Tests for EXP-009 source-level metrics and paired resampling."""

from __future__ import annotations

import unittest

import numpy as np

from src.psc_source_metrics import (
    Population,
    build_population,
    combined_interval,
    component_bootstrap,
    fpr_at_source_recall,
    linked_units,
    one_detection_random_auc,
    one_detection_strongest_auc,
    pair_auc,
    paired_bootstrap,
    paired_difference,
    profile_duty_cycle,
    resample_population,
    typical_detection_auc,
)


def rec(source=None, category="known_pulsar", survey="ao327", night=1, obs="o"):
    return {"source_id": source, "category": category, "survey": survey, "night": night, "obs": obs}


class MetricDefinitionTests(unittest.TestCase):
    def test_pair_auc_counts_ties_as_half(self) -> None:
        self.assertEqual(pair_auc(np.array([2.0]), np.array([1.0])), 1.0)
        self.assertEqual(pair_auc(np.array([1.0]), np.array([1.0])), 0.5)

    def test_typical_auc_uses_each_sources_median(self) -> None:
        scores = np.array([0.9, 0.1, 0.1, 0.5])
        sources = [np.array([0, 1, 2])]
        self.assertEqual(typical_detection_auc(scores, sources, np.array([3])), 0.0)

    def test_repeats_do_not_outweigh_a_source(self) -> None:
        scores = np.r_[np.full(50, 0.9), [0.1], [0.5]]
        sources = [np.arange(50), np.array([50])]
        self.assertEqual(typical_detection_auc(scores, sources, np.array([51])), 0.5)

    def test_fpr_uses_each_sources_best_row(self) -> None:
        scores = np.array([0.1, 0.9, 0.8, 0.5, 0.2])
        sources = [np.array([0, 1]), np.array([2])]
        negatives = np.array([3, 4])
        rng = np.random.default_rng(0)
        self.assertEqual(fpr_at_source_recall(scores, sources, negatives, rng, recall=1.0, tie_draws=5), 0.0)

    def test_ties_are_broken_fairly_for_discrete_statistics(self) -> None:
        scores = np.full(200, 0.5)
        sources = [np.array([i]) for i in range(100)]
        negatives = np.arange(100, 200)
        fpr = fpr_at_source_recall(scores, sources, negatives, np.random.default_rng(0), tie_draws=200)
        self.assertAlmostEqual(fpr, 0.9, delta=0.03)

    def test_one_detection_strongest_ignores_the_score(self) -> None:
        scores = np.array([0.1, 0.9, 0.5])
        sigma = np.array([100.0, 1.0, 0.0])
        self.assertEqual(one_detection_strongest_auc(scores, [np.array([0, 1])], np.array([2]), sigma), 0.0)

    def test_one_detection_random_averages_over_rows(self) -> None:
        scores = np.array([1.0, 0.0, 0.5])
        value = one_detection_random_auc(scores, [np.array([0, 1])], np.array([2]), np.random.default_rng(0), draws=500)
        self.assertAlmostEqual(value, 0.5, delta=0.08)

    def test_profile_duty_cycle(self) -> None:
        self.assertEqual(profile_duty_cycle(np.array([0, 0, 1, 1.0])), 0.5)
        self.assertEqual(profile_duty_cycle(np.ones(4)), 1.0)


class PopulationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            rec("s1", survey="ao327"),
            rec("s1", survey="gbncc"),
            rec("s2"),
            rec(None, "rfi", night=1, obs="a"),
            rec(None, "rfi", night=1, obs="b"),
            rec(None, "rfi", night=2, obs="c"),
            rec(None, "noise", night=9, obs="n1"),
            rec(None, "noise", night=9, obs="n2"),
        ]
        category = np.array([r["category"] for r in self.records])
        self.population = build_population(self.records, category == "known_pulsar", category != "known_pulsar")

    def test_shared_source_is_one_unit_across_surveys(self) -> None:
        self.assertEqual(len(self.population.sources), 2)
        self.assertEqual(sorted(len(s) for s in self.population.sources), [1, 2])

    def test_negative_units_are_rfi_nights_and_noise_observations(self) -> None:
        self.assertEqual(len(self.population.negatives), 4)
        self.assertEqual(sorted(len(n) for n in self.population.negatives), [1, 1, 1, 2])

    def test_resampling_draws_whole_units(self) -> None:
        for seed in range(20):
            sample = resample_population(self.population, np.random.default_rng(seed))
            self.assertEqual(len(sample.sources), 2)
            for rows in sample.sources:
                self.assertIn(rows.tolist(), [[0, 1], [2]])

    def test_linked_units_keep_a_block_together(self) -> None:
        blocks = linked_units(self.population, [0, 0, 1, 1, 1, 2, 3, 4])
        self.assertEqual(len(blocks), 5)
        sized = sorted((len(b.sources), len(b.negatives)) for b in blocks)
        self.assertIn((1, 1), sized)


class PairedStatisticsTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(1)
        self.population = Population(
            sources=[np.array([i, i + 1]) for i in range(0, 40, 2)],
            negatives=[np.array([i]) for i in range(40, 80)],
        )
        labels = np.r_[np.ones(40), np.zeros(40)]
        self.good = labels + rng.normal(0, 0.5, 80)
        self.poor = labels + rng.normal(0, 3.0, 80)

    def test_same_draws_give_zero_difference_for_a_model_against_itself(self) -> None:
        draws = paired_bootstrap({"a": self.good[None], "b": self.good[None]}, self.population, n_boot=25, seed=3, tie_draws=1)
        np.testing.assert_array_equal(draws["a"][:, 0], draws["b"][:, 0])

    def test_a_better_model_shows_a_positive_interval(self) -> None:
        draws = paired_bootstrap({"good": self.good[None], "poor": self.poor[None]}, self.population, n_boot=120, seed=3, tie_draws=1)
        a = typical_detection_auc(self.good, self.population.sources, self.population.negative_rows)
        b = typical_detection_auc(self.poor, self.population.sources, self.population.negative_rows)
        result = paired_difference(draws["good"][:, 0], draws["poor"][:, 0], [a], [b])
        self.assertGreater(result["low"], 0)
        self.assertAlmostEqual(result["difference"], a - b)

    def test_between_assignment_variance_widens_the_interval(self) -> None:
        draws = np.random.default_rng(0).normal(0.8, 0.02, 1000)
        narrow = combined_interval(0.8, draws, [0.8])
        same = combined_interval(0.8, draws, [0.8, 0.8, 0.8])
        wide = combined_interval(0.8, draws, [0.75, 0.8, 0.85])
        self.assertAlmostEqual(narrow[1] - narrow[0], same[1] - same[0], places=6)
        self.assertGreater(wide[1] - wide[0], narrow[1] - narrow[0])

    def test_component_bootstrap_runs_on_blocks(self) -> None:
        blocks = [Population([self.population.sources[i]], [self.population.negatives[i]]) for i in range(20)]
        draws = component_bootstrap({"a": self.good[None]}, blocks, n_boot=10, seed=0, tie_draws=1)
        self.assertEqual(draws["a"].shape, (10, 2))


if __name__ == "__main__":
    unittest.main()
