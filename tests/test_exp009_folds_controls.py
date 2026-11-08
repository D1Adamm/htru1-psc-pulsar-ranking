"""Tests for EXP-009 fold building, safe writes, and control transforms."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.datasets.psc_controls import (
    INTERVENTIONS,
    candidate_rng,
    fill_blank_rows,
    rebin_phase,
    roll_phase,
    row_shuffled_tensor,
    shuffle_whole_array,
    shuffle_within_rows,
)
from src.datasets.psc_folds import (
    assert_under,
    assign_folds,
    leak_report,
    linked_components,
    rotation,
    shared_sources,
    write_exclusive,
)


def record(plot_id, category, obs, night, group=None, survey="ao327", source=None):
    return {
        "plot_id": plot_id,
        "category": category,
        "obs": obs,
        "night": night,
        "survey": survey,
        "fold_group_id": group,
        "source_id": source or group,
    }


def synthetic_records():
    records = []
    for i in range(30):
        records.append(record(f"p{i}", "known_pulsar", f"pobs{i // 2}", 100 + i, group=f"g{i // 3}"))
    for i in range(30):
        records.append(record(f"r{i}", "rfi", f"robs{i}", 200 + i // 4))
    for i in range(30):
        records.append(record(f"n{i}", "noise", f"nobs{i // 5}", 300))
    return records


class FoldTests(unittest.TestCase):
    def test_links_join_group_observation_and_rfi_night(self) -> None:
        records = [
            record("a", "known_pulsar", "o1", 1, group="g1"),
            record("b", "known_pulsar", "o2", 2, group="g1"),
            record("c", "rfi", "o3", 5),
            record("d", "rfi", "o4", 5),
            record("e", "noise", "o1", 9),
            record("f", "noise", "o5", 9),
        ]
        components = linked_components(records)
        self.assertEqual(components[0], components[1])
        self.assertEqual(components[2], components[3])
        self.assertEqual(components[0], components[4], "shared observation must link classes")
        self.assertNotEqual(components[4], components[5], "noise from one night may be split by observation")

    def test_folds_are_deterministic_and_leak_free(self) -> None:
        records = synthetic_records()
        components = linked_components(records)
        first = assign_folds(records, components, seed=3)
        self.assertEqual(first, assign_folds(records, components, seed=3))
        self.assertEqual(set(leak_report(records, first).values()), {0})
        self.assertEqual(set(first), {0, 1, 2, 3, 4})

    def test_leak_report_detects_a_split_source(self) -> None:
        records = [record("a", "known_pulsar", "o1", 1, group="g"), record("b", "known_pulsar", "o2", 2, group="g")]
        self.assertEqual(leak_report(records, [0, 1])["source_id"], 1)

    def test_rotation_is_test_validation_then_train(self) -> None:
        self.assertEqual(rotation(4), {"test": [4], "val": [0], "train": [1, 2, 3]})

    def test_shared_sources(self) -> None:
        records = [
            record("a", "known_pulsar", "o1", 1, group="g", survey="ao327"),
            record("b", "known_pulsar", "o2", 2, group="g", survey="gbncc"),
            record("c", "known_pulsar", "o3", 3, group="h", survey="gbncc"),
        ]
        self.assertEqual(shared_sources(records), {"g"})


class SafeWriteTests(unittest.TestCase):
    def test_write_exclusive_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sub" / "x.txt"
            write_exclusive(path, "one")
            with self.assertRaises(FileExistsError):
                write_exclusive(path, "two")
            self.assertEqual(path.read_text(), "one")

    def test_assert_under_refuses_outside_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            assert_under(root / "a" / "b", root)
            with self.assertRaises(ValueError):
                assert_under(root / ".." / "elsewhere", root)


class ControlTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        self.fvp = rng.random((16, 50))
        self.tvp = rng.random((12, 50))
        self.tvp[3] = 0.0

    def test_row_shuffle_is_deterministic_per_candidate(self) -> None:
        a = row_shuffled_tensor(self.fvp, self.tvp, "123")
        b = row_shuffled_tensor(self.fvp, self.tvp, "123")
        c = row_shuffled_tensor(self.fvp, self.tvp, "124")
        self.assertTrue((a == b).all())
        self.assertFalse((a == c).all())

    def test_row_shuffle_keeps_row_values_and_blank_rows(self) -> None:
        out = shuffle_within_rows(self.tvp, candidate_rng("x"))
        np.testing.assert_array_equal(np.sort(out, axis=1), np.sort(self.tvp, axis=1))
        self.assertTrue((out[3] == 0).all())

    def test_row_shuffle_destroys_a_vertical_stripe(self) -> None:
        stripe = np.zeros((32, 100))
        stripe[:, 40:44] = 1.0
        shuffled = shuffle_within_rows(stripe, candidate_rng("x"))
        self.assertAlmostEqual(stripe.mean(0).max(), 1.0)
        self.assertLess(shuffled.mean(0).max(), 0.5)
        np.testing.assert_array_equal(shuffled.sum(1), stripe.sum(1))

    def test_whole_shuffle_keeps_pixels_but_not_blank_rows(self) -> None:
        out = shuffle_whole_array(self.tvp, candidate_rng("x"))
        np.testing.assert_array_equal(np.sort(out.ravel()), np.sort(self.tvp.ravel()))
        self.assertFalse((out[3] == 0).all())

    def test_interventions_keep_the_cnn_shape_and_do_not_mutate_inputs(self) -> None:
        fvp, tvp = self.fvp.copy(), self.tvp.copy()
        for name, transform in INTERVENTIONS.items():
            self.assertEqual(tuple(transform(fvp, tvp, "1").shape), (2, 32, 32), name)
        np.testing.assert_array_equal(fvp, self.fvp)
        np.testing.assert_array_equal(tvp, self.tvp)

    def test_channel_removal_targets_one_channel(self) -> None:
        self.assertTrue((INTERVENTIONS["fvp_removed"](self.fvp, self.tvp, "1")[0] == 0).all())
        self.assertTrue((INTERVENTIONS["tvp_removed"](self.fvp, self.tvp, "1")[1] == 0).all())

    def test_fill_blank_rows_uses_nearest_valid_row(self) -> None:
        filled = fill_blank_rows(self.tvp)
        np.testing.assert_array_equal(filled[3], self.tvp[2])
        np.testing.assert_array_equal(fill_blank_rows(np.zeros((3, 4))), np.zeros((3, 4)))

    def test_roll_and_rebin(self) -> None:
        array = np.arange(8.0).reshape(1, 8)
        np.testing.assert_array_equal(roll_phase(array, 1)[0], [7, 0, 1, 2, 3, 4, 5, 6])
        np.testing.assert_array_equal(rebin_phase(array, 4)[0], [1.5, 5.5])


if __name__ == "__main__":
    unittest.main()
