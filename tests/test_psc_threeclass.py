"""Tests for the three-class head, cached PSC dataset, and source-aware splitting."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.datasets.psc_dataset import (
    PSCCachedDataset,
    class_weights,
    split_indices_by_source_group,
)
from src.models.baseline_cnn import BaselineCNN
from src.models.three_class_head import (
    CLASS_NAMES,
    NUM_CLASSES,
    PULSAR_INDEX,
    ThreeClassCNN,
    collapse_to_pulsar_score,
)


def fake_payload(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "FvP": rng.random((64, 200)).tolist(),
        "TvP": rng.random((24, 200)).tolist(),
        "profile": rng.random(200).tolist(),
    }


class TestThreeClassModel(unittest.TestCase):
    def test_class_order_is_pinned(self):
        # Binary collapse indexes the pulsar class directly; reordering silently
        # inverts every comparison against EXP-001..003.
        self.assertEqual(CLASS_NAMES[PULSAR_INDEX], "known_pulsar")
        self.assertEqual(NUM_CLASSES, 3)

    def test_forward_shape_and_input_validation(self):
        model = ThreeClassCNN()
        out = model(torch.randn(4, 2, 32, 32))
        self.assertEqual(tuple(out.shape), (4, 3))
        with self.assertRaises(ValueError):
            model(torch.randn(4, 3, 32, 32))

    def test_loads_pretrained_backbone_and_skips_resized_head(self):
        base = BaselineCNN()
        model = ThreeClassCNN()
        info = model.load_pretrained_backbone(base.state_dict())

        # Every conv layer transfers; the 1-output layer cannot.
        self.assertTrue(any(k.startswith("features.") for k in info["loaded"]))
        self.assertIn("classifier.4.weight", info["skipped"])
        for key in base.state_dict():
            if key.startswith("features."):
                self.assertTrue(
                    torch.allclose(base.state_dict()[key], model.state_dict()[key]),
                    f"{key} did not transfer",
                )

    def test_rejects_backbone_that_does_not_cover_features(self):
        model = ThreeClassCNN()
        with self.assertRaises(RuntimeError):
            model.load_pretrained_backbone({"classifier.1.weight": torch.zeros(128, 2048)})

    def test_freeze_backbone_stops_gradients(self):
        model = ThreeClassCNN()
        model.freeze_backbone()
        self.assertTrue(all(not p.requires_grad for p in model.features.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.classifier.parameters()))
        model.unfreeze_backbone()
        self.assertTrue(all(p.requires_grad for p in model.features.parameters()))

    def test_collapse_to_pulsar_score(self):
        logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
        scores = collapse_to_pulsar_score(logits)
        self.assertGreater(float(scores[0]), 0.9)
        self.assertLess(float(scores[1]), 0.1)
        with self.assertRaises(ValueError):
            collapse_to_pulsar_score(torch.randn(4, 2))


class TestSourceAwareSplit(unittest.TestCase):
    def test_no_source_group_straddles_splits(self):
        groups = [f"g{i//4:03d}" for i in range(200)]
        tr, va, te = split_indices_by_source_group(groups, seed=1)
        gt = {groups[i] for i in tr}
        gv = {groups[i] for i in va}
        ge = {groups[i] for i in te}
        self.assertEqual(gt & gv, set())
        self.assertEqual(gt & ge, set())
        self.assertEqual(gv & ge, set())

    def test_every_sample_lands_somewhere_exactly_once(self):
        groups = [f"g{i//3:03d}" for i in range(150)]
        tr, va, te = split_indices_by_source_group(groups, seed=7)
        self.assertEqual(sorted(tr + va + te), list(range(150)))

    def test_repeat_detections_stay_together(self):
        # One pulsar with 17 detections must land wholly on one side.
        groups = ["busy"] * 17 + [f"g{i}" for i in range(40)]
        tr, va, te = split_indices_by_source_group(groups, seed=3)
        busy = [i for i, g in enumerate(groups) if g == "busy"]
        for bucket in (tr, va, te):
            inside = [i for i in busy if i in bucket]
            self.assertIn(len(inside), (0, len(busy)))

    def test_seed_is_deterministic(self):
        groups = [f"g{i//2:03d}" for i in range(80)]
        self.assertEqual(
            split_indices_by_source_group(groups, seed=5),
            split_indices_by_source_group(groups, seed=5),
        )

    def test_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            split_indices_by_source_group([])
        with self.assertRaises(ValueError):
            split_indices_by_source_group(["a", "b"], fractions=(0.5, 0.4, 0.4))


class TestClassWeights(unittest.TestCase):
    def test_rarer_class_gets_more_weight(self):
        w = class_weights([0] * 90 + [1] * 9 + [2] * 1, 3)
        self.assertGreater(float(w[2]), float(w[1]))
        self.assertGreater(float(w[1]), float(w[0]))

    def test_missing_class_raises(self):
        with self.assertRaises(ValueError):
            class_weights([0] * 10 + [1] * 10, 3)


class TestCachedDataset(unittest.TestCase):
    def _make(self, tmp: str, specs):
        cache = Path(tmp)
        rows = []
        for i, (plot_id, category) in enumerate(specs):
            (cache / f"{plot_id}.json").write_text(json.dumps(fake_payload(i)))
            rows.append(
                {"plot_id": plot_id, "source_category": category, "source_group": f"g{i}"}
            )
        return rows, cache

    def test_loads_and_labels_correctly(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows, cache = self._make(
                tmp, [("1", "known_pulsar"), ("2", "rfi"), ("3", "noise")]
            )
            ds = PSCCachedDataset(rows, cache)
            self.assertEqual(len(ds), 3)
            image, label, plot_id = ds[0]
            self.assertEqual(tuple(image.shape), (2, 32, 32))
            self.assertEqual(int(label), PULSAR_INDEX)
            self.assertEqual(plot_id, "1")
            self.assertEqual(ds.class_counts(), {0: 1, 1: 1, 2: 1})

    def test_missing_cache_entry_is_skipped_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows, cache = self._make(tmp, [("1", "known_pulsar")])
            rows.append(
                {"plot_id": "999", "source_category": "rfi", "source_group": "gx"}
            )
            ds = PSCCachedDataset(rows, cache)
            self.assertEqual(len(ds), 1)
            self.assertEqual(len(ds.skipped), 1)
            self.assertIn("not cached", ds.skipped[0][1])

    def test_corrupt_cache_entry_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows, cache = self._make(tmp, [("1", "known_pulsar")])
            (cache / "2.json").write_text("{not json")
            rows.append({"plot_id": "2", "source_category": "rfi", "source_group": "g9"})
            ds = PSCCachedDataset(rows, cache)
            self.assertEqual(len(ds), 1)
            self.assertEqual(len(ds.skipped), 1)

    def test_unknown_category_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows, cache = self._make(tmp, [("1", "maybe")])
            ds = PSCCachedDataset(rows, cache)
            self.assertEqual(len(ds), 0)
            self.assertIn("unknown category", ds.skipped[0][1])

    def test_matches_the_evaluation_preprocessing_exactly(self):
        # Training must see the same tensors the ranking pipeline scores.
        from src.datasets.psc_preprocessing import psc_json_to_tensor

        with tempfile.TemporaryDirectory() as tmp:
            rows, cache = self._make(tmp, [("1", "known_pulsar")])
            payload = json.loads((cache / "1.json").read_text())
            expected, _ = psc_json_to_tensor(payload, plot_id="1")
            ds = PSCCachedDataset(rows, cache)
            self.assertTrue(torch.allclose(ds[0][0], expected))


if __name__ == "__main__":
    unittest.main()
