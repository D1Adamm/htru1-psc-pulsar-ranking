#!/usr/bin/env python3
"""Evaluate a saved multi-survey three-class checkpoint, broken down by survey.

EXP-008's record reports held-out accuracy separately for AO327 and GBNCC, but
the committed metrics file only ever held the combined numbers and the per-survey
figures were computed by hand. This script closes that gap: it rebuilds the
identical deterministic ``source_group`` split, scores an existing checkpoint,
and writes combined *and* per-survey metrics to JSON.

It trains nothing and touches no checkpoint. Reproducing EXP-008:

    python scripts/evaluate_psc_multisurvey.py

The split is reconstructed, not read from the checkpoint. When the checkpoint
carries a ``test_indices`` list from its training run, the reconstruction is
checked against it and a mismatch is fatal — that is the guarantee that these
numbers describe the same held-out candidates the model was never trained on.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.psc_dataset import (
    PSCCachedDataset,
    read_manifest_rows,
    split_indices_by_source_group,
)
from src.models.three_class_head import CLASS_NAMES, ThreeClassCNN
from src.psc_metrics import (
    binary_collapsed_report,
    class_report,
    duty_cycle_roc_auc,
    group_indices_by_survey,
)

DEFAULT_MANIFEST = REPO_ROOT / "data" / "psc" / "psc_eval_multisurvey.csv"
DEFAULT_CACHE = REPO_ROOT / "data" / "psc" / "raw"
DEFAULT_CHECKPOINT = REPO_ROOT / "outputs" / "models" / "psc_multisurvey_best.pt"
DEFAULT_METRICS = (
    REPO_ROOT / "outputs" / "metrics" / "psc_multisurvey_eval_by_survey.json"
)
BASELINE_CHECKPOINT = REPO_ROOT / "outputs" / "models" / "baseline_cnn_best.pt"

SPLIT_NAMES = ("train", "val", "test")


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def score_indices(
    model: nn.Module,
    dataset: PSCCachedDataset,
    indices: list[int],
    device: torch.device,
    batch_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(logits, labels)`` for ``indices``, in the order given.

    Scoring the whole split once and slicing per survey afterwards is what keeps
    the combined and per-survey numbers arithmetically consistent — they come
    from one forward pass, not three.
    """
    model.eval()
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False)
    all_logits, all_labels = [], []
    for images, labels, _ids in loader:
        all_logits.append(model(images.to(device)).cpu())
        all_labels.append(labels)
    if not all_logits:
        return torch.empty(0, len(CLASS_NAMES)), torch.empty(0, dtype=torch.long)
    return torch.cat(all_logits), torch.cat(all_labels)


def verify_split_matches_checkpoint(
    indices: list[int], recorded: list[int] | None
) -> bool | None:
    """Check a reconstructed test split against the one stored at training time.

    Returns True on a match, None when the checkpoint records no indices to
    check against, and raises otherwise. A mismatch means the manifest, cache,
    or seed has moved since training, so the "held-out" set may contain
    candidates the model was trained on — numbers from it would be meaningless
    rather than merely different.
    """
    if recorded is None:
        return None
    if list(recorded) != list(indices):
        raise RuntimeError(
            "reconstructed test split does not match the one stored in the "
            f"checkpoint ({len(indices)} vs {len(recorded)} indices). The "
            "manifest, cache, or seed differs from the training run, so these "
            "metrics would not be held-out. Refusing to report them."
        )
    return True


def subset_report(
    logits: torch.Tensor,
    labels: torch.Tensor,
    images: list[torch.Tensor],
) -> dict:
    """The full metric bundle for one set of candidates."""
    return {
        "three_class": class_report(logits, labels),
        "binary_collapsed": binary_collapsed_report(logits, labels),
        "duty_cycle_baseline_roc_auc": duty_cycle_roc_auc(images, labels),
    }


def evaluate_by_survey(
    logits: torch.Tensor,
    labels: torch.Tensor,
    surveys: list[str],
    images: list[torch.Tensor],
) -> dict:
    """Combined metrics plus one block per survey.

    ``surveys`` and ``images`` are parallel to ``logits``/``labels`` — i.e. they
    are already restricted to the evaluated split and in the same order.
    """
    if not (len(logits) == len(labels) == len(surveys) == len(images)):
        raise ValueError(
            "logits/labels/surveys/images must be parallel, got "
            f"{len(logits)}/{len(labels)}/{len(surveys)}/{len(images)}"
        )

    positions = list(range(len(surveys)))
    result = {
        "combined": subset_report(logits, labels, images),
        "by_survey": {},
    }
    for survey, rows in group_indices_by_survey(surveys, positions).items():
        picker = torch.tensor(rows, dtype=torch.long)
        result["by_survey"][survey] = subset_report(
            logits[picker], labels[picker], [images[i] for i in rows]
        )
    return result


def format_report(result: dict) -> str:
    """Human-readable rendering of :func:`evaluate_by_survey` output."""
    blocks = [("combined", result["combined"])] + sorted(result["by_survey"].items())
    lines = []
    for name, block in blocks:
        three = block["three_class"]
        binary = block["binary_collapsed"]
        lines.append(f"=== {name} ===")
        lines.append(f"  n                    : {three['n']}")
        lines.append(f"  three-class accuracy : {_fmt(three['accuracy'])}")
        for cls in three["class_order"]:
            m = three["per_class"][cls]
            lines.append(
                f"    {cls:13s} P={_fmt(m['precision'])} R={_fmt(m['recall'])} "
                f"F1={_fmt(m['f1'])} n={m['support']}"
            )
        lines.append("  confusion (rows=true, cols=pred): " + str(three["confusion_matrix"]))
        lines.append(f"  binary pulsar-vs-rest ROC-AUC : {_fmt(binary['roc_auc'])}")
        lines.append(f"  binary pulsar-vs-rest PR-AUC  : {_fmt(binary['pr_auc'])}")
        lines.append(
            f"  duty-cycle baseline   ROC-AUC : {_fmt(block['duty_cycle_baseline_roc_auc'])}"
            "   <- the bar to beat"
        )
        lines.append("")
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--metrics-json", type=Path, default=DEFAULT_METRICS)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Split seed; must match the seed the checkpoint was trained with",
    )
    p.add_argument(
        "--split",
        choices=SPLIT_NAMES,
        default="test",
        help="Which split to evaluate (default: the held-out test split)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.checkpoint.resolve() == BASELINE_CHECKPOINT.resolve():
        raise ValueError("that is the binary HTRU1 baseline, not a three-class checkpoint")

    device = select_device()
    print(f"Device: {device}")

    rows = read_manifest_rows(args.manifest)
    dataset = PSCCachedDataset(rows, args.cache_dir)
    print(f"Manifest: {len(rows)} rows -> {len(dataset)} usable ({len(dataset.skipped)} skipped)")
    if len(dataset) == 0:
        raise RuntimeError("no cached candidates; run rank_psc_candidates.py first")

    splits = split_indices_by_source_group(dataset.source_groups, seed=args.seed)
    indices = splits[SPLIT_NAMES.index(args.split)]
    print(
        "  split by source_group (seed "
        f"{args.seed}): " + " ".join(f"{n}={len(s)}" for n, s in zip(SPLIT_NAMES, splits))
    )

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    recorded = checkpoint.get("test_indices") if args.split == "test" else None
    split_verified = verify_split_matches_checkpoint(indices, recorded)
    if split_verified:
        print(f"  test split matches the checkpoint's stored indices ({len(indices)})")

    model = ThreeClassCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    logits, labels = score_indices(model, dataset, indices, device, args.batch_size)
    result = evaluate_by_survey(
        logits,
        labels,
        [dataset.surveys[i] for i in indices],
        [dataset.images[i] for i in indices],
    )

    print()
    print(format_report(result))

    result["provenance"] = {
        "checkpoint": str(args.checkpoint.relative_to(REPO_ROOT)),
        "manifest": str(args.manifest.relative_to(REPO_ROOT)),
        "split": args.split,
        "split_seed": args.seed,
        "split_sizes": dict(zip(SPLIT_NAMES, (len(s) for s in splits))),
        "n_manifest_rows": len(rows),
        "n_skipped": len(dataset.skipped),
        "split_matches_checkpoint_test_indices": split_verified,
        "checkpoint_best_epoch": checkpoint.get("best_epoch"),
        "checkpoint_val_accuracy": checkpoint.get("val_accuracy"),
        "trained_no_model_modified": True,
    }

    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_json, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Saved metrics: {args.metrics_json}")


if __name__ == "__main__":
    main()
