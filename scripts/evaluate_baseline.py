#!/usr/bin/env python3
"""Evaluate a trained BaselineCNN checkpoint on the untouched HTRU1 test_batch."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.htru1_dataset import HTRU1Dataset
from src.models.baseline_cnn import BaselineCNN

DEFAULT_CHECKPOINT = REPO_ROOT / "outputs" / "models" / "baseline_cnn_best.pt"
DEFAULT_TEST_BATCH = REPO_ROOT / "data" / "raw" / "htru1-batches-py" / "test_batch"
DEFAULT_PREDICTIONS_CSV = (
    REPO_ROOT / "outputs" / "predictions" / "htru1_test_predictions.csv"
)
DEFAULT_METRICS_JSON = REPO_ROOT / "outputs" / "metrics" / "htru1_test_metrics.json"
DEFAULT_BATCH_SIZE = 64
DEFAULT_THRESHOLD = 0.5


def select_device() -> torch.device:
    """Pick CUDA, Apple MPS, or CPU depending on availability."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def confusion_counts(
    probs: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, int]:
    """Return TP / FP / TN / FN at a probability threshold."""
    probs_t = torch.as_tensor(probs, dtype=torch.float32).view(-1)
    labels_t = torch.as_tensor(labels, dtype=torch.float32).view(-1)
    preds = (probs_t >= threshold).float()

    tp = int(((preds == 1.0) & (labels_t == 1.0)).sum().item())
    fp = int(((preds == 1.0) & (labels_t == 0.0)).sum().item())
    tn = int(((preds == 0.0) & (labels_t == 0.0)).sum().item())
    fn = int(((preds == 0.0) & (labels_t == 1.0)).sum().item())
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def classification_metrics_from_probs(
    probs: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, float]:
    """Threshold metrics plus ROC-AUC and PR-AUC from pulsar probabilities."""
    probs_np = np.asarray(
        probs.detach().cpu() if isinstance(probs, torch.Tensor) else probs,
        dtype=np.float64,
    ).reshape(-1)
    labels_np = np.asarray(
        labels.detach().cpu() if isinstance(labels, torch.Tensor) else labels,
        dtype=np.float64,
    ).reshape(-1)

    counts = confusion_counts(probs_np, labels_np, threshold=threshold)
    tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    # ROC/PR need both classes present.
    if labels_np.min() == labels_np.max():
        roc_auc = float("nan")
        pr_auc = float("nan")
    else:
        roc_auc = float(roc_auc_score(labels_np, probs_np))
        pr_auc = float(average_precision_score(labels_np, probs_np))

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        **counts,
    }


def ranking_metrics(
    probs: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    top_ns: tuple[int, ...] = (10, 50, 100, 200),
    top_fractions: tuple[float, ...] = (0.01, 0.05, 0.10),
) -> dict[str, float]:
    """Count true pulsars in top-N and recall within top percentage cuts."""
    probs_np = np.asarray(
        probs.detach().cpu() if isinstance(probs, torch.Tensor) else probs,
        dtype=np.float64,
    ).reshape(-1)
    labels_np = np.asarray(
        labels.detach().cpu() if isinstance(labels, torch.Tensor) else labels,
        dtype=np.float64,
    ).reshape(-1)

    order = np.argsort(-probs_np, kind="mergesort")
    ranked_labels = labels_np[order]
    n_total = len(ranked_labels)
    n_pulsars = int((labels_np == 1.0).sum())

    metrics: dict[str, float] = {}
    for n in top_ns:
        k = min(int(n), n_total)
        metrics[f"pulsars_in_top_{n}"] = float((ranked_labels[:k] == 1.0).sum())

    for frac in top_fractions:
        k = max(1, int(round(n_total * frac)))
        k = min(k, n_total)
        found = float((ranked_labels[:k] == 1.0).sum())
        pct = int(round(frac * 100))
        metrics[f"recall_at_top_{pct}pct"] = (
            found / n_pulsars if n_pulsars > 0 else 0.0
        )
        metrics[f"top_{pct}pct_size"] = float(k)

    metrics["n_pulsars"] = float(n_pulsars)
    metrics["n_total"] = float(n_total)
    return metrics


def probability_ranks(probs: torch.Tensor | np.ndarray) -> np.ndarray:
    """Rank 1 = highest pulsar probability (stable for ties)."""
    probs_np = np.asarray(
        probs.detach().cpu() if isinstance(probs, torch.Tensor) else probs,
        dtype=np.float64,
    ).reshape(-1)
    order = np.argsort(-probs_np, kind="mergesort")
    ranks = np.empty(len(probs_np), dtype=np.int64)
    ranks[order] = np.arange(1, len(probs_np) + 1)
    return ranks


def build_prediction_rows(
    filenames: list[str],
    labels: torch.Tensor | np.ndarray,
    probs: torch.Tensor | np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[dict[str, object]]:
    """Build CSV rows with filename, label, probability, prediction, and rank."""
    labels_np = np.asarray(
        labels.detach().cpu() if isinstance(labels, torch.Tensor) else labels,
        dtype=np.float64,
    ).reshape(-1)
    probs_np = np.asarray(
        probs.detach().cpu() if isinstance(probs, torch.Tensor) else probs,
        dtype=np.float64,
    ).reshape(-1)
    if len(filenames) != len(labels_np) or len(filenames) != len(probs_np):
        raise ValueError("filenames, labels, and probs must have the same length")

    ranks = probability_ranks(probs_np)
    preds = (probs_np >= threshold).astype(np.int64)
    rows: list[dict[str, object]] = []
    for i, filename in enumerate(filenames):
        rows.append(
            {
                "filename": filename,
                "true_label": int(labels_np[i]),
                "pulsar_probability": float(probs_np[i]),
                "predicted_label": int(preds[i]),
                "rank": int(ranks[i]),
            }
        )
    return rows


@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, torch.Tensor, torch.Tensor, list[str]]:
    """Return mean loss plus concatenated logits, labels, and filenames."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    all_filenames: list[str] = []

    for images, labels, filenames in loader:
        images = images.to(device)
        labels = labels.to(device).view(-1, 1)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += float(loss.item())
        n_batches += 1
        all_logits.append(logits.detach().cpu().view(-1))
        all_labels.append(labels.detach().cpu().view(-1))
        all_filenames.extend(list(filenames))

    return (
        total_loss / max(n_batches, 1),
        torch.cat(all_logits, dim=0),
        torch.cat(all_labels, dim=0),
        all_filenames,
    )


def save_predictions_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "filename",
        "true_label",
        "pulsar_probability",
        "predicted_label",
        "rank",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_metrics_json(path: Path, metrics: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
        f.write("\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--test-batch", type=Path, default=DEFAULT_TEST_BATCH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--predictions-csv", type=Path, default=DEFAULT_PREDICTIONS_CSV)
    parser.add_argument("--metrics-json", type=Path, default=DEFAULT_METRICS_JSON)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")
    if not args.test_batch.is_file():
        raise FileNotFoundError(f"Missing test batch: {args.test_batch}")

    device = select_device()
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test batch: {args.test_batch}")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = BaselineCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    pos_weight = checkpoint.get("pos_weight")
    if pos_weight is not None:
        pos_weight = torch.as_tensor(pos_weight, dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    dataset = HTRU1Dataset([args.test_batch])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    test_loss, logits, labels, filenames = run_inference(
        model, loader, criterion, device
    )
    probs = torch.sigmoid(logits)

    cls_metrics = classification_metrics_from_probs(
        probs, labels, threshold=args.threshold
    )
    rank_metrics = ranking_metrics(probs, labels)
    rows = build_prediction_rows(
        filenames, labels, probs, threshold=args.threshold
    )

    metrics: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "test_batch": str(args.test_batch),
        "n_test": len(dataset),
        "threshold": args.threshold,
        "test_loss": test_loss,
        "confusion_matrix": {
            "tp": cls_metrics["tp"],
            "fp": cls_metrics["fp"],
            "tn": cls_metrics["tn"],
            "fn": cls_metrics["fn"],
        },
        **{k: v for k, v in cls_metrics.items() if k not in {"tp", "fp", "tn", "fn"}},
        **rank_metrics,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_f1": checkpoint.get("val_f1"),
        "checkpoint_pos_weight": (
            float(torch.as_tensor(checkpoint["pos_weight"]).item())
            if checkpoint.get("pos_weight") is not None
            else None
        ),
        "checkpoint_random_seed": checkpoint.get("random_seed"),
    }

    save_predictions_csv(args.predictions_csv, rows)
    save_metrics_json(args.metrics_json, metrics)

    print(f"Test loss: {test_loss:.6f}")
    print(f"Precision: {cls_metrics['precision']:.6f}")
    print(f"Recall: {cls_metrics['recall']:.6f}")
    print(f"F1: {cls_metrics['f1']:.6f}")
    print(f"Accuracy: {cls_metrics['accuracy']:.6f}")
    print(f"ROC-AUC: {cls_metrics['roc_auc']:.6f}")
    print(f"PR-AUC: {cls_metrics['pr_auc']:.6f}")
    print(
        "Confusion matrix: "
        f"TP={cls_metrics['tp']}  FP={cls_metrics['fp']}  "
        f"TN={cls_metrics['tn']}  FN={cls_metrics['fn']}"
    )
    print(
        "Pulsars in top-N: "
        f"10={int(rank_metrics['pulsars_in_top_10'])}  "
        f"50={int(rank_metrics['pulsars_in_top_50'])}  "
        f"100={int(rank_metrics['pulsars_in_top_100'])}  "
        f"200={int(rank_metrics['pulsars_in_top_200'])}"
    )
    print(
        "Recall at top-%: "
        f"1%={rank_metrics['recall_at_top_1pct']:.6f}  "
        f"5%={rank_metrics['recall_at_top_5pct']:.6f}  "
        f"10%={rank_metrics['recall_at_top_10pct']:.6f}"
    )
    print(f"Wrote predictions: {args.predictions_csv}")
    print(f"Wrote metrics: {args.metrics_json}")


if __name__ == "__main__":
    main()
