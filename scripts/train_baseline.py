#!/usr/bin/env python3
"""Train BaselineCNN on HTRU1 training batches with an 80/20 validation split.

Does not touch ``test_batch``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.htru1_dataset import HTRU1Dataset
from src.models.baseline_cnn import BaselineCNN

DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 1e-3
DEFAULT_EPOCHS = 5
DEFAULT_VAL_FRACTION = 0.2
DEFAULT_CHECKPOINT = REPO_ROOT / "outputs" / "models" / "baseline_cnn_best.pt"
HTRU1_DIR = REPO_ROOT / "data" / "raw" / "htru1-batches-py"
TRAIN_BATCHES = [HTRU1_DIR / f"data_batch_{i}" for i in range(1, 6)]


def select_device() -> torch.device:
    """Pick CUDA, Apple MPS, or CPU depending on availability."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_pos_weight(labels: torch.Tensor) -> torch.Tensor:
    """Return ``n_non_pulsar / n_pulsar`` for ``BCEWithLogitsLoss``.

    Labels must use ``1 = pulsar`` and ``0 = non-pulsar``.
    """
    labels = labels.detach().float().view(-1)
    n_pulsar = int((labels == 1.0).sum().item())
    n_non_pulsar = int((labels == 0.0).sum().item())
    if n_pulsar == 0:
        raise ValueError("pos_weight requires at least one pulsar (label=1) example")
    return torch.tensor([n_non_pulsar / n_pulsar], dtype=torch.float32)


def split_train_val_indices(
    n_samples: int,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SEED,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return shuffled train/validation index tensors (80/20 by default)."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    if n_samples < 2:
        raise ValueError(f"need at least 2 samples to split, got {n_samples}")

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_samples, generator=generator)
    n_val = max(1, int(round(n_samples * val_fraction)))
    n_train = n_samples - n_val
    if n_train < 1:
        raise ValueError("train split is empty; decrease val_fraction")
    return perm[:n_train], perm[n_train:]


def binary_classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Precision / recall / F1 from logits using sigmoid and a hard threshold."""
    logits = logits.detach().view(-1)
    labels = labels.detach().float().view(-1)
    preds = (torch.sigmoid(logits) >= threshold).float()

    tp = float(((preds == 1.0) & (labels == 1.0)).sum().item())
    fp = float(((preds == 1.0) & (labels == 0.0)).sum().item())
    fn = float(((preds == 0.0) & (labels == 1.0)).sum().item())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for images, labels, _filenames in loader:
        images = images.to(device)
        labels = labels.to(device).view(-1, 1)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for images, labels, _filenames in loader:
        images = images.to(device)
        labels = labels.to(device).view(-1, 1)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += float(loss.item())
        n_batches += 1
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    metrics = binary_classification_metrics(
        torch.cat(all_logits, dim=0),
        torch.cat(all_labels, dim=0),
        threshold=0.5,
    )
    return total_loss / max(n_batches, 1), metrics


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_f1: float,
    pos_weight: torch.Tensor,
    random_seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_f1": float(val_f1),
            "pos_weight": pos_weight.detach().cpu(),
            "random_seed": int(random_seed),
        },
        path,
    )


def build_dataloaders(
    dataset: HTRU1Dataset,
    train_indices: torch.Tensor,
    val_indices: torch.Tensor,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        Subset(dataset, train_indices.tolist()),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices.tolist()),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=DEFAULT_VAL_FRACTION,
        help="Fraction of the 50k training pool used for validation",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on the training pool size (smoke tests)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Where to save the best validation-F1 checkpoint",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)

    missing = [path for path in TRAIN_BATCHES if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing HTRU1 training batch file(s):\n"
            + "\n".join(f"  - {path}" for path in missing)
        )

    device = select_device()
    print(f"Device: {device}")

    dataset = HTRU1Dataset(TRAIN_BATCHES)
    n_pool = len(dataset)
    if args.max_samples is not None:
        n_pool = min(n_pool, args.max_samples)

    train_indices, val_indices = split_train_val_indices(
        n_pool, val_fraction=args.val_fraction, seed=args.seed
    )
    train_labels = dataset.labels[train_indices]
    pos_weight = compute_pos_weight(train_labels).to(device)

    print(f"Training pool size: {n_pool}")
    print(f"Train size: {len(train_indices)}  Val size: {len(val_indices)}")
    print(f"pos_weight (non-pulsar / pulsar): {float(pos_weight.item()):.4f}")

    train_loader, val_loader = build_dataloaders(
        dataset, train_indices, val_indices, batch_size=args.batch_size
    )

    model = BaselineCNN().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_precision={val_metrics['precision']:.4f}  "
            f"val_recall={val_metrics['recall']:.4f}  "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            save_checkpoint(
                args.checkpoint,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_f1=best_f1,
                pos_weight=pos_weight,
                random_seed=args.seed,
            )
            print(f"  saved best checkpoint -> {args.checkpoint}")

    print(f"Done. Best validation F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
