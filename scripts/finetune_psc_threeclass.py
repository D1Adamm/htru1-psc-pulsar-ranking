#!/usr/bin/env python3
"""Fine-tune the HTRU1 backbone into a three-class PSC AO327 classifier.

Classes: known_pulsar / rfi / noise. HTRU1 has no RFI labels, so the three-way
head is trained entirely on PSC data on top of HTRU1-pretrained convolutional
features.

Splits are made by ``source_group`` so repeat detections of one pulsar never
straddle train and test. Results are reported three ways:

  * three-class accuracy and per-class recall on the held-out test split
  * binary-collapsed pulsar-vs-rest ROC-AUC, comparable to EXP-001..003
  * the duty-cycle baseline on the same test split, as the bar to beat

Never writes to the HTRU1 baseline checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.psc_dataset import (
    PSCCachedDataset,
    class_weights,
    read_manifest_rows,
    split_indices_by_source_group,
)
from src.models.three_class_head import (
    CLASS_NAMES,
    NUM_CLASSES,
    PULSAR_INDEX,
    ThreeClassCNN,
    collapse_to_pulsar_score,
)
from src.psc_metrics import duty_cycle

DEFAULT_MANIFEST = REPO_ROOT / "data" / "psc" / "psc_eval_candidates_v3.csv"
DEFAULT_CACHE = REPO_ROOT / "data" / "psc" / "raw"
DEFAULT_PRETRAINED = REPO_ROOT / "outputs" / "models" / "baseline_cnn_best.pt"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "models" / "psc_threeclass_best.pt"
DEFAULT_METRICS = REPO_ROOT / "outputs" / "metrics" / "psc_threeclass_metrics.json"


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    """Return (logits, labels) over a loader."""
    model.eval()
    all_logits, all_labels = [], []
    for images, labels, _ids in loader:
        logits = model(images.to(device))
        all_logits.append(logits.cpu())
        all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


def three_class_report(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    preds = logits.argmax(dim=1)
    report: dict[str, object] = {
        "accuracy": float((preds == labels).float().mean()),
        "n": int(len(labels)),
    }
    confusion = [[0] * NUM_CLASSES for _ in range(NUM_CLASSES)]
    for t, p in zip(labels.tolist(), preds.tolist()):
        confusion[t][p] += 1
    report["confusion_matrix"] = confusion
    report["class_order"] = list(CLASS_NAMES)
    for i, name in enumerate(CLASS_NAMES):
        n_true = int((labels == i).sum())
        n_correct = int(((labels == i) & (preds == i)).sum())
        n_pred = int((preds == i).sum())
        report[f"recall_{name}"] = n_correct / n_true if n_true else float("nan")
        report[f"precision_{name}"] = n_correct / n_pred if n_pred else float("nan")
    return report


def binary_report(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    """Collapse to pulsar-vs-rest so the number is comparable to EXP-001..003."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    scores = collapse_to_pulsar_score(logits).numpy()
    binary = (labels == PULSAR_INDEX).long().numpy()
    out = {"n": int(len(binary)), "n_pulsar": int(binary.sum())}
    if binary.min() != binary.max():
        out["roc_auc"] = float(roc_auc_score(binary, scores))
        out["pr_auc"] = float(average_precision_score(binary, scores))
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--metrics-json", type=Path, default=DEFAULT_METRICS)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Train only the head; keep HTRU1 convolutional filters fixed",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.output.resolve() == DEFAULT_PRETRAINED.resolve():
        raise ValueError("refusing to overwrite the HTRU1 baseline checkpoint")

    torch.manual_seed(args.seed)
    device = select_device()
    print(f"Device: {device}")

    rows = read_manifest_rows(args.manifest)
    dataset = PSCCachedDataset(rows, args.cache_dir)
    print(f"Manifest: {len(rows)} rows -> {len(dataset)} usable ({len(dataset.skipped)} skipped)")
    if len(dataset) == 0:
        raise RuntimeError("no cached candidates; run rank_psc_candidates.py first")

    counts = dataset.class_counts()
    print("  class counts: " + ", ".join(f"{CLASS_NAMES[i]}={counts.get(i,0)}" for i in range(NUM_CLASSES)))

    train_idx, val_idx, test_idx = split_indices_by_source_group(
        dataset.source_groups, seed=args.seed
    )
    print(f"  split by source_group: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    overlap = {dataset.source_groups[i] for i in train_idx} & {
        dataset.source_groups[i] for i in test_idx
    }
    if overlap:
        raise RuntimeError(f"source group leaked across train/test: {sorted(overlap)[:5]}")

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size)

    model = ThreeClassCNN().to(device)
    checkpoint = torch.load(args.pretrained, map_location=device, weights_only=False)
    info = model.load_pretrained_backbone(checkpoint["model_state_dict"])
    print(f"  loaded {len(info['loaded'])} pretrained tensors, skipped {info['skipped']}")
    if args.freeze_backbone:
        model.freeze_backbone()
        print("  backbone frozen")

    weights = class_weights([dataset.labels[i] for i in train_idx], NUM_CLASSES).to(device)
    print("  class weights: " + ", ".join(f"{CLASS_NAMES[i]}={weights[i]:.3f}" for i in range(NUM_CLASSES)))
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )

    best_val, best_state, best_epoch = -1.0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, n_batches = 0.0, 0
        for images, labels, _ids in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            n_batches += 1

        v_logits, v_labels = evaluate(model, val_loader, device)
        v_acc = float((v_logits.argmax(1) == v_labels).float().mean())
        t_logits, t_labels = evaluate(model, train_loader, device)
        t_acc = float((t_logits.argmax(1) == t_labels).float().mean())
        print(
            f"  epoch {epoch:3d}  loss={total/max(n_batches,1):.4f}  "
            f"train_acc={t_acc:.4f}  val_acc={v_acc:.4f}"
            + ("  *" if v_acc > best_val else "")
        )
        if v_acc > best_val:
            best_val, best_epoch = v_acc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    print(f"\nBest epoch {best_epoch} (val acc {best_val:.4f})")

    logits, labels = evaluate(model, test_loader, device)
    three = three_class_report(logits, labels)
    binary = binary_report(logits, labels)

    # Duty-cycle baseline on the identical test split.
    from sklearn.metrics import roc_auc_score

    duty = np.array([duty_cycle(dataset.images[i]) for i in test_idx])
    is_pulsar = np.array([1 if dataset.labels[i] == PULSAR_INDEX else 0 for i in test_idx])
    duty_auc = (
        float(roc_auc_score(is_pulsar, -duty)) if is_pulsar.min() != is_pulsar.max() else float("nan")
    )

    print("\n=== HELD-OUT TEST ===")
    print(f"  three-class accuracy : {three['accuracy']:.4f}  (n={three['n']})")
    for name in CLASS_NAMES:
        print(f"    recall {name:13s}: {three[f'recall_{name}']:.4f}   precision: {three[f'precision_{name}']:.4f}")
    print(f"  binary pulsar-vs-rest ROC-AUC : {binary.get('roc_auc', float('nan')):.4f}")
    print(f"  duty-cycle baseline   ROC-AUC : {duty_auc:.4f}   <- the bar to beat")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": list(CLASS_NAMES),
            "best_epoch": best_epoch,
            "val_accuracy": best_val,
            "random_seed": args.seed,
            "pretrained_from": str(args.pretrained),
            "manifest": str(args.manifest),
            "test_indices": test_idx,
        },
        args.output,
    )

    metrics = {
        "three_class": three,
        "binary_collapsed": binary,
        "duty_cycle_baseline_roc_auc": duty_auc,
        "best_epoch": best_epoch,
        "val_accuracy": best_val,
        "seed": args.seed,
        "split_sizes": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "class_counts": {CLASS_NAMES[i]: counts.get(i, 0) for i in range(NUM_CLASSES)},
        "frozen_backbone": bool(args.freeze_backbone),
    }
    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_json, "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"\nSaved checkpoint: {args.output}")
    print(f"Saved metrics:    {args.metrics_json}")


if __name__ == "__main__":
    main()
