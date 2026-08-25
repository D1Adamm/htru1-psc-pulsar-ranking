"""Classification metrics shared by the PSC three-class scripts.

Kept out of any one script so training and post-hoc evaluation report the same
numbers computed by the same code, rather than by two implementations that can
drift apart.

Undefined quantities are ``None`` rather than NaN: ``json.dump`` writes NaN as
the non-standard literal ``NaN``, and a metrics file that a strict JSON reader
rejects is not a reproducible record.
"""

from __future__ import annotations

import torch

from src.models.three_class_head import CLASS_NAMES, NUM_CLASSES, PULSAR_INDEX


def duty_cycle(image: torch.Tensor) -> float:
    """Fraction of phase bins above half maximum in the FvP phase profile.

    The EXP-003 baseline: pulsars are compact in phase, this RFI is broad.
    Lower means narrower means more pulsar-like.
    """
    profile = image[0].mean(dim=0)
    lo, hi = float(profile.min()), float(profile.max())
    if hi - lo < 1e-9:
        return 1.0
    normalized = (profile - lo) / (hi - lo)
    return float((normalized > 0.5).float().mean())


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def class_report(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    """Three-class accuracy, confusion matrix, and per-class P/R/F1/support.

    ``support`` is the number of true examples of that class, so a reader can
    tell a genuine 1.0 from one earned on two candidates.
    """
    if logits.ndim != 2 or logits.shape[1] != NUM_CLASSES:
        raise ValueError(f"expected logits of shape (B, {NUM_CLASSES}), got {tuple(logits.shape)}")
    if len(labels) != len(logits):
        raise ValueError(f"logits/labels length mismatch: {len(logits)} vs {len(labels)}")

    preds = logits.argmax(dim=1)
    confusion = [[0] * NUM_CLASSES for _ in range(NUM_CLASSES)]
    for t, p in zip(labels.tolist(), preds.tolist()):
        confusion[t][p] += 1

    per_class: dict[str, dict] = {}
    for i, name in enumerate(CLASS_NAMES):
        support = int((labels == i).sum())
        n_correct = int(((labels == i) & (preds == i)).sum())
        n_pred = int((preds == i).sum())
        precision = _ratio(n_correct, n_pred)
        recall = _ratio(n_correct, support)
        if precision and recall:
            f1: float | None = 2 * precision * recall / (precision + recall)
        elif precision is None or recall is None:
            f1 = None
        else:
            f1 = 0.0
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    return {
        "n": int(len(labels)),
        "accuracy": _ratio(int((preds == labels).sum()), len(labels)),
        "class_order": list(CLASS_NAMES),
        "confusion_matrix": confusion,
        "per_class": per_class,
    }


def binary_collapsed_report(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    """Collapse to pulsar-vs-rest so the number is comparable to EXP-001..003.

    ROC-AUC and PR-AUC are omitted when the subset is single-class, where
    neither is defined.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    from src.models.three_class_head import collapse_to_pulsar_score

    scores = collapse_to_pulsar_score(logits).numpy()
    binary = (labels == PULSAR_INDEX).long().numpy()
    out: dict[str, object] = {
        "n": int(len(binary)),
        "n_pulsar": int(binary.sum()),
        "roc_auc": None,
        "pr_auc": None,
    }
    if len(binary) and binary.min() != binary.max():
        out["roc_auc"] = float(roc_auc_score(binary, scores))
        out["pr_auc"] = float(average_precision_score(binary, scores))
    return out


def duty_cycle_roc_auc(images: list[torch.Tensor], labels: torch.Tensor) -> float | None:
    """Pulsar-vs-rest ROC-AUC of the one-line duty-cycle statistic.

    Reported next to every model number: if the CNN cannot beat this, that is
    the finding.
    """
    from sklearn.metrics import roc_auc_score

    binary = (labels == PULSAR_INDEX).long().numpy()
    if not len(binary) or binary.min() == binary.max():
        return None
    scores = [-duty_cycle(image) for image in images]
    return float(roc_auc_score(binary, scores))


def group_indices_by_survey(surveys: list[str], indices: list[int]) -> dict[str, list[int]]:
    """Map survey name -> the subset of ``indices`` belonging to that survey.

    ``indices`` are positions into ``surveys``. Returned in sorted survey order
    so the metrics file is stable across runs.
    """
    grouped: dict[str, list[int]] = {}
    for index in indices:
        grouped.setdefault(surveys[index], []).append(index)
    return {survey: grouped[survey] for survey in sorted(grouped)}
