"""Source-level ranking metrics and paired resampling for EXP-009.

Repeat detections make row-level ranking misleading, so a pulsar source gets one
entry per metric:

  * typical-detection AUC: each source's median row score against all negatives
  * FPR at 90% of sources: take each source's best row, then the share of negatives
    at or above the point where 90% of sources are found
  * one-detection AUC: one row per source, chosen without looking at the score

Resampling draws whole units: sources, RFI nights, and noise observations. A source
seen in two surveys is one unit. Fold assignments score the same rows, so they are
never treated as independent samples: each draw resamples units once, computes the
metric per assignment, and averages.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm, rankdata


@dataclass
class Population:
    """Row indices grouped into source units and negative units."""

    sources: list[np.ndarray]
    negatives: list[np.ndarray]
    source_keys: list[str] = field(default_factory=list)
    negative_keys: list[tuple] = field(default_factory=list)

    @property
    def negative_rows(self) -> np.ndarray:
        return np.concatenate(self.negatives)


def build_population(
    records: list[dict],
    positive_mask: np.ndarray,
    negative_mask: np.ndarray,
) -> Population:
    """Group selected rows into sources, RFI nights, and noise observations."""
    sources: dict[str, list[int]] = {}
    negatives: dict[tuple, list[int]] = {}
    for i, rec in enumerate(records):
        if positive_mask[i]:
            sources.setdefault(rec["source_id"], []).append(i)
        elif negative_mask[i]:
            if rec["category"] == "rfi":
                key = ("rfi", rec["survey"], rec["night"])
            else:
                key = ("noise", rec["obs"])
            negatives.setdefault(key, []).append(i)
    return Population(
        sources=[np.array(v) for v in sources.values()],
        negatives=[np.array(v) for v in negatives.values()],
        source_keys=list(sources),
        negative_keys=list(negatives),
    )


def pair_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Mann-Whitney AUC, with ties counted as half."""
    ranks = rankdata(np.r_[positive, negative])
    n_pos, n_neg = len(positive), len(negative)
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def typical_detection_auc(scores: np.ndarray, sources: list[np.ndarray], negatives: np.ndarray) -> float:
    medians = np.array([np.median(scores[rows]) for rows in sources])
    return pair_auc(medians, scores[negatives])


def fpr_at_source_recall(
    scores: np.ndarray,
    sources: list[np.ndarray],
    negatives: np.ndarray,
    rng: np.random.Generator,
    recall: float = 0.9,
    tie_draws: int = 50,
) -> float:
    """FPR where ``recall`` of sources are found, breaking ties at random.

    Statistics with few distinct values, such as duty cycle on 32 bins, have many
    ties. Counting them all as false positives would penalize those statistics, so
    ties are broken with negligible random noise and the result is averaged.
    """
    best = np.array([scores[rows].max() for rows in sources])
    negative_scores = scores[negatives]
    position = int(np.ceil(recall * len(best))) - 1
    results = []
    for _ in range(tie_draws):
        jittered_best = best + 1e-9 * rng.random(len(best))
        jittered_negative = negative_scores + 1e-9 * rng.random(len(negative_scores))
        threshold = np.sort(jittered_best)[::-1][position]
        results.append(float((jittered_negative >= threshold).mean()))
    return float(np.mean(results))


def one_detection_random_auc(
    scores: np.ndarray,
    sources: list[np.ndarray],
    negatives: np.ndarray,
    rng: np.random.Generator,
    draws: int = 100,
) -> float:
    negative_scores = scores[negatives]
    values = [
        pair_auc(scores[[rng.choice(rows) for rows in sources]], negative_scores)
        for _ in range(draws)
    ]
    return float(np.mean(values))


def one_detection_strongest_auc(
    scores: np.ndarray,
    sources: list[np.ndarray],
    negatives: np.ndarray,
    sigma: np.ndarray,
) -> float:
    """One row per source: the highest PRESTO sigma, which ignores the model score."""
    chosen = np.array([rows[np.argmax(sigma[rows])] for rows in sources])
    return pair_auc(scores[chosen], scores[negatives])


def profile_duty_cycle(profile: np.ndarray) -> float:
    """Share of phase bins above half maximum, for any profile length.

    Matches ``psc_metrics.duty_cycle``, which is restricted to the 32-bin CNN input.
    """
    low, high = float(profile.min()), float(profile.max())
    if high - low < 1e-9:
        return 1.0
    return float((((profile - low) / (high - low)) > 0.5).mean())


def endpoints(
    score_sets: np.ndarray,
    population: Population,
    rng: np.random.Generator,
    tie_draws: int = 50,
) -> dict[str, float]:
    """Mean metrics across assignments. ``score_sets`` is (assignments, rows)."""
    negatives = population.negative_rows
    auc = [typical_detection_auc(s, population.sources, negatives) for s in score_sets]
    fpr = [fpr_at_source_recall(s, population.sources, negatives, rng, tie_draws=tie_draws) for s in score_sets]
    return {
        "typical_auc": float(np.mean(auc)),
        "fpr_at_90": float(np.mean(fpr)),
        "typical_auc_by_assignment": [float(x) for x in auc],
        "fpr_at_90_by_assignment": [float(x) for x in fpr],
    }


def resample_population(population: Population, rng: np.random.Generator) -> Population:
    """Draw units with replacement. A repeated source stays one source per draw."""
    source_draw = rng.integers(0, len(population.sources), len(population.sources))
    negative_draw = rng.integers(0, len(population.negatives), len(population.negatives))
    return Population(
        sources=[population.sources[i] for i in source_draw],
        negatives=[population.negatives[i] for i in negative_draw],
    )


def paired_bootstrap(
    models: dict[str, np.ndarray],
    population: Population,
    n_boot: int = 2000,
    seed: int = 0,
    tie_draws: int = 3,
) -> dict[str, np.ndarray]:
    """Metric draws per model, on identical unit resamples.

    Returns ``{model: array of shape (n_boot, 2)}`` holding typical AUC and FPR.
    """
    rng = np.random.default_rng(seed)
    draws = {name: np.zeros((n_boot, 2)) for name in models}
    for b in range(n_boot):
        sample = resample_population(population, rng)
        negatives = sample.negative_rows
        for name, score_sets in models.items():
            auc = [typical_detection_auc(s, sample.sources, negatives) for s in score_sets]
            fpr = [fpr_at_source_recall(s, sample.sources, negatives, rng, tie_draws=tie_draws) for s in score_sets]
            draws[name][b] = (np.mean(auc), np.mean(fpr))
    return draws


def linked_units(population: Population, components: list[int]) -> list[Population]:
    """Split a population by fold component, for the sensitivity bootstrap.

    Observations can tie sources and negatives together, so this resamples those
    linked blocks whole instead of treating their units as independent.
    """
    blocks: dict[int, Population] = {}

    def block(rows: np.ndarray) -> Population:
        return blocks.setdefault(components[int(rows[0])], Population([], []))

    for rows in population.sources:
        block(rows).sources.append(rows)
    for rows in population.negatives:
        block(rows).negatives.append(rows)
    return list(blocks.values())


def component_bootstrap(
    models: dict[str, np.ndarray],
    blocks: list[Population],
    n_boot: int = 500,
    seed: int = 0,
    tie_draws: int = 3,
) -> dict[str, np.ndarray]:
    """Like :func:`paired_bootstrap`, but drawing whole linked blocks."""
    rng = np.random.default_rng(seed)
    draws = {name: np.zeros((n_boot, 2)) for name in models}
    for b in range(n_boot):
        sources: list[np.ndarray] = []
        negatives: list[np.ndarray] = []
        while not sources or not negatives:
            for i in rng.integers(0, len(blocks), len(blocks)):
                sources.extend(blocks[i].sources)
                negatives.extend(blocks[i].negatives)
        negative_rows = np.concatenate(negatives)
        for name, score_sets in models.items():
            auc = [typical_detection_auc(s, sources, negative_rows) for s in score_sets]
            fpr = [fpr_at_source_recall(s, sources, negative_rows, rng, tie_draws=tie_draws) for s in score_sets]
            draws[name][b] = (np.mean(auc), np.mean(fpr))
    return draws


def combined_interval(
    point: float,
    draws: np.ndarray,
    assignment_values: list[float] | None = None,
    level: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap percentile interval, widened for between-assignment variation.

    Total variance = bootstrap variance + (1 + 1/m) * between-assignment variance.
    With one assignment the between term cannot be estimated, so the interval leaves
    out model-seed variation and must be read with the pilot allowance.
    """
    alpha = (1 - level) / 2
    low, high = np.percentile(draws, [100 * alpha, 100 * (1 - alpha)])
    if not assignment_values or len(assignment_values) < 2:
        return float(low), float(high)
    m = len(assignment_values)
    boot_var = float(np.var(draws, ddof=1))
    total_var = boot_var + (1 + 1 / m) * float(np.var(assignment_values, ddof=1))
    if boot_var <= 0:
        half = norm.ppf(1 - alpha) * np.sqrt(total_var)
        return float(point - half), float(point + half)
    scale = np.sqrt(total_var / boot_var)
    return float(point - (point - low) * scale), float(point + (high - point) * scale)


def paired_difference(
    a_draws: np.ndarray,
    b_draws: np.ndarray,
    a_values: list[float],
    b_values: list[float],
    level: float = 0.95,
) -> dict[str, float]:
    """Interval for ``a - b`` on one metric, using shared bootstrap draws."""
    values = [x - y for x, y in zip(a_values, b_values)]
    point = float(np.mean(values))
    low, high = combined_interval(point, a_draws - b_draws, values, level)
    return {"difference": point, "low": low, "high": high, "level": level}
