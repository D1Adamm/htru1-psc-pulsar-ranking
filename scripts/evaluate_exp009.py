#!/usr/bin/env python3
"""Evaluate EXP-009 runs on identical rows and apply the pre-specified rules.

Stages:

  baseline-check  Recompute the no-fit bars and compare to the frozen protocol values.
                  Writes nothing and needs no training.
  pilot           Assignment 0. Reports a go/extend decision, never a final verdict.
  full            Assignments 0-2. Applies Claims A, B, and C.

``--dry-run`` reports which runs are present or missing and writes nothing. Reports
go to ``outputs/exp009/evaluation/<stage>/`` and existing reports are never replaced.
Noise results are provisional: each survey's noise comes from one night.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.psc_controls import INTERVENTIONS
from src.datasets.psc_folds import (
    assert_under,
    attach_identity,
    git_state,
    linked_components,
    load_candidate_records,
    load_folds,
    read_identity,
    sha256_file,
    shared_sources,
    write_exclusive,
)
from src.datasets.psc_preprocessing import arrays_to_tensor
from src.models.three_class_head import ThreeClassCNN
from src.psc_metrics import duty_cycle
from src.psc_source_metrics import (
    build_population,
    combined_interval,
    component_bootstrap,
    endpoints,
    linked_units,
    one_detection_random_auc,
    one_detection_strongest_auc,
    pair_auc,
    paired_bootstrap,
    paired_difference,
    profile_duty_cycle,
    typical_detection_auc,
)

DEFAULT_MANIFEST = REPO_ROOT / "data" / "psc" / "psc_eval_multisurvey.csv"
DEFAULT_CACHE = REPO_ROOT / "data" / "psc" / "raw"
DEFAULT_IDENTITY = REPO_ROOT / "data" / "psc" / "identity" / "source_identity_v1.csv"
DEFAULT_FOLDS = REPO_ROOT / "data" / "psc" / "folds" / "folds_v1.json"
RUN_ROOT = REPO_ROOT / "outputs" / "exp009" / "runs"
BASELINE_SCORES = REPO_ROOT / "outputs" / "exp009" / "baselines" / "scores.csv"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "exp009" / "evaluation"
HTRU1_SCORE_FILES = (
    REPO_ROOT / "outputs" / "predictions" / "psc_ranked_v3.csv",
    REPO_ROOT / "outputs" / "predictions" / "psc_ranked_gbncc_full.csv",
)

PILOT_ALLOWANCE = 0.04  # largest forest AUC difference across assignments was 0.043
FAILURE_MARGIN = 0.02
NONINFERIOR_MARGIN = 0.03
FPR_CONTRADICTION = 0.10
FINAL_LEVEL = 1 - 0.05 / 3  # Bonferroni across the three within-survey arms
WITHIN_ARMS = ("E1", "E2", "E3")
CROSS_ARMS = ("L1", "L2")
DUTY_INPUT = "duty_cnn_input"
DUTY_NATIVE = "duty_native"
FROZEN_BARS = {
    # population: (duty on CNN input, native duty, HTRU1), typical-detection AUC
    "AO327 RFI": (0.969, 0.967, 0.540),
    "GBNCC RFI": (0.791, 0.858, 0.375),
    "AO327 target RFI": (0.967, 0.966, 0.520),
    "GBNCC target RFI": (0.787, 0.855, 0.363),
}


def expected_runs(stage: str) -> list[str]:
    """Run ids a stage needs, built the same way as the training script."""
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("train_exp009", REPO_ROOT / "scripts" / "train_exp009.py")
    module = module_from_spec(spec)
    sys.modules["train_exp009"] = module
    spec.loader.exec_module(module)
    specs = module.plan_specs("pilot")
    if stage == "full":
        specs += module.plan_specs("extension") + module.plan_specs("e3-completion")
    return [s.run_id for s in specs]


def static_statistics(records: list[dict], cache_dir: Path) -> dict[str, np.ndarray]:
    """Scores that need no fitting. Higher always means more pulsar-like."""
    cnn_input, native = [], []
    for rec in records:
        with open(cache_dir / f"{rec['plot_id']}.json") as f:
            payload = json.load(f)
        fvp = np.asarray(payload["FvP"], dtype=float)
        tvp = np.asarray(payload["TvP"], dtype=float)
        cnn_input.append(-duty_cycle(arrays_to_tensor(fvp, tvp)))
        native.append(-profile_duty_cycle(fvp.mean(0)))
    scores: dict[str, float] = {}
    for path in HTRU1_SCORE_FILES:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if row["score"]:
                    scores[row["plot_id"]] = float(row["score"])
    return {
        DUTY_INPUT: np.array(cnn_input),
        DUTY_NATIVE: np.array(native),
        "htru1": np.array([scores[r["plot_id"]] for r in records]),
        "sigma": np.array([r["sigma"] for r in records]),
    }


def populations(records: list[dict]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    survey = np.array([r["survey"] for r in records])
    category = np.array([r["category"] for r in records])
    uncertain = np.array([r["label_uncertain"] for r in records])
    fundamental = np.array([r["fundamental"] for r in records])
    shared = shared_sources(records)
    keep = np.array([r["source_id"] not in shared for r in records])
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, value in (("AO327", "ao327"), ("GBNCC", "gbncc")):
        pulsar = (survey == value) & (category == "known_pulsar")
        rfi = (survey == value) & (category == "rfi")
        noise = (survey == value) & (category == "noise")
        out[f"{name} RFI"] = (pulsar, rfi)
        out[f"{name} all negatives"] = (pulsar, rfi | noise)
        out[f"{name} noise (provisional)"] = (pulsar, noise)
        out[f"{name} RFI, uncertain excluded"] = (pulsar & ~uncertain, rfi)
        out[f"{name} RFI, fundamental"] = (pulsar & fundamental, rfi)
        out[f"{name} target RFI"] = (pulsar & keep, rfi)
    return out


SCORE_DEFINITION = "CNN rows ranked by pulsar log-odds, logit_pulsar - logaddexp(logit_rfi, logit_noise)"


def pulsar_log_odds(logits: np.ndarray) -> np.ndarray:
    """log P(pulsar) - log P(not pulsar) from three-class logits.

    Ranks rows exactly as softmax ``p_pulsar`` does, without the ties that
    probabilities rounded to 8 decimals create once they saturate.
    """
    return logits[..., 0] - np.logaddexp(logits[..., 1], logits[..., 2])


def read_run_scores(run_ids: list[str], records: list[dict], folds: dict) -> tuple[dict, list[str]]:
    """``{run_id: {"test": {...}, "val": {...}}}``; second value lists missing runs."""
    identity_hash = folds["identity_sha256"]
    loaded, missing = {}, []
    for run_id in run_ids:
        path = RUN_ROOT / run_id
        metadata_path = path / "metadata.json"
        if not metadata_path.is_file():
            missing.append(run_id)
            continue
        metadata = json.loads(metadata_path.read_text())
        if metadata["identity_sha256"] != identity_hash:
            raise ValueError(f"{run_id} was trained on a different identity table")
        scores: dict[str, dict[str, float]] = {"test": {}, "val": {}}
        with open(path / "scores.csv", newline="") as f:
            for row in csv.DictReader(f):
                logits = np.array([float(row["logit_known_pulsar"]), float(row["logit_rfi"]), float(row["logit_noise"])])
                scores[row["split"]][row["plot_id"]] = float(pulsar_log_odds(logits))
        loaded[run_id] = {"metadata": metadata, **scores}
    return loaded, missing


def arm_scores(arm: str, runs: dict, plot_ids: list[str], assignments: list[int], suffix: str | None = None) -> np.ndarray:
    """Test scores as (assignments, rows). Rows never scored by this arm stay NaN."""
    out = np.full((len(assignments), len(plot_ids)), np.nan)
    index = {p: i for i, p in enumerate(plot_ids)}
    for run_id, run in runs.items():
        meta = run["metadata"]
        if meta["arm"] != arm or meta["assignment"] not in assignments:
            continue
        if suffix and meta.get("target_survey") != suffix:
            continue
        row = assignments.index(meta["assignment"])
        for plot_id, score in run["test"].items():
            if not np.isnan(out[row, index[plot_id]]):
                raise ValueError(f"{plot_id} was scored twice by {arm} in assignment {meta['assignment']}")
            out[row, index[plot_id]] = score
    return out


def baseline_scores(plot_ids: list[str], assignments: list[int], evaluation: str = "within") -> dict[str, np.ndarray]:
    if not BASELINE_SCORES.is_file():
        return {}
    index = {p: i for i, p in enumerate(plot_ids)}
    out: dict[str, np.ndarray] = {}
    with open(BASELINE_SCORES, newline="") as f:
        for row in csv.DictReader(f):
            if row["evaluation"] != evaluation or int(row["assignment"]) not in assignments:
                continue
            table = out.setdefault(row["model"], np.full((len(assignments), len(plot_ids)), np.nan))
            table[assignments.index(int(row["assignment"])), index[row["plot_id"]]] = float(row["p_pulsar"])
    return out


def pilot_decision(differences: dict[str, dict], allowance: float = PILOT_ALLOWANCE) -> dict:
    """One seed cannot establish failure, so the pilot only chooses what runs next."""
    per_arm = {}
    for arm, d in differences.items():
        if d["low"] > allowance:
            per_arm[arm] = "clear_signal"
        elif d["high"] < allowance:
            per_arm[arm] = "provisional_negative"
        else:
            per_arm[arm] = "inconclusive"
    values = list(per_arm.values())
    if "clear_signal" in values:
        outcome, action = "clear_signal", "Run the full study."
    elif values and all(v == "provisional_negative" for v in values):
        outcome, action = "provisional_negative", "Run the fixed extension; hold preprocessing."
    else:
        outcome, action = "inconclusive", "Run the fixed extension."
    return {"per_arm": per_arm, "outcome": outcome, "action": action, "allowance": allowance}


def claim_a(gbncc_vs_control, gbncc_vs_duty, ao327_vs_duty, fundamental_vs_control, fundamental_vs_duty, coherence_drop, gbncc_fpr_vs_duty) -> str:
    """Image learning. All arguments are ``{arm: result}`` dicts."""
    arms = list(gbncc_vs_control)
    for arm in arms:
        supported = (
            gbncc_vs_control[arm]["low"] > 0
            and gbncc_vs_duty[arm]["low"] > 0
            and ao327_vs_duty[arm]["low"] > -NONINFERIOR_MARGIN
            and fundamental_vs_control[arm]["difference"] > 0
            and fundamental_vs_duty[arm]["difference"] > 0
            and coherence_drop.get(arm, 0) > 0
            and gbncc_fpr_vs_duty[arm]["high"] <= FPR_CONTRADICTION
        )
        if supported:
            return "success"
    if all(gbncc_vs_control[arm]["high"] < FAILURE_MARGIN for arm in arms):
        return "failure"
    return "inconclusive"


def claim_b(by_target: dict[str, dict[str, dict]]) -> str:
    """Transfer. ``by_target`` is ``{target: {arm: difference vs duty on CNN input}}``."""
    targets = list(by_target)
    arms = list(by_target[targets[0]])
    for arm in arms:
        matches = all(by_target[t][arm]["low"] > -NONINFERIOR_MARGIN for t in targets)
        beats = any(by_target[t][arm]["low"] > 0 for t in targets)
        if matches and beats:
            return "success"
    if all(by_target[t][arm]["high"] < 0 for t in targets for arm in arms):
        return "failure"
    return "inconclusive"


def claim_c(auc_difference: dict, fpr_difference: dict, base_auc: float) -> str:
    """Added value of combining CNN scores with one baseline in one survey."""
    if auc_difference["low"] > 0 or fpr_difference["high"] < 0:
        return "success"
    if base_auc >= 0.99:
        return "inconclusive: baseline at ceiling"
    if auc_difference["high"] < 0.01 and auc_difference["high"] - auc_difference["low"] < 0.06:
        return "failure"
    return "inconclusive"


def stacked_scores(arm: str, runs: dict, records: list[dict], baseline: np.ndarray, assignment_position: int, assignment: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit on each run's validation rows, score its test rows, with and without the CNN.

    Neither split trained that CNN, which avoids the leak in EXP-007's hybrid ranker.
    """
    plot_index = {r["plot_id"]: i for i, r in enumerate(records)}
    is_pulsar = np.array([r["category"] == "known_pulsar" for r in records])
    gbncc = np.array([r["survey"] == "gbncc" for r in records], dtype=float)
    uncertain = np.array([r["label_uncertain"] for r in records])
    with_cnn, without_cnn = np.full(len(records), np.nan), np.full(len(records), np.nan)

    for run in runs.values():
        meta = run["metadata"]
        if meta["arm"] != arm or meta["assignment"] != assignment:
            continue
        val_rows = np.array([plot_index[p] for p in run["val"]])
        val_cnn = np.array(list(run["val"].values()))
        keep = ~uncertain[val_rows]
        test_rows = np.array([plot_index[p] for p in run["test"]])
        test_cnn = np.array(list(run["test"].values()))
        base = baseline[assignment_position] if baseline.ndim == 2 else baseline
        fit_base = np.c_[base[val_rows], gbncc[val_rows]][keep]
        fit_both = np.c_[fit_base, val_cnn[keep]]
        labels = is_pulsar[val_rows][keep]
        only = LogisticRegression(max_iter=2000, class_weight="balanced").fit(fit_base, labels)
        both = LogisticRegression(max_iter=2000, class_weight="balanced").fit(fit_both, labels)
        test_base = np.c_[base[test_rows], gbncc[test_rows]]
        without_cnn[test_rows] = only.predict_proba(test_base)[:, 1]
        with_cnn[test_rows] = both.predict_proba(np.c_[test_base, test_cnn])[:, 1]
    return with_cnn, without_cnn


@torch.no_grad()
def intervention_changes(run_ids: list[str], runs: dict, records: list[dict], cache_dir: Path) -> dict:
    """Re-score trained models on altered test images; no parameter changes."""
    index = {r["plot_id"]: i for i, r in enumerate(records)}
    surveys = np.array([r["survey"] for r in records])
    categories = np.array([r["category"] for r in records])
    results: dict = {}
    for run_id in run_ids:
        run = runs[run_id]
        state = torch.load(RUN_ROOT / run_id / "best.pt", map_location="cpu", weights_only=True)
        model = ThreeClassCNN()
        model.load_state_dict(state["model_state_dict"])
        model.eval()
        plot_ids = list(run["test"])
        arrays = {}
        for plot_id in plot_ids:
            with open(cache_dir / f"{plot_id}.json") as f:
                payload = json.load(f)
            arrays[plot_id] = (np.asarray(payload["FvP"], dtype=float), np.asarray(payload["TvP"], dtype=float))
        rows = np.array([index[p] for p in plot_ids])
        original = model(torch.stack([arrays_to_tensor(*arrays[p]) for p in plot_ids]))
        original_class = original.argmax(1).numpy()
        original_score = pulsar_log_odds(original.double().numpy())
        results[run_id] = {}
        for name, transform in INTERVENTIONS.items():
            changed = model(torch.stack([transform(*arrays[p], p) for p in plot_ids]))
            score = pulsar_log_odds(changed.double().numpy())
            effect = {"class_changed": float((changed.argmax(1).numpy() != original_class).mean())}
            for survey in sorted(set(surveys[rows])):
                pulsar = (surveys[rows] == survey) & (categories[rows] == "known_pulsar")
                rfi = (surveys[rows] == survey) & (categories[rows] == "rfi")
                if pulsar.any() and rfi.any():
                    effect[f"auc_drop_{survey}"] = pair_auc(original_score[pulsar], original_score[rfi]) - pair_auc(score[pulsar], score[rfi])
            results[run_id][name] = effect
    return results


def baseline_check(records: list[dict], static: dict[str, np.ndarray]) -> int:
    """Recompute no-fit bars and compare to the frozen protocol values."""
    masks = populations(records)
    failed = 0
    tolerance = 0.0015
    for name, expected in FROZEN_BARS.items():
        population = build_population(records, *masks[name])
        negatives = population.negative_rows
        values = [
            typical_detection_auc(static[key], population.sources, negatives)
            for key in (DUTY_INPUT, DUTY_NATIVE, "htru1")
        ]
        status = all(abs(v - e) <= tolerance for v, e in zip(values, expected))
        failed += not status
        text = ", ".join(f"{v:.3f} (frozen {e:.3f})" for v, e in zip(values, expected))
        print(f"  {name:20s} {len(population.sources):3d} sources: {text} {'OK' if status else 'MISMATCH'}")
    print("Baseline check passed." if not failed else f"Baseline check FAILED for {failed} population(s).")
    return 1 if failed else 0


def evaluate(stage: str, records: list[dict], static: dict, runs: dict, n_boot: int, cache_dir: Path, interventions: bool) -> dict:
    plot_ids = [r["plot_id"] for r in records]
    assignments = [0] if stage == "pilot" else [0, 1, 2]
    level = 0.95 if stage == "pilot" else FINAL_LEVEL
    masks = populations(records)
    sigma = np.array([r["sigma"] for r in records])
    components = linked_components(records)

    within = {arm: arm_scores(arm, runs, plot_ids, assignments) for arm in (*WITHIN_ARMS, "C1")}
    within.update({key: np.tile(static[key], (len(assignments), 1)) for key in static})
    within.update(baseline_scores(plot_ids, assignments))
    arms = [a for a in WITHIN_ARMS if not np.isnan(within[a]).all()]

    report: dict = {"stage": stage, "assignments": assignments, "interval_level": level, "score": SCORE_DEFINITION, "populations": {}, "cross_survey": {}}
    if not BASELINE_SCORES.is_file():
        report["warning"] = "Forest baselines are missing; only no-fit bars are included."

    differences: dict = {}
    for name, (positive, negative) in masks.items():
        if "target" in name:
            continue
        population = build_population(records, positive, negative)
        rng = np.random.default_rng(0)
        negatives = population.negative_rows
        section = {"sources": len(population.sources), "negative_units": len(population.negatives), "models": {}}
        for model, table in within.items():
            if np.isnan(table[:, np.concatenate([*population.sources, negatives])]).any():
                continue
            values = endpoints(table, population, rng)
            values["one_detection_random_auc"] = float(np.mean([one_detection_random_auc(s, population.sources, negatives, rng) for s in table]))
            values["one_detection_strongest_auc"] = float(np.mean([one_detection_strongest_auc(s, population.sources, negatives, sigma) for s in table]))
            section["models"][model] = values
        report["populations"][name] = section
        scored = {m: within[m] for m in section["models"]}
        draws = paired_bootstrap(scored, population, n_boot=n_boot)
        for model, values in section["models"].items():
            values["typical_auc_ci"] = combined_interval(values["typical_auc"], draws[model][:, 0], values["typical_auc_by_assignment"], level)
            values["fpr_at_90_ci"] = combined_interval(values["fpr_at_90"], draws[model][:, 1], values["fpr_at_90_by_assignment"], level)
        contrasts = {}
        for arm in arms:
            if arm not in section["models"]:
                continue
            for reference in ("C1", DUTY_INPUT, DUTY_NATIVE, "period_dm", "all_metadata", "duty_dmc"):
                if reference not in section["models"]:
                    continue
                a, b = section["models"][arm], section["models"][reference]
                contrasts[f"{arm} - {reference}, typical AUC"] = paired_difference(draws[arm][:, 0], draws[reference][:, 0], a["typical_auc_by_assignment"], b["typical_auc_by_assignment"], level)
                contrasts[f"{arm} - {reference}, FPR at 90"] = paired_difference(draws[arm][:, 1], draws[reference][:, 1], a["fpr_at_90_by_assignment"], b["fpr_at_90_by_assignment"], level)
        section["contrasts"] = contrasts
        differences[name] = contrasts

    # Sensitivity: resample whole linked blocks for the two primary populations.
    for name in ("AO327 RFI", "GBNCC RFI"):
        population = build_population(records, *masks[name])
        scored = {m: within[m] for m in report["populations"][name]["models"]}
        draws = component_bootstrap(scored, linked_units(population, components), n_boot=max(100, n_boot // 4))
        section = report["populations"][name]
        section["component_resampling"] = {}
        for arm in arms:
            if arm in scored and "C1" in scored:
                a, b = section["models"][arm], section["models"]["C1"]
                section["component_resampling"][f"{arm} - C1, typical AUC"] = paired_difference(draws[arm][:, 0], draws["C1"][:, 0], a["typical_auc_by_assignment"], b["typical_auc_by_assignment"], level)

    # Cross-survey, on the survey the model never trained on.
    cross_differences: dict = {}
    for target, label in (("gbncc", "GBNCC"), ("ao327", "AO327")):
        population = build_population(records, *masks[f"{label} target RFI"])
        table = {arm: arm_scores(arm, runs, plot_ids, assignments, suffix=target) for arm in CROSS_ARMS}
        table.update({key: np.tile(static[key], (len(assignments), 1)) for key in static})
        source = "ao327" if target == "gbncc" else "gbncc"
        for model, scores in baseline_scores(plot_ids, assignments, f"{source}->{target}").items():
            table[model] = scores
        negatives = population.negative_rows
        all_rows = np.concatenate([*population.sources, negatives])
        table = {m: s for m, s in table.items() if not np.isnan(s[:, all_rows]).any()}
        rng = np.random.default_rng(1)
        section = {"sources": len(population.sources), "models": {m: endpoints(s, population, rng) for m, s in table.items()}}
        draws = paired_bootstrap(table, population, n_boot=n_boot, seed=1)
        section["contrasts"] = {}
        for arm in CROSS_ARMS:
            if arm in table:
                a, b = section["models"][arm], section["models"][DUTY_INPUT]
                difference = paired_difference(draws[arm][:, 0], draws[DUTY_INPUT][:, 0], a["typical_auc_by_assignment"], b["typical_auc_by_assignment"], level)
                section["contrasts"][f"{arm} - {DUTY_INPUT}, typical AUC"] = difference
                cross_differences.setdefault(target, {})[arm] = difference
                section["contrasts"][f"{arm} - {DUTY_NATIVE}, typical AUC"] = paired_difference(draws[arm][:, 0], draws[DUTY_NATIVE][:, 0], a["typical_auc_by_assignment"], section["models"][DUTY_NATIVE]["typical_auc_by_assignment"], level)
        report["cross_survey"][target] = section

    gbncc = differences["GBNCC RFI"]
    versus_control = {arm: gbncc[f"{arm} - C1, typical AUC"] for arm in arms}
    coherence: dict = {}
    if interventions:
        chosen = [run_id for run_id, run in runs.items() if run["metadata"]["arm"] in arms and run["metadata"]["assignment"] in assignments]
        effects = intervention_changes(chosen, runs, records, cache_dir)
        report["interventions"] = effects
        for arm in arms:
            drops = [e["row_shuffled"]["auc_drop_gbncc"] for run_id, e in effects.items() if runs[run_id]["metadata"]["arm"] == arm and "auc_drop_gbncc" in e["row_shuffled"]]
            coherence[arm] = float(np.mean(drops)) if drops else 0.0
        report["coherence_drop_gbncc"] = coherence

    # Added ranking value, with baselines fitted on validation rows only.
    claim_c_results: dict = {}
    reference_tables = {DUTY_INPUT: within[DUTY_INPUT]}
    if "duty_dmc" in within:
        reference_tables["duty_dmc"] = within["duty_dmc"]
    for arm in arms:
        for reference, table in reference_tables.items():
            combined, alone = [], []
            for position, assignment in enumerate(assignments):
                both, base = stacked_scores(arm, runs, records, table, position, assignment)
                combined.append(both)
                alone.append(base)
            models = {"combined": np.array(combined), "baseline": np.array(alone)}
            for label in ("AO327 RFI", "GBNCC RFI"):
                population = build_population(records, *masks[label])
                rng = np.random.default_rng(2)
                a, b = endpoints(models["combined"], population, rng), endpoints(models["baseline"], population, rng)
                draws = paired_bootstrap(models, population, n_boot=n_boot, seed=2)
                auc_difference = paired_difference(draws["combined"][:, 0], draws["baseline"][:, 0], a["typical_auc_by_assignment"], b["typical_auc_by_assignment"], level)
                fpr_difference = paired_difference(draws["combined"][:, 1], draws["baseline"][:, 1], a["fpr_at_90_by_assignment"], b["fpr_at_90_by_assignment"], level)
                claim_c_results[f"{arm} + {reference}, {label}"] = {
                    "combined_auc": a["typical_auc"],
                    "baseline_auc": b["typical_auc"],
                    "auc_difference": auc_difference,
                    "fpr_difference": fpr_difference,
                    "verdict": claim_c(auc_difference, fpr_difference, b["typical_auc"]),
                }
    report["claim_c"] = claim_c_results

    if stage == "pilot":
        pilot_arms = {arm: versus_control[arm] for arm in ("E1", "E2") if arm in versus_control}
        report["pilot_decision"] = pilot_decision(pilot_arms)
        report["note"] = "Pilot evidence from one seed and one assignment. This is not a final verdict."
    else:
        fundamental = differences["GBNCC RFI, fundamental"]
        report["claim_a"] = claim_a(
            versus_control,
            {arm: gbncc[f"{arm} - {DUTY_INPUT}, typical AUC"] for arm in arms},
            {arm: differences["AO327 RFI"][f"{arm} - {DUTY_INPUT}, typical AUC"] for arm in arms},
            {arm: fundamental[f"{arm} - C1, typical AUC"] for arm in arms},
            {arm: fundamental[f"{arm} - {DUTY_INPUT}, typical AUC"] for arm in arms},
            coherence,
            {arm: gbncc[f"{arm} - {DUTY_INPUT}, FPR at 90"] for arm in arms},
        )
        report["claim_b"] = claim_b(cross_differences)
    report["claim_d"] = "Not assessable with current data."
    return report


def render_report(report: dict) -> str:
    lines = [f"# EXP-009 evaluation: {report['stage']}", ""]
    if "pilot_decision" in report:
        decision = report["pilot_decision"]
        lines += [f"**Pilot decision: {decision['outcome']}.** {decision['action']}", "", report["note"], ""]
    for key in ("claim_a", "claim_b", "claim_d"):
        if key in report:
            lines.append(f"- {key}: {report[key]}")
    lines += ["", "Noise results are provisional: each survey's noise comes from one night.", ""]
    for name, section in report["populations"].items():
        lines += [f"## {name} ({section['sources']} sources)", "", "| Model | Typical AUC | FPR at 90% | One random | One strongest |", "|---|---|---|---|---|"]
        for model, values in section["models"].items():
            ci = values.get("typical_auc_ci", (float("nan"), float("nan")))
            lines.append(f"| {model} | {values['typical_auc']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | {values['fpr_at_90']:.3f} | {values['one_detection_random_auc']:.3f} | {values['one_detection_strongest_auc']:.3f} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("baseline-check", "pilot", "full"), required=True)
    parser.add_argument("--dry-run", action="store_true", help="Report run availability; write nothing")
    parser.add_argument("--name", help="Output directory under evaluation/ (default: the stage name)")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--no-interventions", action="store_true", help="Skip re-scoring on altered images")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--identity", type=Path, default=DEFAULT_IDENTITY)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = OUTPUT_ROOT / (args.name or args.stage)
    assert_under(output_dir / "report.json", OUTPUT_ROOT)
    folds = load_folds(args.folds)
    if folds["identity_sha256"] != sha256_file(args.identity):
        raise ValueError("folds were built from a different identity table")
    records = load_candidate_records(args.manifest, args.cache_dir)
    attach_identity(records, read_identity(args.identity))
    static = static_statistics(records, args.cache_dir)

    if args.stage == "baseline-check":
        return baseline_check(records, static)

    run_ids = expected_runs(args.stage)
    runs, missing = read_run_scores(run_ids, records, folds)
    print(f"{args.stage}: {len(runs)} of {len(run_ids)} runs present; forest baselines present: {BASELINE_SCORES.is_file()}")
    if args.dry_run:
        print(f"Missing: {missing if len(missing) < 8 else str(len(missing)) + ' runs'}")
        print("Dry run passed. Nothing was evaluated or written.")
        return 0
    if missing:
        print(f"Cannot evaluate; missing runs: {missing}")
        return 1

    if output_dir.exists():
        print(f"Refusing to overwrite {output_dir}. Inspect or move it before rerunning.")
        return 1
    report = evaluate(args.stage, records, static, runs, args.n_boot, args.cache_dir, not args.no_interventions)
    report["code"] = git_state(REPO_ROOT)
    write_exclusive(output_dir / "report.json", json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_exclusive(output_dir / "report.md", render_report(report))
    print(f"Wrote {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
