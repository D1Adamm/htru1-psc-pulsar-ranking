# EXP-002 — First cross-survey PSC AO327 ranking

Applies the EXP-001 HTRU1 checkpoint to a labeled PSC AO327 candidate set and
measures ranking quality. See [exp001_htru1_baseline.md](exp001_htru1_baseline.md)
for the model provenance.

## Setup

| Field | Value |
|---|---|
| `experiment_id` | `EXP-002` |
| `date` | 2026-08-05 |
| `git_commit` | `f0a872d` (+ uncommitted ranking script, manifest, and this record) |
| `model` | EXP-001, `baseline_cnn_best.pt`, sha256 `7f852a3e7fc46ade…` |
| `manifest` | `data/psc/psc_eval_candidates.csv` (150 candidates) |
| `script` | `scripts/rank_psc_candidates.py` |
| `results` | `outputs/predictions/psc_ranked_candidates.csv` |
| `cache` | `data/psc/raw/` (150 JSON files, one fetch each) |
| `threshold` | 0.5 |
| `failures` | 0 / 150 |

Composition: 50 known pulsars, 50 RFI, 50 noise — all AO327, all from the PSC
psrsearch consensus-graded tabs, all via the same JSON endpoint and the same
blind-search pipeline.

## Headline result

The model is a competent pulsar-versus-noise discriminator and is **effectively
blind to RFI**.

| Comparison | ROC-AUC |
|---|---|
| pulsar vs noise | **0.9524** |
| pulsar vs RFI | **0.6168** |

| Category | n | median score | ≥ 0.5 |
|---|---|---|---|
| known_pulsar | 50 | 0.9940 | 40 / 50 |
| rfi | 50 | 0.9752 | **50 / 50** |
| noise | 50 | 0.0651 | 1 / 50 |

The decisive number is not the AUC — it is the overlap:

- lowest-scoring RFI candidate: **0.6146**
- 10 of 50 real pulsars score **below** that

**No threshold separates them.** Any cut that retains ~80% of the known pulsars
admits 100% of the RFI. As a triage tool on this candidate stream, the model
would move every RFI candidate to the top of the review queue.

## Per-detection vs per-source

Repeat detections inflate the aggregate numbers. Counting each `source_group`
once, at its best rank:

| Metric | Per-detection (n=150) | Per-source (n=61 groups) |
|---|---|---|
| ROC-AUC | 0.7846 | **0.5773** |
| PR-AUC | 0.7427 | 0.6883 |
| recall@10 | 0.200 (10/10 precision) | 0.370 |
| recall@50 | 0.600 | 0.704 |

Median rank of a distinct pulsar source: **41 of 150**.

> **CORRECTION (2026-08-05, later same day).** The per-source ROC-AUC of 0.5773
> above is **distorted and should not be quoted**, and the instruction to "report
> the per-source number as the honest one" was wrong.
>
> The v1 manifest assigned **all 50 noise candidates to a single source group**
> (a hardcoded special case, since they shared one MJD and pointing). The
> collapse rule then took the best-ranked member per group — which for a
> *negative* class means representing all 50 noise candidates by their single
> **highest-scoring** member, 0.7223, against a class median of 0.0642.
>
> Best-rank-per-group is right for positives (was the source recovered at all?)
> and maximally pessimistic for negatives. Recomputed:
>
> | collapse rule | overall ROC-AUC |
> |---|---|
> | best-rank per group, all classes (as published) | 0.5773 ← distorted |
> | collapse positives only, keep all negatives | **0.7611** |
> | median per group, all classes | 0.5959 |
> | no collapsing (per-detection) | 0.7846 |
>
> **The headline pulsar-vs-RFI finding is unaffected**: 0.5735 per-source versus
> 0.6168 per-detection. Both are poor under every rule, so EXP-003's conclusion
> stands. What was distorted is the *overall* figure that mixes noise and RFI.
>
> Note also that `scripts/build_psc_manifest.py` has since been generalized and
> no longer reproduces this manifest byte-for-byte: source-group IDs are now
> zero-padded to four digits, and noise is grouped positionally like every other
> class rather than by special case. The v1 manifest is kept frozen so this
> record stays consistent with `psc_ranked_candidates.csv`.

## Hypothesis that was tested

Predicted before running: RFI would score pulsar-like, because 38/50 RFI rows
have DM = 0.0 and the model cannot see DM — the Period–DM channel was excluded
by design, so the most diagnostic feature for this population is unavailable.

**Direction confirmed, mechanism not.** RFI does score high, but the DM = 0.0
subset is not differentially responsible:

| RFI subset | n | median score |
|---|---|---|
| DM = 0.0 | 38 | 0.9717 |
| DM > 0.0 | 12 | 0.9905 |

The non-zero-DM RFI scores *slightly higher*. So the failure is not specifically
"zero-DM is invisible" — it is that the FvP/TvP representation alone does not
carry whatever distinguishes AO327 RFI from a pulsar. Adding a DM branch may
still help, but this result does not by itself establish that it would.

## Missed pulsars (score < 0.5)

| plot_id | score | rank | DM | P (ms) | chi2 |
|---|---|---|---|---|---|
| 9157474 | 0.0263 | 147 | 16.1 | 1387.534 | 5.08 |
| 3391638 | 0.0382 | 133 | 34.6 | 1797.622 | 7.32 |
| 9214140 | 0.1517 | 110 | 43.4 | 2939.411 | 24.65 |
| 9814776 | 0.1823 | 104 | 44.3 | 2167.897 | 7.28 |
| 3391597 | 0.2585 | 101 | 35.9 | 1825.962 | 6.79 |
| 9812952 | 0.2761 | 99 | 43.4 | 2124.293 | 28.55 |
| 9214234 | 0.2962 | 98 | 52.8 | 2655.315 | 22.78 |
| 9166721 | 0.4088 | 95 | 15.2 | 2720.799 | 26.75 |
| 4615826 | 0.4156 | 94 | 21.5 | 1771.728 | 33.25 |
| 3731347 | 0.4469 | 93 | 9.3 | 1382.439 | 36.77 |

These skew toward **low chi²** (weak detections) and **longer periods**. Top-ranked
positives have chi² in the hundreds to thousands. Missing weak candidates is an
expected and unalarming failure mode; it is not the interesting result here.

## Caveats — the effective sample is much smaller than 150

1. **Positives:** 50 detections but only **27 distinct (position, period) sources**;
   one source contributes 17. `4162.940 / 1387.534 = 3.0002`, consistent with a
   harmonic of that same pulsar.
2. **RFI:** 42/50 have periods in 665–675 ms and 38/50 have DM exactly 0.0 — this
   is essentially **one** zero-DM interference signal seen across 19 dates, not 50
   independent examples. The pulsar-vs-RFI result therefore generalizes to *this
   interference mode*, not to AO327 RFI in general.
3. **Noise:** all 50 share one MJD (57167) and one pointing (02:25:52 +07:01:40) —
   one noise realization. The 0.95 pulsar-vs-noise AUC rests on a single field.
4. Known pulsars here were detected by the survey, so they skew bright.
5. Scores are experimental cross-survey values, **not calibrated probabilities**.

## Excluded from the manifest

Six rows supplied under an "RFI" heading came from an individual-grading view
(grader name + timestamp columns) rather than the consensus tab, and one
(`6004822`) carried grade **Known**, not RFI. Filing it as a negative would have
placed a known pulsar in the negative class. All six are excluded pending
clarification. `build_psc_manifest.py` now hard-fails on any grade mismatch.

Note: `6004822` has DM = 0.0 and P = 33.321 ms, which is consistent with mains-
frequency interference, so its "Known" grade may itself be questionable.

## What this changes

The project's finding is no longer "does an HTRU1 CNN transfer to AO327." It is
**where** it transfers: the model rejects noise well and this RFI mode not at all.

> **Superseded in part by [EXP-003](exp003_domain_shift_diagnosis.md).** This
> section originally attributed the failure to the *input representation* being
> insufficient. EXP-003 refutes that: a duty-cycle statistic computed on the same
> preprocessed tensors separates pulsar from RFI at ROC-AUC 0.9356 versus the
> CNN's 0.6168. The information is present in FvP/TvP; the model does not use it.
> Read this as a **transfer failure**, not a representation failure. The
> measurements in this record stand; the causal interpretation does not.
