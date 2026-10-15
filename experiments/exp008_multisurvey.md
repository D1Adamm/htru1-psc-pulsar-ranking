# EXP-008 — Multi-survey training across two telescopes

EXP-006/007 established that an Arecibo-only model collapses on Green Bank and
that blending it with robust features does not repair it. This asks whether
training on both telescopes at once does.

| Field | Value |
|---|---|
| `experiment_id` | `EXP-008` |
| `date` | 2026-08-06 |
| `manifest` | `data/psc/psc_eval_multisurvey.csv` — 2,746 candidates |
| `checkpoint` | `outputs/models/psc_multisurvey_best.pt` |
| `metrics` | `outputs/metrics/psc_multisurvey_metrics.json` (combined) |
| `metrics, per survey` | `outputs/metrics/psc_multisurvey_eval_by_survey.json` |
| `split` | 60/20/20 by `source_group`, 2,308 groups |
| `best epoch` | 25 of 40 (val accuracy 0.9194) |

Training pool:

| survey | pulsar | rfi | noise | sources | nights |
|---|---|---|---|---|---|
| AO327 (Arecibo, 327 MHz) | 500 | 548 | 500 | 1,196 | — |
| GBNCC (Green Bank, 350 MHz) | 300 | 398 | 500 | 1,112 | 41 / 71 / 2 |

GBNCC RFI is DM-filtered (`--max-dm 1.0`); pulsars and noise are not.

## Headline result — combining works, and nothing is forgotten

Held-out split, broken down by telescope:

| three-class accuracy | AO327 | GBNCC |
|---|---|---|
| Arecibo-only model (EXP-005) | 0.9587 | **0.2905** |
| multi-survey model | **0.9684** | **0.8458** |

The Green Bank failure is largely repaired, and **Arecibo did not degrade — it
improved slightly.** Sequential fine-tuning would have traded one for the other;
joint training does not, because no single survey's texture is ever sufficient
when both appear in every batch.

Per-class on the held-out split:

| | AO327 recall | GBNCC recall |
|---|---|---|
| known_pulsar | 0.9130 | 0.6458 |
| rfi | 0.9904 | 0.8571 |
| noise | 0.9917 | **0.9266** |

**GBNCC noise recall went from 0.0000 to 0.9266.** The Arecibo-only model never
once assigned the noise class to any of 500 Green Bank noise candidates — its
"noise" concept was built from a single Arecibo observation (one MJD, five
pointings) and did not transfer at all.

## It finally beats the baseline on both telescopes

| binary pulsar-vs-rest | model | duty cycle |
|---|---|---|
| AO327 held-out | **0.9961** | 0.9726 |
| GBNCC held-out | **0.9155** | 0.8535 |

EXP-006 and EXP-007 both had the one-line statistic winning cross-telescope. This
is the first configuration where the CNN wins on a telescope other than the one
it was tuned for.

## Cross-telescope baseline this replaces

Scoring the full GBNCC set with the *Arecibo-only* model (before any Green Bank
data entered training) gives the cleanest statement of the original failure —
1,198 candidates, all three classes, versus EXP-006's 19 RFI:

| | value |
|---|---|
| three-class accuracy | 0.2905 (below chance for 3 classes) |
| pulsar recall | 0.9800 |
| rfi recall | 0.1357 |
| noise recall | 0.0000 |
| binary pulsar-vs-rest | 0.7828 |
| duty cycle, same set | 0.8113 (rfi) / 0.8811 (noise) |

It classified almost everything as a pulsar.

## Reproducing these numbers

The per-survey figures above were originally computed by hand, and only the
combined metrics were written to `psc_multisurvey_metrics.json` — so nothing in
the repository regenerated the 0.9684 / 0.8458 breakdown. It does now:

```bash
python scripts/evaluate_psc_multisurvey.py
```

No training, no checkpoint is written, and the HTRU1 baseline is refused
outright as a `--checkpoint` argument. It rebuilds the split with
`split_indices_by_source_group(seed=42)`, scores
`outputs/models/psc_multisurvey_best.pt`, and writes combined and per-survey
metrics to `outputs/metrics/psc_multisurvey_eval_by_survey.json`. (That JSON
is a regenerable artifact of the command above. It was gitignored when this
record was written and has been tracked since 2026-10-14.)

The rebuilt split is **not** read from the checkpoint; it is reconstructed and
then compared against the `test_indices` the training run stored, and a mismatch
aborts the evaluation rather than reporting numbers from a split that is no
longer held out. On the committed manifest and cache it matches: all 2,746 rows
load, the split is 1,645 / 558 / 543, and all 543 test indices are identical to
the checkpoint's.

### Recorded vs. reproduced

Every recorded figure reproduces exactly — there is no discrepancy to explain.

| | recorded | reproduced |
|---|---|---|
| AO327 three-class accuracy | 0.9684 | 0.9684 (306/316) |
| GBNCC three-class accuracy | 0.8458 | 0.8458 (192/227) |
| AO327 pulsar / rfi / noise recall | 0.9130 / 0.9904 / 0.9917 | identical |
| GBNCC pulsar / rfi / noise recall | 0.6458 / 0.8571 / 0.9266 | identical |
| AO327 binary, model vs duty cycle | 0.9961 / 0.9726 | identical |
| GBNCC binary, model vs duty cycle | 0.9155 / 0.8535 | identical |

They also close arithmetically against the combined metrics that *were*
committed, which is the check that the breakdown describes the same evaluation:
`0.9684 × 316 + 0.8458 × 227 = 498`, and `498 / 543 = 0.9171`, the combined
accuracy in `psc_multisurvey_metrics.json`. Likewise the per-survey pulsar
recalls give `(84 + 31) / 140 = 0.8214`, its combined `recall_known_pulsar`.

Beyond what was recorded, the run adds per-class **F1 and support**, per-survey
**confusion matrices**, and per-survey **PR-AUC** — the last of which is the more
honest number on GBNCC, where pulsars are 48 of 227:

| | AO327 | GBNCC |
|---|---|---|
| binary ROC-AUC | 0.9961 | 0.9155 |
| binary PR-AUC | 0.9915 | **0.7969** |
| pulsar F1 | 0.9438 | **0.7045** |

The GBNCC pulsar column is the weak result in caveat 2, stated two more ways.

## Caveats — read before quoting the 0.8458

1. **GBNCC is now in training, so this is not a generalization test.** The
   0.8458 is in-domain for Green Bank. It shows multi-survey training *can*
   fit two telescopes at once without forgetting; it does **not** show the model
   would handle a third telescope it has never seen. That needs a held-out
   survey, which the PSC menu has but which is not yet collected.
2. **GBNCC pulsar recall is the weak class at 0.6458** — 17 of 48 held-out Green
   Bank pulsars are missed. Arecibo contributes 500 pulsars against GBNCC's 300,
   so the pulsar class is still Arecibo-weighted.
3. **GBNCC noise spans only 2 nights** versus 41 and 71 for its pulsars and RFI.
   The 0.9266 noise recall may be narrower than it looks.
4. **The RFI DM filter is a selection.** GBNCC RFI was cut to DM < 1.0, so that
   class is physically clean but not a random sample of Green Bank interference.
5. Mild overfitting persists: train accuracy 0.99 against validation ~0.92.

## Bug found while building this

`--max-dm` was applying the dispersion cut to **every** class. On the first GBNCC
build it kept **0 of 300 pulsars** and **1 of 500 noise** — real pulsars are
dispersed by the interstellar medium (DM 20–220 here) and noise carries whatever
the search trialed. The cut is a physical validity check that is only meaningful
for RFI, where DM ≈ 0 by definition and a high-DM label is impossible. Now scoped
to the RFI category.

Separately, the two manifests were built independently and **1,039 of 1,112 GBNCC
`source_group` IDs collided with AO327 ones** (both number from `_g0000`). Left
alone, the group-aware split would have tied an Arecibo pulsar to an unrelated
Green Bank candidate and leaked across the train/test boundary. Groups are now
namespaced by survey in the combined manifest.

## What this answers

The question was whether stacking surveys would degrade the model. It does not —
provided training is **joint, not sequential**. Every run starts from the HTRU1
backbone and sees all surveys together; nothing is chained, so there is nothing
to forget.

The open question is unchanged and now sharper: does a two-survey model transfer
to a *third* telescope? EXP-006 showed one survey does not generalise to two.
Whether two generalise to three is the experiment that would actually establish
the method.
