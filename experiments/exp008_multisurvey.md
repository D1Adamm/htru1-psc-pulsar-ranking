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
| `metrics` | `outputs/metrics/psc_multisurvey_metrics.json` |
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
