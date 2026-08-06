# EXP-005 — Three-class fine-tuning on PSC AO327

Acts on the EXP-003 diagnosis. That experiment showed the HTRU1 model could not
separate AO327 pulsars from RFI and that the failure was model transfer rather
than the input representation. This fine-tunes on PSC labels and asks whether
that fixes it.

| Field | Value |
|---|---|
| `experiment_id` | `EXP-005` |
| `date` | 2026-08-06 |
| `backbone` | EXP-001 HTRU1 checkpoint, sha256 `7f852a3e7fc46ade…` (unmodified) |
| `manifest` | `data/psc/psc_eval_candidates_v3.csv` — 1,548 candidates |
| `script` | `scripts/finetune_psc_threeclass.py` |
| `checkpoint` | `outputs/models/psc_threeclass_best.pt` |
| `metrics` | `outputs/metrics/psc_threeclass_metrics.json` |
| `seed` | 42 |
| `split` | 60/20/20 by `source_group` — 929 train / 304 val / 315 test |
| `best epoch` | 25 of 40 (val accuracy 0.9693) |

**Scope note.** Three-class was out of scope until 2026-08-05, when the owner
reversed that decision. The HTRU1 model remains binary; HTRU1 has no RFI labels,
so the three-way head is trained *entirely* on PSC data on top of the frozen-in-
place HTRU1 convolutional features. See the scope-change section of `CLAUDE.md`.

## Headline result

Fine-tuning fixes the RFI failure.

| pulsar vs RFI | before (EXP-002/003 model) | after fine-tuning |
|---|---|---|
| all candidates | 0.7534 | **0.9984** |
| period-matched band | 0.7185 | **0.9949** |

Three-class performance on the held-out split (n = 315):

| class | recall | precision |
|---|---|---|
| known_pulsar | 0.9455 | 0.9369 |
| rfi | **0.9820** | 0.9820 |
| noise | 0.9468 | 0.9570 |

Three-class accuracy **0.9587**. Binary-collapsed pulsar-vs-rest ROC-AUC
**0.9957**, versus **0.9688** for the duty-cycle baseline on the identical
split — the first time the CNN has beaten the one-line statistic.

RFI recall went from "cannot reject it at all" to 0.982. That is the finding.

## The period confound was checked, not assumed

The three classes sit in nearly disjoint period bands in this sample, so a model
could score well by learning period rather than morphology. Period alone, used
as a classifier, reaches **ROC-AUC 0.9561** on this test split — so the concern
was real, not hypothetical.

The model is not doing that:

| | ROC-AUC |
|---|---|
| all test candidates | 0.9957 |
| restricted to the 250–5,810 ms band, where period carries no information | **0.9949** |
| period alone as a classifier | 0.9561 |

Performance essentially does not drop inside the band where the shortcut is
unavailable, and the model beats the shortcut outright. The learning is real.

## Caveats

1. **The period-matched check rests on 18 RFI candidates.** Bootstrap 95% CI on
   that number is [0.9830, 1.0000]. The point estimate is solid but the negative
   class is small; this is the single weakest link in the result.
2. **Label noise is unquantified.** PSC grades are crowd-sourced and the owner
   reports mislabeled candidates in the tails of each tab. A three-class model
   must learn the RFI/noise boundary that this noise corrupts most directly; a
   binary model would have been indifferent to it. The `psc_count` column
   (number of graders) is captured in the manifest but is not yet used as a
   confidence filter.
3. **Sampling is still not representative.** The 1,548 candidates come from
   period-sorted tab exports (v1 + v2 combined), not a uniform draw. The RFI seen
   here is two populations — a ~671 ms zero-DM mode and a long-period high-DM
   mode — not AO327 interference in general.
4. **Mild overfitting.** Train accuracy reached 1.000 while validation plateaued
   at ~0.966. The best-validation checkpoint is the one saved; validation did not
   degrade, so this is acceptable, but the ~262k-parameter head on 929 training
   samples leaves little margin.
5. **Not a validated detector.** Scores remain uncalibrated. This is decision
   support with documented limits, unchanged from the original scope.

## What changed for the project

EXP-003 concluded the two-channel representation carries the information and the
model fails to use it. EXP-005 confirms that directly: given PSC labels, the same
architecture on the same inputs learns the distinction and reaches 0.995.

The open question is no longer "does the model work" but "does it generalise
beyond this sample." Answering that needs RFI diversity the PSC interface does
not currently make it possible to collect (see EXP-004, parked).

Natural next steps, none blocking:

- Filter by `psc_count` to build a high-consensus subset and retrain; compare.
- Re-run with a frozen backbone (`--freeze-backbone`) to see how much of the gain
  is the HTRU1 features versus the new head.
- Score the original EXP-002 150-candidate set with this model for a like-for-like
  comparison against the numbers already published in that record.
