# EXP-006 — Does the fine-tuned model generalise to a third telescope?

EXP-005 fine-tuned on Arecibo (AO327) and reached 0.998 on pulsar-vs-RFI, with
the caveat that it had only seen one survey's interference. This tests that
caveat directly on **GBNCC — Green Bank Telescope, ~350 MHz** — a telescope
neither model has ever seen.

| Field | Value |
|---|---|
| `experiment_id` | `EXP-006` |
| `date` | 2026-08-06 |
| `test set` | `data/psc/psc_eval_gbncc_rfi.csv` — 19 candidates |
| `source` | `data/psc/source/gbncc/rfi.tsv` (400 rows, cut to DM < 1.0) |
| `models` | EXP-001 binary checkpoint; EXP-005 three-class checkpoint |
| `training exposure to GBNCC` | **none** |

## The test set is unusually trustworthy

400 GBNCC RFI rows were cut to DM < 1.0, leaving 19. Terrestrial interference
has zero dispersion by definition, so this cut selects physically unambiguous
RFI rather than relying on crowd-sourced grades.

What survived is even cleaner than that: **14 of 19 sit at 60.0 Hz**, plus one at
30.0 Hz and one at 20.0 Hz. These are mains-electricity harmonics. The labels are
certain in a way no other set in this project can claim.

## Result — the CNN does not generalise

| method | correct on 19 GBNCC RFI |
|---|---|
| duty-cycle statistic | **19 / 19** |
| three-class CNN (fine-tuned on Arecibo) | **2 / 19** |
| HTRU1 binary CNN (never fine-tuned) | 2 / 19 |

The three-class model classified **17 of 19 mains-interference signals as
pulsars**, several above P(pulsar) = 0.99. Its RFI recall fell from **0.982 on
Arecibo to 0.105 on Green Bank**.

**Fine-tuning bought nothing here.** The Arecibo-adapted model and the untouched
HTRU1 model both scored 2/19. Whatever the fine-tuning learned, it did not
transfer one telescope further.

## The information was in the images the whole time

The duty-cycle threshold was chosen using **only Arecibo data** (0.285, giving
0.927 accuracy there), then applied blind to Green Bank. It rejected **19 of 19**.

| | median duty cycle |
|---|---|
| Arecibo pulsars | 0.094 |
| Arecibo RFI | 0.469 |
| GBNCC 60 Hz RFI | **0.500** |

The Green Bank interference is just as broad in phase as Arecibo interference —
**0 of 19 are narrower than the median Arecibo pulsar**. A one-line statistic
sees this immediately. The neural network, given the same 32×32 inputs, does not.

## Why the CNN fails

The training set contained essentially no short-period RFI:

- Arecibo RFI periods: 16.7 – 13,459.5 ms, with only **2 of 548 below 60 ms**
- GBNCC RFI periods: 14.4 – 50.0 ms — entirely outside that range

So the model is extrapolating, not interpolating. EXP-005's period-matched
control showed it does not lean on period *within* its training range; that
result stands and is not contradicted here. What EXP-006 adds is that the model
collapses *outside* that range, and the mains-harmonic RFI that dominates Green
Bank lives there.

## What this does to EXP-005

EXP-005's numbers are not wrong, but their scope is now measured rather than
assumed. **0.998 was an Arecibo number.** The honest statement is:

> Fine-tuning on one survey's labels produces near-perfect performance on that
> survey and does not transfer to another telescope.

That is a sharper and more useful finding than the 0.998 alone, and it is the
kind of result that only shows up when someone actually tests it.

## What it does to the duty-cycle baseline

The duty-cycle statistic has now beaten the CNN twice, in different ways:

- **EXP-003** — 0.936 vs 0.617 before fine-tuning
- **EXP-006** — 19/19 vs 2/19 across telescopes

After fine-tuning the CNN wins decisively *in-domain* (EXP-005: 0.9957 vs
0.9688). Out of domain, it loses completely. The simple feature is less accurate
where it has been trained and far more robust where it has not — which is the
classic shape of a model that has memorised survey-specific texture.

## Caveats

1. **n = 19.** Small. But 19/19 against 2/19 is not a marginal effect.
2. **RFI only.** GBNCC pulsars and noise were not collected, so no ROC-AUC or
   three-class accuracy can be computed here — only RFI recall.
3. **One RFI type.** Mains harmonics are one family of interference. This does
   not characterise GBNCC interference in general.
4. The DM < 1.0 cut was the owner's judgment that higher-DM rows in the GBNCC
   RFI tab are mislabelled. The full 400-row export is kept in
   `data/psc/source/gbncc/rfi.tsv` so the cut can be revisited.

## Implications

- A per-survey fine-tune is not a general RFI detector. Any deployment should be
  scoped to the survey it was trained on, or trained on several surveys at once.
- **The DM curve is now clearly worth adding.** For this RFI population DM = 0 is
  the definitive discriminator, and it is the channel the project excluded from
  the start. EXP-003 concluded a DM branch was not the indicated fix for
  *Arecibo's* 671 ms RFI; for Green Bank's mains RFI it plainly is.
- A hybrid ranker — CNN score plus duty cycle plus a DM ≈ 0 filter — would have
  caught all 19 of these. That remains the cheapest real improvement available.
