# EXP-007 — Does combining the CNN with simple features rescue generalisation?

EXP-006 left the CNN at 2/19 on an unseen telescope while a duty-cycle threshold
scored 19/19. The obvious next move is to combine them. This tests whether that
works.

| Field | Value |
|---|---|
| `experiment_id` | `EXP-007` |
| `date` | 2026-08-06 |
| `script` | `scripts/build_hybrid_ranker.py` |
| `metrics` | `outputs/metrics/exp007_hybrid_ranker.json` |
| `model` | EXP-005 three-class checkpoint (unmodified) |
| `fit on` | Arecibo train+val split, by `source_group` |
| `tested on` | Arecibo held-out pulsar-vs-RFI (n=221); GBNCC RFI (n=19) |

Features, all oriented so larger = more pulsar-like: the three-class CNN's
collapsed pulsar score, negated duty cycle, and negated DM-curve peak fraction.
A logistic regression is fitted on the Arecibo training split only.

## Result — combining does not help

| features | Arecibo pulsar-vs-RFI | GBNCC 19 RFI |
|---|---|---|
| CNN only | **0.9984** | 2 / 19 |
| duty cycle only | 0.9693 | **19 / 19** |
| DM only | 0.9169 | 19 / 19 *(confounded)* |
| CNN + duty | 0.9936 | 2 / 19 |
| duty + DM | 0.9783 | 19 / 19 *(confounded)* |
| CNN + duty + DM | 0.9936 | 2 / 19 *(confounded)* |

**Every combination containing the CNN collapses to 2/19.** Adding robust
features does not repair it.

The fitted coefficients explain why. In `CNN + duty`:

```
cnn_score      = +3.554
neg_duty_cycle = +1.895
```

The regression trusts the CNN nearly twice as much as duty cycle — a correct
decision on Arecibo, where the CNN is near-perfect, and the wrong one everywhere
else. The CNN is not merely wrong on Green Bank; it is *confidently* wrong
(several predictions above P(pulsar) = 0.99), so a linear combination fitted
in-domain inherits that confidence.

## The honest comparison

Only two rows are free of the DM-selection confound (below), and they are the
ones that matter:

| | Arecibo | GBNCC |
|---|---|---|
| CNN only | 0.9984 | 2 / 19 |
| duty cycle only | 0.9693 | 19 / 19 |

**Dropping the CNN costs 0.029 in-domain and gains everything out-of-domain.**
For a reviewer-facing triage tool that will meet interference the training set
never contained, the one-line statistic is the better component.

## Selection confound

The GBNCC RFI set was selected by DM < 1.0. Any DM-using variant scores 19/19
there by rediscovering that filter, not by doing astronomy. Those rows are marked
*(confounded)* and must not be quoted as evidence that DM generalises. The
Arecibo column is unaffected — that set was never DM-selected.

An honest GBNCC test of a DM-aware model needs a non-DM-selected GBNCC sample.

## Incidental finding: a resampling bug, caught by a test

`dmc_to_vector` originally resampled DM curves with `np.interp`. Downsampling a
401-point curve to 64 points that way can step **straight over a narrow peak** —
a synthetic single-bin spike at index 100 resampled to a vector of all zeros,
losing the signal completely.

Fixed to use per-bin maxima when downsampling. Real DM curves are broad enough
that this rarely bit, but for a curve whose peak location *is* the signal, silently
dropping peaks is not acceptable. Verified on real cached curves afterwards:
GBNCC 60 Hz RFI → peak_fraction 0.000; Arecibo pulsar → 0.360.

## Caveats

1. **GBNCC n = 19, RFI only.** No pulsars or noise from that telescope, so only
   recall can be measured there, not AUC.
2. **Linear combination only.** A non-linear combiner, or a rule that defers to
   duty cycle when the CNN is out of distribution, was not tested and might do
   better. What is established is that a straightforward fitted blend does not.
3. The `neg_peak_fraction` feature is misnamed — the fit shows the pulsar-positive
   direction is *higher* peak fraction (pulsars are dispersed), so the negation
   was unnecessary. Logistic regression absorbs the sign, so no number changes.

## Implication for EXP-008

The DM branch is still worth building — DM is the physically correct
discriminator for mains RFI — but EXP-007 lowers the expected payoff of bolting
it onto *this* CNN. The failure is not missing information; it is that the CNN
has learned survey-specific texture and states it confidently. A DM input may
well be learned the same way.

The stronger design is likely **train across surveys** (EXP-009), so no single
survey's texture is ever sufficient, rather than adding another input to a
single-survey model.
