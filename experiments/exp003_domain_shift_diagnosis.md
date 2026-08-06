# EXP-003 — Why the model cannot separate AO327 pulsars from RFI

Diagnostic follow-up to [exp002_psc_cross_survey_ranking.md](exp002_psc_cross_survey_ranking.md).
Read-only: no retraining, no refetching, no architecture change. EXP-001 and
EXP-002 artifacts untouched.

| Field | Value |
|---|---|
| `experiment_id` | `EXP-003` |
| `date` | 2026-08-05 |
| `model` | EXP-001 checkpoint, sha256 `7f852a3e7fc46ade…` |
| `data` | 150 cached PSC JSON in `data/psc/raw/` + HTRU1 `test_batch` |
| `script` | `scripts/diagnose_psc_domain_shift.py` |
| `outputs` | `outputs/figures/exp003_category_comparison.png`, `outputs/metrics/exp003_domain_shift.json` |

## Verdict

**The input representation is fine. The model is the problem.**

A single hand-crafted feature computed from the *same* preprocessed
`(2, 32, 32)` tensors the CNN receives separates pulsars from RFI far better
than the CNN does:

| Discriminator | pulsar vs RFI | pulsar vs noise |
|---|---|---|
| duty cycle of the phase profile | **0.9356** | **0.9688** |
| clutter (fraction of bright pixels) | 0.7674 | — |
| **BaselineCNN score** | **0.6168** | 0.9524 |

Mean duty cycle: pulsar **0.118**, RFI **0.429**, noise **0.532**. Pulsars are
compact in phase; this RFI is broad. That difference survives the resize and the
normalization intact — the CNN simply does not use it.

This is a **transfer failure**, not a representation failure.

## Hypotheses tested and eliminated

### 1. Axis/orientation transpose — REFUTED

Both surveys are 32×32 after resize, so a transpose would pass every shape
assertion in the codebase. Checked empirically by measuring which axis carries
the narrow pulse feature:

| | phase axis | column peakiness | row peakiness |
|---|---|---|---|
| HTRU1 channels 1–2 | 1 (columns) | 4.61 | 0.99 |
| PSC `FvP` | 1 (columns) | 7.11 | 3.08 |

**Axes match.** No transpose is needed and EXP-002 stands on this point.

### 2. Phase-bin count artifact — REFUTED by within-class control

Phase-bin count is badly confounded with class in this sample:

| bins | n | composition | mean score |
|---|---|---|---|
| 24 | 6 | all noise | 0.216 |
| 50 | 45 | 43 noise, 2 RFI | 0.140 |
| 100 | 3 | 2 RFI, 1 noise | 0.622 |
| 200 | 96 | 50 pulsar, 46 RFI | 0.884 |

Score predicts "has 200 bins" (AUC 0.9502) *better* than it predicts "is a
pulsar" (AUC 0.7846), which initially looked like the model keying on a
resampling artifact rather than astrophysics.

The within-class control refutes it. Holding class fixed and varying bin count:

- RFI at 200 bins: mean score **0.9446**
- RFI at 50 bins: mean score **0.9372** (gap 0.007)
- Noise scores low at 24, 50, *and* 100 bins

Bin count does essentially nothing within a class. The aggregate correlation is
sample confounding, not model behavior. **Recorded because it is a trap worth
not falling into twice.**

### 3. Normalization mismatch — REFUTED

Re-scored all 150 cached candidates under three normalizations, model and
everything else held fixed:

| normalization | pulsar vs RFI | pulsar vs noise |
|---|---|---|
| per-channel min–max (shipped) | 0.6168 | 0.9524 |
| global min–max over both channels | 0.6556 | 0.8796 |
| per-row (sub-band) baseline subtraction | 0.6104 | 0.9712 |

No variant rescues pulsar-vs-RFI. The spread (0.610–0.656) is small and none
approach the 0.9356 the duty-cycle feature reaches on identical inputs.

*(A per-channel z-score variant scored 0.1844, but that pushes inputs outside the
`[0, 1]` range the model was trained on, so it measures breakage, not signal.
Excluded from the table.)*

### 4. Representation insufficient — REFUTED

This was the leading hypothesis entering EXP-003 and the duty-cycle result kills
it. See Verdict above.

## Visual evidence

`outputs/figures/exp003_category_comparison.png` shows preprocessed FvP/TvP for
the top-scoring pulsars, top-scoring RFI, the 10 missed pulsars, and noise.

- **Pulsars:** narrow bright vertical stripe, dark clean background.
- **RFI:** broad, often slanted or curved bright band, cluttered background,
  frequently with horizontal banding in TvP.
- **Missed pulsars:** no visible pulse at all — pure speckle. The model is not
  wrong on these; the signal genuinely does not survive downsampling to 32 phase
  bins from 200. This is a *preprocessing* limit on weak candidates, distinct
  from the RFI problem.
- **Noise:** speckle with horizontal banding.

The pulsar/RFI distinction is obvious by eye, which is what motivated the
duty-cycle test.

## Why the model fails despite the signal being present

HTRU1 is Parkes at 1.4 GHz; AO327 is Arecibo at 327 MHz. HTRU1's negatives
certainly contain RFI, but 327 MHz interference has different morphology — the
broad slanted bands visible in the figure are not what the model learned to
reject. The CNN learned "bright coherent vertical structure ⇒ pulsar," which is
correct at Parkes and badly wrong here.

## Caveats

1. **The duty-cycle feature was chosen after inspecting the figure.** It is
   post-hoc on this sample and has not been validated on held-out data. The
   effect size is large (0.94 vs 0.62) but it should be confirmed on new
   candidates before being quoted as a baseline.
2. Effective sample is small: 27 distinct pulsar sources, essentially **one**
   ~671 ms zero-DM interference mode, and **one** noise field. Conclusions apply
   to this interference mode, not AO327 RFI generally.
3. Scores remain uncalibrated cross-survey values.

## What this changes for the project

The EXP-002 write-up concluded this was "a measured negative result about the
input representation." **That conclusion is now retracted** — the representation
carries the distinction; the model fails to exploit it.

The implied next step changes accordingly:

- **Was:** add a DM-curve branch so the model can see dispersion.
- **Now:** fine-tune or domain-adapt on labeled PSC data, since the features the
  model needs are already in its input. A DM branch may still help, but it is no
  longer the indicated fix.

A useful framing for the writeup: an HTRU1-trained CNN transferred to AO327 is
outperformed at pulsar-vs-RFI by a trivial duty-cycle statistic on its own
inputs. That is a concrete, quantified statement about the limits of naive
cross-survey transfer.
