# Poster assets

Generated 2026-10-15 from repository commit `be1f7f6` (branch
`track-experiment-outputs`). No code, model, or result in the repository was
changed to make these.

## New figures: real PSC candidates, native resolution

| File | Content |
|---|---|
| `psc_7914553_noise_FvP.png`, `psc_7914553_noise_TvP.png` | Noise, PSC plot 7914553 |
| `psc_4502281_rfi_FvP.png`, `psc_4502281_rfi_TvP.png` | RFI, PSC plot 4502281 |
| `psc_9886890_pulsar_FvP.png`, `psc_9886890_pulsar_TvP.png` | Pulsar, PSC plot 9886890 |
| `psc_fvp_tvp_comparison.png` | Combined figure: columns Noise / RFI / Pulsar, rows FvP / TvP |

All PNGs are 300 DPI. Individual images are 1800×1500 px; the combined figure is 4500×2850 px.

### How they were made
- **Source:** the `FvP` and `TvP` arrays from
  `https://pulsars.nanograv.org/api/psrsearch/plots/{plot_id}/json`.
  - 9886890 was read from the project cache (`data/psc/raw/`).
  - 7914553 and 4502281 were fetched on 2026-10-15.
- **What is plotted:** the arrays exactly as served.
  - **No changes to the data:** no resizing, smoothing, clipping, or renormalisation. PSC already serves them scaled to [0, 1].
  - **Shared color scale:** every panel uses the same fixed 0–1 viridis scale, so colors can be compared across panels.
  - **One pixel per value:** `interpolation="nearest"`, so each pixel is one real phase bin × sub-band (or sub-integration).
- **Axes:**
  - x is pulse phase over one folded period.
  - y is the sub-band or sub-integration index. These are indices rather than MHz or seconds, because the row orientation in the PSC arrays was not independently confirmed.
- **Script:** `make_poster_figures.py --json-dir <dir containing {id}.json>`.

These are not the CNN's inputs. The model sees these arrays resized to 32×32 and min–max normalised per channel. `exp003_category_comparison.png` shows that preprocessed view.

### Candidate metadata (from each JSON's `meta` block)

| Plot ID | Class | Telescope | Period | DM | sigma | FvP shape | TvP shape |
|---|---|---|---|---|---|---|---|
| 7914553 | Noise | Arecibo (AO327) | 58.7 ms | 1032.3 | 3.2 | 64×100 | 32×100 |
| 4502281 | RFI | Arecibo (AO327) | 41.7 ms | 0.0 | 27.5 | 128×50 | 48×50 |
| 9886890 | Pulsar | Arecibo (AO327) | 4162.9 ms | 15.5 | 113.9 | 64×200 | 16×200 |

### Caveats for interpreting the images
- **Pulsar 9886890 shows three pulses per window.**
  - It was folded at 4162.9 ms, which is 3× the pulsar's true period of about 1387.5 ms.
  - EXP-002 records it as a harmonic detection of that pulsar.
  - It is a real, PSC-graded known-pulsar detection, but the three stripes are one pulsar seen three times per fold, not three pulses per rotation.
- **9886890's first time sub-integration (TvP row 0) is all zero** in the served data, which appears as a dark band at the bottom of its TvP panel. This is how the data was served, not a plotting artifact.
- **Where the labels come from:**
  - 9886890 is in the project's labeled manifests (PSC Known Pulsars tab, consensus grade Known).
  - The Noise and RFI labels for 7914553 and 4502281 were supplied for this figure. Those two candidates are not in any project manifest and were not used in any experiment.
  - Their metadata is consistent with the labels: 4502281 has DM = 0, as terrestrial RFI should; 7914553 has very low significance (sigma 3.2).

## Copied existing results (byte-identical to `outputs/`)

| File | Original path | Experiment |
|---|---|---|
| `exp003_category_comparison.png` | `outputs/figures/` | EXP-003: preprocessed 32×32 inputs for top pulsars, top RFI, missed pulsars, top noise (AO327 v1, HTRU1 model scores) |
| `psc_multisurvey_eval_by_survey.json` | `outputs/metrics/` | EXP-008: combined and per-survey (AO327 / GBNCC) held-out metrics |
| `psc_threeclass_metrics.json` | `outputs/metrics/` | EXP-005: AO327-only three-class model, held-out metrics |
| `htru1_test_metrics.json` | `outputs/metrics/` | EXP-001: HTRU1 baseline on the untouched test batch |

The raw PSC JSON is not included here: candidate data access is restricted to authorized PSC participants.
