# EXP-009 — Pre-specified CNN pilot protocol

Frozen before any EXP-009 CNN training. It asks whether a CNN learns pulsar
characteristics that generalize across independent astronomical sources, rather
than dataset shortcuts. Nothing here reports a CNN result.

| Field | Value |
|---|---|
| `experiment_id` | `EXP-009` |
| `status` | Protocol frozen; no CNN trained |
| `base commit` | `291a80a` |
| `manifest` | `data/psc/psc_eval_multisurvey.csv` (2,746 rows, unchanged) |
| `identities` | `data/psc/identity/source_identity_v1.csv`, from ATNF v2.9.0 |
| `folds` | `data/psc/folds/folds_v1.json`, seeds 5, 44, 52 |
| `pretrained` | `outputs/models/baseline_cnn_best.pt`, sha256 `7f852a3e…`, read-only |

## Claims

| Claim | Question | Main evidence |
|---|---|---|
| A. Image learning | Does it use image structure on unseen sources? | CNN vs phase-coherence control, and vs duty cycle |
| B. Cross-survey transfer | Does it work on a survey it never trained on? | Leave-one-survey-out vs duty cycle |
| C. Added ranking value | Does it improve ranking when combined with simple features? | Combined vs features alone |
| D. Deployment readiness | Is it ready for real review queues? | Not assessable with current data |

Metadata scores show how confounded the dataset is. A CNN does not have to beat
them to be useful.

## Experiments

All arms use the EXP-008 recipe: `ThreeClassCNN`, Adam at lr 1e-3, batch size 32,
40 epochs, class-weighted cross-entropy, best epoch by validation accuracy, on CPU.

| Arm | Initialization | Trained on | Tested on | Pilot / full runs |
|---|---|---|---|---|
| E1 | HTRU1, fine-tuned | Both surveys | Unseen folds | 5 / 15 |
| E2 | Random | Both surveys | Unseen folds | 5 / 15 |
| E3 | HTRU1 convolutions frozen | Both surveys | Unseen folds | 5 / 15 |
| C1 | Random, row-shuffled inputs | Both surveys | Unseen folds, same shuffling | 5 / 15 |
| L1 | HTRU1, fine-tuned | One survey | Other survey | 2 / 6 |
| L2 | Random | One survey | Other survey | 2 / 6 |

Only 4,944 of 267,603 parameters are convolutional, and the HTRU1 checkpoint comes
from epoch 1, so E3 freezes very little.

## Data handling

- **Folds:** test fold k, validation fold k+1, training on the other three. One
  source, one observation, and one RFI night each stay in a single fold. Noise is
  split by observation only, because each survey's noise comes from one night.
- **Cross-survey:** validation is fold 0 of the training survey. The 2 pulsars seen
  in both surveys (J0304+1932, J1136+1551) are removed from both sides.
- **Uncertain labels:** 9 rows are excluded from training and epoch selection but
  stay in primary evaluation, with an exclusion analysis reported too. They were
  flagged partly for showing no clear pulse, so removing them would flatter any image
  method: removing them and applying the identity merges raises AO327 duty on the
  CNN input from 0.969 to 0.997.
- **Harmonics:** non-fundamental folds stay in the pulsar class. Fundamental-only
  results are reported separately (242 rows, 35 sources per survey), excluding
  fundamental-period rows detected at the wrong DM.

## Identity decisions

Recorded in `data/psc/identity/manual_decisions.csv`. Plot inspection was mine and is
not an expert review.

| Case | Decision | Evidence |
|---|---|---|
| Row 3013353 | Split into J1926+1648, away from J1935+1616 | Exact 1/3 harmonic, DM matches, 2.1° from J1935+1616 |
| J2252+2455 / J2251+24, J1132+2513 / J1132+25, J2214+5357 / J2213+53 | Same pulsar in each pair | Periods agree to 0.03% or better, DM within 0.5 |
| Rows 8880312, 6454363 | Merged into J1136+1551, J2317+2149 | Same observation; slanted pulses suggest a wrong-DM fold |
| J1645-0317, J0843+0719, J1854+36, J1632-1013, J1623-2631 | Accepted | Period and DM match, with at least one fundamental row each |
| J1022+1001, J1921+1948, J1651-1709 | Accepted | Sky and DM agree |
| J1746+2245, J1848-1414, J1848-1952, J0109+11, J1915+1714_P | Own group, no identity claimed | Evidence too weak |
| 2359664, 2926640, 2903877, 2555102, 2917025 | Kept as unmatched pulsar rows | Some possible signal, no reason to reject |
| 2798868, 2302283, 2317487, 7241578, 2302282, 2920296, 9063165, 7095671, 2302215 | Label-uncertain | No clear coherent pulse in inspection |

Result: 102 source groups (42 AO327, 62 GBNCC, 2 shared) and 83 named pulsars.

## Comparators

- **Duty on the CNN input** is the primary image comparator, recomputed for any
  preprocessing variant. Beating it means the CNN got more from the same resized
  input. It uses only the FvP channel; the CNN also sees TvP.
- **Native-resolution duty** is reported alongside. Only beating this supports saying
  the CNN outperforms the simple statistic.
- **Forest baselines** (period + DM, geometry + acquisition, sigma + chi², all
  metadata, duty, duty + DMc) are context. DMc peak position may partly reflect how
  the search range was set.

## Phase-coherence control

C1 shuffles phase bins independently within each row at native resolution. Measured
pulsar-vs-RFI separation, real → shuffled, AO327 / GBNCC:

| Statistic | AO327 | GBNCC | Reading |
|---|---|---|---|
| Adjacent-row correlation, FvP | 0.741 → 0.506 | 0.714 → 0.500 | Removed |
| Mean-profile duty, FvP | 0.977 → 0.606 | 0.814 → 0.589 | Mostly removed |
| Blank time rows | 0.953 → 0.952 | 0.507 → 0.507 | Kept |
| Per-row bright fraction, TvP | 0.941 → 0.909 | 0.658 → 0.604 | Survives |
| Pixel spread, TvP | 0.988 → 0.990 | 0.723 → 0.681 | Survives |

It is a coherence control, not a complete morphology control. A high C1 score is not
a failure; the real-minus-control gap measures what coherent structure adds. On
AO327 the gap has little room, so **GBNCC decides the control contrast**. Trained E1
and E2 are also scored on row- and whole-array-shuffled test images.

## Metrics

Defined before training, per source:

- **Typical-detection AUC:** each source's median row score against all negative rows.
- **FPR at 90% of sources:** each source's best row sets the point where 90% of sources
  are found; ties are broken randomly and averaged.
- **One-detection sensitivity:** one random row per source (averaged over draws), and
  the highest-sigma row. If a claim holds only for typical-detection AUC, it is
  reported as sensitive to repeats.

Primary negatives are RFI. Noise-only results are provisional and support no claim.

## Provisional bars

Typical-detection AUC, pulsar vs RFI, all 800 pulsar rows in evaluation. Forest
values are refit at implementation.

| Endpoint | Duty, CNN input | Duty, native | HTRU1 scores |
|---|---|---|---|
| AO327 | 0.969 | 0.967 | 0.540 |
| GBNCC | 0.791 | 0.858 | 0.375 |
| AO327 as unseen target | 0.967 | 0.966 | 0.520 |
| GBNCC as unseen target | 0.787 | 0.855 | 0.363 |

`python scripts/evaluate_exp009.py --stage baseline-check` reproduces these.

## Statistics

1. All models score the same out-of-fold rows, without refitting during bootstrap.
2. Each metric is computed per assignment, then averaged. Scores are never averaged
   across models, which would measure an ensemble.
3. Resampling is stratified by class, drawing sources, RFI nights, and noise
   observations, with 2,000 draws shared across models so differences are paired. A
   sensitivity analysis draws whole linked fold components.
4. Assignments reuse the same rows, so they are not independent samples, and fold
   scores are never treated as n = 15. Total variance is bootstrap variance plus
   (1 + 1/3) × between-assignment variance.
5. Final intervals are 98.3% (Bonferroni across three arms), tested in the order
   Claim A, B, C. Everything else is descriptive.

## Added ranking value

For each run, a logistic regression with a survey indicator is fit on validation rows
only and applied to the test fold: baseline alone vs baseline + CNN. Neither split
trained that CNN, avoiding the EXP-007 leak. Near-perfect baselines leave little
room, so a null there does not count against the CNN.

## Go/no-go rules

One seed and one assignment leave out model-seed variation, so the pilot cannot declare
failure. The 0.04 allowance reflects forest differences across assignments of up to
0.043; CNN variation could be larger.

**Pilot metric:** E1 or E2 minus C1, typical-detection AUC, GBNCC pulsar vs RFI,
unadjusted 95% paired interval.

| Pilot result | Rule | Action |
|---|---|---|
| Technical failure | Leak check fails, training unstable, or scores do not reproduce | Fix and rerun; no scientific meaning |
| Clear signal | Lower bound above +0.04 | Run the full study |
| Inconclusive | Interval spans the allowance | Run the fixed extension |
| Provisional negative | Upper bound below +0.04 | Run the fixed extension; hold preprocessing |

**Fixed extension:** E1, E2, C1 on assignments 1–2 (30 runs), plus L1, L2 with 2 more
seeds per direction (8 runs), with no protocol changes.

**After three assignments:**

| Claim | Success | Genuine failure | Inconclusive |
|---|---|---|---|
| A | On GBNCC beats C1 and duty on the CNN input (lower bound above 0); on AO327 within 0.03 of duty; fundamental and coherence checks agree; GBNCC FPR difference upper bound at most +0.10 | Upper bound below +0.02 against C1 on GBNCC, for every arm | Anything else |
| B | At least matches duty on the CNN input in both directions (within 0.03), and beats it in at least one | Clearly below duty in both directions | Anything else |
| C | Clearly improves on duty or duty + DMc | No gain with a tight interval | Wide interval or baseline at ceiling |
| D | Not assessable | Not assessable | Not assessable |

If results are still inconclusive, the study stops and reports that. No more runs on
the same data; the next step is new data. Any protocol change after seeing results
makes what follows exploratory until tested on new data.

## Order of work

1. Refit forest baselines.
2. Run one real fold, and record its time and outputs.
3. Run the remaining 23 pilot runs, and verify protected hashes.
4. Evaluate the pilot and apply the rule above.
5. Run the extension or full study, then re-evaluate.
6. Run preprocessing arms, one change at a time, after the pilot: antialiasing,
   resizing (first priority, since native duty beats 32-bin duty on GBNCC),
   normalization, HTRU1 alignment, masked-row fill, row weighting. These need a
   separate review.
7. Collect new sources and nights, and test the frozen model on them.

## Catalogue provenance

Source identities were matched against the ATNF Pulsar Catalogue (Manchester et al.
2005, AJ 129, 1993). The catalogue is not committed.

| Item | Value |
|---|---|
| Catalogue version | 2.9.0, 4,503 pulsars, as stated in the file header |
| `psrcat.db` sha256 | `bac32471fef3e60cfd734f1db85503a9bd0a565018cb98edff54475f87eb642f` |
| `psrcat.db` size | 10,548,603 bytes |
| Source package | `psrcat_pkg.tar.gz`, from `https://www.atnf.csiro.au/research/pulsar/psrcat/downloads/` |
| Package sha256 | `55daa29f71f344c8de4064969b5812d6cb3a9b851efae0bef004ac9d5f6740c0` |
| Package size | 1,249,005 bytes |
| Downloaded | 2026-10-30, server last-modified 2026-10-07 |
| Local path (ignored by git) | `data/raw/atnf/psrcat_v2.9.0.db` |

**Reproducibility limitation.** The package came from the "current" download link,
whose contents can change with no change of URL. The archived
`psrcat_pkg.v2.9.0.tar.gz` is a different size (1,233,779 bytes), and it was not
downloaded, so whether it holds an identical `psrcat.db` is unknown. Consequences:

- A fresh clone cannot rebuild the identities unless it has a `psrcat.db` with the
  hash above. `build_source_identity.py` stops if the hash differs, rather than
  silently producing different identities.
- Training and evaluation do not need the catalogue. They use the committed
  `source_identity_v1.csv` and `folds_v1.json`, which remain the record of what was
  used.
- Rebuilding also needs the untracked PSC JSON cache in `data/psc/raw/`.
- If the exact `psrcat.db` is lost, identities can only be re-derived from a different
  catalogue file, and any differences must be reviewed before using them.

## Limitations

- Label decisions rest on non-expert plot inspection, and the 9 uncertain rows need an
  experienced reviewer.
- Each survey's noise comes from one night.
- GBNCC RFI is DM-selected, and AO327 RFI in the pulsar period band is mostly one
  ~671 ms mode.
- AO327 is near its ceiling.
- HTRU1 may contain some of the same pulsars, which cannot be measured, so E2 matters.
- All assignments reuse the same rows. True confirmation needs new sources and nights.
- PSC authorization for the cached data is not documented in the repo.
