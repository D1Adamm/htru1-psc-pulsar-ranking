# Pulsar 499: cross-survey pulsar candidate ranking

A research project on whether a small CNN can help human reviewers rank pulsar
candidates across surveys.

- **Training data:** the public **HTRU1** dataset (Parkes, 1.4 GHz).
- **Test data:** labeled candidates from the Pulsar Search Collaboratory (PSC), from
  **AO327** (Arecibo, 327 MHz) and **GBNCC** (Green Bank, 350 MHz).

Pulsar searches produce far more candidates than anyone can review by hand, and
almost all of them are interference or noise. A model doesn't have to make the final
call. It only has to push promising candidates toward the top of the review queue.

This is decision support for reviewers. It does not reject candidates automatically,
and its scores are **not** calibrated probabilities.

## The approach

- **Inputs.** Each candidate becomes a `2 × 32 × 32` image made from two diagnostic
  plots. The first is frequency vs. phase: HTRU1 channel 1, PSC `FvP`. The second is
  time vs. phase: HTRU1 channel 2, PSC `TvP`. PSC arrays are resized to 32 × 32 and
  scaled per channel to `[0, 1]`.
- **Binary model.** `BaselineCNN` has two convolutional blocks and one logit
  (267,345 parameters). It is trained on HTRU1 as pulsar vs. non-pulsar.
- **Three-class model.** `ThreeClassCNN` (EXP-005 onward) reuses the HTRU1
  convolutional layers. It adds a pulsar / RFI / noise head trained only on PSC
  labels, because HTRU1 has no RFI labels. The HTRU1 model itself stays binary.
- **Baselines.** Every CNN number is reported next to simpler baselines:
  - **Duty cycle:** a one-line statistic measuring how wide the signal is in phase.
  - **Phase-scrambled control:** a CNN trained on images whose phase structure has
    been destroyed.
  - **Metadata models:** models given only period, DM and observation settings.

## What we found

The project went through three stages. Each one changed how the previous results
should be read.

1. **The HTRU1 model doesn't transfer to Arecibo interference** (EXP-001 to EXP-003).
   - It ranks HTRU1's own test batch well, with ROC-AUC 0.975.
   - On 150 AO327 candidates it separates pulsars from noise (0.952) but not from RFI
     (0.617).
   - A duty-cycle statistic computed on the same 32 × 32 inputs reaches 0.936. So the
     information is in the input, and the failure is in model transfer.
2. **Fine-tuning on PSC labels looked like a fix, but the evaluation leaked**
   (EXP-005 to EXP-008).
   - **Arecibo:** fine-tuning on AO327 scored near-perfectly on Arecibo.
   - **Green Bank:** that model failed there.
   - **Joint training:** training on both surveys appeared to fix that failure.
   - **Leak:** these experiments split data by `source_group`, which doesn't capture
     source identity, so test sets shared pulsars with training. Their figures are
     not source-independent performance; see the
     [corrections record](experiments/exp005_exp008_corrections.md).
3. **With source-independent evaluation, there is no evidence that the CNN learned
   pulsar shape** (EXP-009).
   - **Folds:** EXP-009 matched candidates to the ATNF catalogue and built folds that
     never split one pulsar, observation or RFI night across train and test.
   - **GBNCC:** the CNNs beat duty cycle on their own 32-bin input (0.88 vs. 0.79).
     But the phase-scrambled control scored at least as well (0.918) in every
     assignment.
   - **AO327:** every method is near its ceiling.
   - **Cross-survey:** models trained on one survey and tested on the other fell
     below duty cycle.
   - **Metadata:** period and DM alone reach 0.99. The labeled sample is therefore
     heavily confounded, which limits what any image model can show on it.

**Summary:** none of the EXP-009 numbers show the CNN beating simple baselines
because it recognizes pulsar shape. The next step is new data: new sources, new RFI
nights, and noise from more than one night.

## Experiments

Each record is self-contained, with config, environment, hashes, results and
caveats. Read EXP-005 to EXP-008 together with the
[corrections record](experiments/exp005_exp008_corrections.md).

- **[EXP-001](experiments/exp001_htru1_baseline.md)**: HTRU1 baseline.
  - The saved metrics regenerate byte for byte from the saved checkpoint.
  - Test F1 0.716, ROC-AUC 0.975, PR-AUC 0.790.
- **[EXP-002](experiments/exp002_psc_cross_survey_ranking.md)**: first cross-survey
  ranking, over 150 labeled AO327 candidates.
  - Found the pulsar-vs-RFI failure.
  - Its causal interpretation is superseded by EXP-003; its measurements stand.
- **[EXP-003](experiments/exp003_domain_shift_diagnosis.md)**: diagnosis.
  - Ruled out a transposed-axis bug, a phase-bin artifact and normalization mismatch.
  - Duty cycle on the same inputs reaches 0.936.
- **EXP-004 (parked, no record)**: a 1,500-row diversified sample was rejected
  because its classes fell into separate period bands. See the message on commit
  `f376957`.
- **[EXP-005](experiments/exp005_threeclass_finetune.md)**: three-class head
  fine-tuned on AO327 labels.
  - Pulsar-vs-RFI 0.998 on its test split.
  - The backbone was fine-tuned, not frozen.
  - Only 14 of 315 test rows come from sources absent from training.
  - The figure is therefore not independent held-out performance.
- **[EXP-006](experiments/exp006_gbncc_generalization.md)**: the Arecibo-trained
  model on 19 GBNCC RFI candidates (DM < 1).
  - It classed 17 of 19 as pulsars.
  - A duty-cycle threshold set on Arecibo rejected all 19.
  - Caveat: the set is small and mostly 60 Hz mains interference.
- **[EXP-007](experiments/exp007_hybrid_ranker.md)**: combining the CNN with duty
  cycle and DM features in a logistic regression fitted on Arecibo.
  - Every combination containing the CNN still got 2 of 19 GBNCC RFI right.
- **[EXP-008](experiments/exp008_multisurvey.md)**: joint training on AO327 + GBNCC
  (2,746 candidates).
  - Three-class accuracy 0.968 on AO327 and 0.846 on GBNCC.
  - These figures reproduce from the checkpoint, but they are not source-independent.
  - GBNCC was in training, so its figure is in-domain.
- **[EXP-009](experiments/exp009_results.md)**: pre-specified study on
  source-independent folds, defined in the
  [protocol](experiments/exp009_cnn_pilot_protocol.md). It covered 5 folds × 3
  assignments, 72 runs in total.
  - **Image learning: inconclusive, leaning negative.** The phase-scrambled control
    matched or beat every real CNN on GBNCC pulsar vs. RFI.
  - **Cross-survey transfer: inconclusive, leaning negative.** All four estimates
    fell below duty cycle.
  - **Added ranking value: positive only against the weaker baseline.** Adding the
    CNN to duty cycle raised GBNCC from 0.791 to 0.895. It added nothing over duty
    cycle + DM-curve peak (0.915).

## Known limitations

- **The labeled sample is confounded.** In the current sample, period, DM and
  observation settings nearly determine the class:
  - **GBNCC RFI was selected by DM < 1.**
  - **AO327 RFI in the pulsar period band is mostly one ~671 ms mode.**
  - **Metadata models reach 0.99–1.00.**
- **The effective sample is small.**
  - 2,746 candidates come from far fewer distinct pulsars, RFI nights and
    observations.
  - Each survey's noise comes from a single night, so noise results are provisional.
- **There is no fully independent test yet.** EXP-009's three fold assignments
  reuse the same rows, and nothing has been tested on new sources or nights.
- **AO327 is at its ceiling.** Most methods score 0.94–0.98 there, so it can't tell
  methods apart.
- **Labels are crowd-sourced.** PSC grades can mislabel RFI vs. noise. Nine EXP-009
  rows flagged as uncertain still await an expert's review.
- **Masked rows may be a shortcut.** Filling masked rows in test images cut E1's
  GBNCC AUC by 0.159. The models may partly key on masking patterns. This was found
  after the fact and hasn't been tested.
- **Preprocessing loses information.** Squeezing up to 200 phase bins into 32 can
  erase weak pulsars. Duty cycle at native resolution beats duty cycle on the 32-bin
  input on GBNCC (0.858 vs. 0.791).
- **HTRU1 may overlap PSC.** HTRU1 may contain some of the same pulsars, and this
  can't be measured. The randomly initialized arm in EXP-009 performs like the
  HTRU1-initialized one.
- **Scores are not probabilities.** Cross-survey outputs are uncalibrated. Call them
  pulsar-likelihood scores.
- **This is not a validated discovery pipeline.** Nothing here should be used to
  discard a real candidate.

## Repository map

**Data and models**

| Path | What it does |
|---|---|
| `src/datasets/htru1_dataset.py` | Loads HTRU1 batches. Keeps channels 1–2, scales to `[0,1]`, and flips labels so `1.0 = pulsar`. Refuses to mix train and test files. |
| `src/datasets/psc_preprocessing.py` | Turns PSC candidate JSON into the same `(2, 32, 32)` tensor. `arrays_to_tensor()` is the shared resize and normalize step. |
| `src/datasets/psc_dataset.py` | PSC manifest reading and class weights. |
| `src/datasets/psc_folds.py` | EXP-009 source-aware folds, candidate records and write-once output helpers. |
| `src/datasets/psc_controls.py` | EXP-009 phase-scrambled control and test-time interventions. |
| `src/models/baseline_cnn.py` | `BaselineCNN`: two conv blocks → one raw logit. No sigmoid in `forward`. |
| `src/models/three_class_head.py` | `ThreeClassCNN`: the HTRU1 convolutional layers with a pulsar / RFI / noise head. |
| `src/psc_metrics.py`, `src/psc_source_metrics.py` | Duty cycle, per-source ranking metrics and bootstrap intervals. |

**Scripts**

| Path | What it does |
|---|---|
| `scripts/validate_htru1_dataset.py` | Checks the HTRU1 download: counts, shapes, no train/test overlap. |
| `scripts/train_baseline.py` | Trains the HTRU1 binary model. **Overwrites the best checkpoint without asking.** |
| `scripts/evaluate_baseline.py` | Scores the untouched HTRU1 `test_batch`. |
| `scripts/inspect_psc_candidate.py`, `scripts/predict_psc_candidate.py` | Inspect or score a single PSC candidate. |
| `scripts/build_psc_manifest.py` | Builds labeled manifests from the PSC tab exports. |
| `scripts/rank_psc_candidates.py` | Scores a manifest. Caches JSON and rate-limits fetches. |
| `scripts/diagnose_psc_domain_shift.py` | EXP-003 diagnostics. |
| `scripts/finetune_psc_threeclass.py` | EXP-005 and EXP-008 three-class training. |
| `scripts/build_hybrid_ranker.py` | EXP-007 feature combination. |
| `scripts/evaluate_psc_multisurvey.py` | Reproduces EXP-008's per-survey metrics. |
| `scripts/build_source_identity.py`, `scripts/build_source_folds.py` | EXP-009 ATNF cross-match and source-independent folds. Both have a `--check` mode that confirms a byte-for-byte rebuild. |
| `scripts/train_exp009.py`, `scripts/run_exp009_baselines.py`, `scripts/evaluate_exp009.py` | EXP-009 training, baselines and evaluation. They refuse to overwrite existing outputs. |
| `scripts/verify_exp009_protected.py` | Checks that files protected during EXP-009 are unchanged. The README has changed since EXP-009 closed, so the check now reports it as changed. That's expected. |

**Data and outputs**

| Path | Tracked? |
|---|---|
| `data/raw/htru1-batches-py/` | No. Download it separately (see below). |
| `data/psc/source/` | **Yes.** Exports of the PSC grading tabs. |
| `data/psc/psc_eval_*.csv` | **Yes.** Labeled manifests, including the 2,746-row multi-survey manifest. |
| `data/psc/identity/`, `data/psc/folds/` | **Yes.** EXP-009 source identities (ATNF matches and manual decisions) and frozen folds. |
| `data/psc/raw/` | No. A local cache of PSC candidate JSON (2,746 files, ~1 GB). Rebuilding it needs authorized PSC access. |
| `data/raw/atnf/` | No. The ATNF catalogue file used by EXP-009. Its hash is in the EXP-009 protocol. |
| `outputs/models/`, `outputs/metrics/`, `outputs/predictions/`, `outputs/figures/` | **Yes.** Checkpoints, metrics, predictions and figures for EXP-001 to EXP-008. |
| `outputs/exp009/` | **Yes**, except the checkpoints. Run scores, epoch logs, metadata, baseline scores and evaluation reports. The 72 `best.pt` files stay local, and their hashes are in each run's `metadata.json`. |

## Setup and reproducing

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m unittest discover -s tests      # 238 tests, no network needed
```

HTRU1 isn't in the repository. Download it from
[Zenodo record 3205409](https://zenodo.org/records/3205409) and unpack it to
`data/raw/htru1-batches-py/`.

**HTRU1 baseline (EXP-001).** `train_baseline.py` overwrites
`outputs/models/baseline_cnn_best.pt`, so back that file up before retraining.

```bash
python scripts/validate_htru1_dataset.py
python scripts/evaluate_baseline.py
```

**EXP-008 per-survey metrics.** Evaluation only; it trains nothing.

```bash
python scripts/evaluate_psc_multisurvey.py
```

**EXP-009.** These commands need the local PSC cache. The training scripts skip
runs that are already complete.

```bash
python scripts/build_source_identity.py --check
python scripts/build_source_folds.py --check
python scripts/train_exp009.py --plan pilot --skip-completed
python scripts/train_exp009.py --plan extension --skip-completed
python scripts/train_exp009.py --plan e3-completion --skip-completed
python scripts/evaluate_exp009.py --stage full --name <new_directory>
```

## Data sources and citation

- **HTRU1.** See the [project page](https://as595.github.io/HTRU1/) and
  [Zenodo, DOI 10.5281/zenodo.3205409](https://doi.org/10.5281/zenodo.3205409).
  Cite [Morello et al. 2014 (SPINN)](https://arxiv.org/abs/1406.3627) and
  [Keith et al. 2010 (HTRU survey)](https://arxiv.org/abs/1006.5744).
- **ATNF Pulsar Catalogue** (v2.9.0, used for EXP-009 source identities). Cite
  Manchester, R. N., Hobbs, G. B., Teoh, A. & Hobbs, M. 2005, AJ, 129, 1993
  ([ADS](https://ui.adsabs.harvard.edu/abs/2005AJ....129.1993M)), and the
  [catalogue website](https://www.atnf.csiro.au/research/pulsar/psrcat/). The
  catalogue file isn't redistributed here; only the derived matches are.
- **PSC.** See [pulsars.nanograv.org](https://pulsars.nanograv.org/).
  - Candidate access is restricted to authorized participants. Use only candidates
    you may access, and don't bulk-scrape.
  - Raw candidate JSON is not included in this repository.
  - Permission to redistribute the PSC-derived files that are included (candidate
    lists, labels, metadata, model scores and figures) is not yet documented.
