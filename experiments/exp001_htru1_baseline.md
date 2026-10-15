# EXP-001 — HTRU1 binary baseline (two-channel)

Reference record for the HTRU1-trained `BaselineCNN` checkpoint that all
subsequent cross-survey PSC scores are attributed to. Every PSC ranking result
should cite this experiment id until a new checkpoint replaces it.

## Identity

| Field | Value |
|---|---|
| `experiment_id` | `EXP-001` |
| `date_trained` | 2026-08-04 (checkpoint mtime 17:29:58 local) |
| `date_evaluated` | 2026-08-04 17:33:28 local |
| `date_recorded` | 2026-08-05 |
| `git_commit` (at recording) | `f0a872d8082bf19f1f0d740174df0fd22e4e4445` |
| `code_commit` (last change to model/train/eval code) | `32a24e1` (train + eval), `cb755a3` (model) |
| `tracked_tree_clean` | yes |

The training run predates the recording commit. No code touching
`src/models/baseline_cnn.py`, `scripts/train_baseline.py`, or
`scripts/evaluate_baseline.py` has landed between the run and this record —
commits `5344b85` and `f0a872d` touched only PSC files and `.gitignore` — so
`f0a872d` reproduces this result exactly (verified below).

## Environment

| Field | Value |
|---|---|
| `python_version` | 3.13.14 |
| `torch_version` | 2.13.0 |
| `numpy_version` | 2.5.1 |
| `sklearn_version` | 1.9.0 |
| `platform` | macOS-26.5.2-arm64-arm-64bit-Mach-O |
| `device_eval` | `mps` (confirmed) |
| `device_train` | not logged; `select_device()` would have chosen `mps` on this host |

## Data

| Field | Value |
|---|---|
| `train_batches` | `data_batch_1` … `data_batch_5` (50,000 candidates) |
| `test_batch` | `test_batch` (10,000 candidates, 199 pulsars) |
| `channels` | HTRU1 channels 1 and 2 → Frequency–Phase, Time–Phase |
| `input_shape` | `(2, 32, 32)`, float32, values scaled by `/255.0` |
| `label_convention` | `1.0 = pulsar`, `0.0 = non-pulsar` (flipped from HTRU1 source `0 = pulsar`) |
| `validation_fraction` | 0.2 → 40,000 train / 10,000 val |
| `test_batch_used_for_tuning` | no — `HTRU1Dataset` rejects mixed train/test paths, and `train_baseline.py` never opens `test_batch` |

## Configuration

| Field | Value | Provenance |
|---|---|---|
| `random_seed` | 42 | recorded in checkpoint |
| `pos_weight` | 48.8132 | recorded in checkpoint |
| `batch_size` | 64 | script default — **not independently verified** |
| `learning_rate` | 1e-3 | script default — **not independently verified** |
| `optimizer` | Adam | script default |
| `loss` | `BCEWithLogitsLoss(pos_weight=48.8132)` | script default |
| `epochs_requested` | 5 (default) — **unverified**, see caveat | — |
| `best_epoch` | 1 | recorded in checkpoint |
| `validation_f1` | 0.7285714285714285 | recorded in checkpoint |
| `model_params` | 267,345 trainable | recomputed |

The stored `pos_weight` of 48.8132 implies 803 pulsars in a 40,000-sample
training split (40000 / 49.8132 = 803.0), which confirms the run used the **full
training pool** — not a `--max-samples` smoke test.

## HTRU1 test metrics (threshold 0.5, n = 10,000)

| Metric | Value |
|---|---|
| `test_loss` | 0.542780780725798 |
| `precision` | 0.6771300448430493 |
| `recall` | 0.7587939698492462 |
| `f1` | 0.7156398104265403 |
| `accuracy` | 0.988 |
| `roc_auc` | 0.9748779608685196 |
| `pr_auc` | 0.7903631989299045 |

Confusion matrix: TP 151, FP 72, TN 9729, FN 48.

Ranking (the metric family that matters for the cross-survey goal):

| Metric | Value |
|---|---|
| `pulsars_in_top_10` | 10 / 10 |
| `pulsars_in_top_50` | 50 / 50 |
| `pulsars_in_top_100` | 92 / 100 |
| `pulsars_in_top_200` | 146 / 199 |
| `recall_at_top_1pct` (k=100) | 0.4623115577889447 |
| `recall_at_top_5pct` (k=500) | 0.8894472361809045 |
| `recall_at_top_10pct` (k=1000) | 0.9396984924623115 |

Accuracy is reported only for completeness — at 199/10,000 positives it is not
informative and must not be quoted as the headline result.

## Artifacts and hashes

| Path | SHA-256 | Tracked? |
|---|---|---|
| `outputs/models/baseline_cnn_best.pt` | `7f852a3e7fc46ade884c32a4a5599cc4c2734d0fd262a2c17620a4f6e4695d5c` | yes (since 2026-10-14) |
| `outputs/models/exp001_baseline_cnn_best.pt` | same as above — preservation copy | yes (since 2026-10-14) |
| `outputs/metrics/htru1_test_metrics.json` | `9830360575d0cab8fb73cd922d0452dbf46f37701c8f378df0850cf0f9c1d4b2` | yes (since 2026-10-14) |
| `outputs/predictions/htru1_test_predictions.csv` | `67e207c19e691c4ecef2d17890a868fd77ea95d4efe78fae1dabcd2033b78963` | yes (since 2026-10-14) |

Checkpoint state dict keys: `features.0.{weight,bias}`, `features.3.{weight,bias}`,
`classifier.1.{weight,bias}`, `classifier.4.{weight,bias}`.
Checkpoint metadata keys: `epoch`, `val_f1`, `pos_weight`, `random_seed`.

These artifacts were gitignored when this record was written, so the metrics
above are reproduced in full in this file. Since 2026-10-14 the artifacts are
tracked as well, unchanged; the hashes above still match them.

## Reproducibility check (2026-08-05)

Re-ran evaluation against the existing checkpoint into scratch paths, leaving the
saved artifacts untouched:

```bash
python scripts/evaluate_baseline.py \
  --predictions-csv <scratch>/repro_predictions.csv \
  --metrics-json    <scratch>/repro_metrics.json
```

Result:

- metrics JSON — **identical on all 28 keys**
- predictions CSV — **byte-for-byte identical**, all 10,000 rows

Evaluation is deterministic on this host at commit `f0a872d`, and the saved
metrics genuinely belong to the saved checkpoint.

## Open caveats

1. **Epoch count is unverified.** `best_epoch = 1`. No training log was kept, so
   nothing on disk distinguishes "5 epochs ran and epoch 1 won" from "only 1
   epoch ran." Settling this requires a retrain, which would need the checkpoint
   backed up first (see below). Until then, do not claim a 5-epoch run.
2. **`batch_size` and `learning_rate` are assumed defaults.** They are not stored
   in the checkpoint and cannot be recovered from it.
3. **`train_baseline.py` overwrites `outputs/models/baseline_cnn_best.pt`
   silently** on any run that improves validation F1. The preservation copy at
   `outputs/models/exp001_baseline_cnn_best.pt` exists to survive that. Verify
   its hash before treating it as EXP-001.
4. **Val/test split integrity.** The 80/20 split is index-based over the
   concatenated 50k training pool with seed 42; `test_batch` is never loaded
   during training. Filename non-overlap between train and test is asserted by
   `scripts/validate_htru1_dataset.py`, but that does not rule out physically
   related candidates, harmonics, or duplicates across the two sets.
5. **These are in-domain HTRU1 numbers only.** They say nothing about PSC AO327
   performance, and the sigmoid output is not a calibrated probability on either
   survey.

## Downstream note

The PSC RFI false positive (plot `11732926`) was scored against this checkpoint.
That result is not yet preserved anywhere in the repository; capturing it belongs
to the PSC evaluation manifest work, not to this record.
