# EXP-009 — CNN pilot and three-assignment results

Results of the pre-specified study in
[`exp009_cnn_pilot_protocol.md`](exp009_cnn_pilot_protocol.md). The question is whether a
CNN trained on PSC AO327 + GBNCC images learns pulsar morphology that a duty-cycle
statistic and a phase-scrambled control do not, under source-independent splits.

| Field | Value |
|---|---|
| `experiment_id` | `EXP-009` |
| `date` | 2026-11-03 |
| `branch` | `exp009-cnn-pilot` |
| `code` | `f0618d7` (62 runs), `f3ab381` (10 E3 completion runs and all evaluation); training code is identical in both |
| `manifest` | `data/psc/psc_eval_multisurvey.csv`, 2,746 rows, sha256 `273379bd…` |
| `identity` | `data/psc/identity/source_identity_v1.csv`, sha256 `e225cc64…` |
| `folds` | `data/psc/folds/folds_v1.json`, sha256 `ef4681d2…`, assignment seeds 5 / 44 / 52 |
| `backbone` | EXP-001 HTRU1 checkpoint, sha256 `7f852a3e…` (unmodified) |
| `recipe` | `ThreeClassCNN`, Adam lr 1e-3, batch 32, 40 epochs, class-weighted CE, best epoch by val accuracy, CPU, 4 threads |
| `model seeds` | `42 + 100*assignment + fold`; cross-survey `+10` (AO327→GBNCC) / `+11` (GBNCC→AO327) |
| `runs` | 72: E1, E2, E3, C1 × 5 folds × 3 assignments, plus L1, L2 × 2 directions × 3 assignments |
| `training time` | 2,411 s CPU in total |
| `run outputs` | `outputs/exp009/runs/<run_id>/` (untracked) |
| `baselines` | `outputs/exp009/baselines/scores.csv`, sha256 `34f7ba29…` |
| `pilot report` | `outputs/exp009/evaluation/pilot/report.json`, sha256 `01989bb8…` (original scoring) |
| `pilot report, rescored` | `outputs/exp009/evaluation/pilot_logodds/report.json`, sha256 `86981ec7…` |
| `full report` | `outputs/exp009/evaluation/full/report.json`, sha256 `1ac917cf…` |
| `intervals` | Paired bootstrap over sources, RFI nights and noise observations, 2,000 draws, plus between-assignment variance; 95% for the pilot, 98.3% (Bonferroni over 3 claims) for the final stage |

Protected files (`experiments/exp009_protected_hashes.sha256`, 100 entries) were
verified unchanged after every stage.

## Headline

**No claim succeeded cleanly.** Claims A and B are inconclusive, and each leans
negative. Claim C succeeds against duty on the CNN input on GBNCC, but not against the
stronger duty + DM-curve baseline.

On GBNCC pulsar vs RFI, the row-shuffled control C1 scored higher than every real
CNN arm in every assignment. The real arms do use phase structure (scrambling it at
test time costs them AUC), but a model that never sees that structure does at least
as well. EXP-009 therefore gives no evidence that the CNN's advantage over duty cycle
comes from pulsar morphology.

Per the protocol, the study stops here on this data. The next step is new sources
and nights.

## Pilot (assignment 0)

**Provisional negative** for both E1 and E2, so the fixed extension ran and
preprocessing arms stayed on hold.

| GBNCC pulsar vs RFI, typical AUC, 95% | Original scoring | Log-odds scoring |
|---|---|---|
| E1 − C1 | −0.054 [−0.110, −0.011] | −0.054 [−0.110, −0.011] |
| E2 − C1 | −0.040 [−0.077, −0.010] | −0.040 [−0.078, −0.011] |

## Final stage (three assignments)

### Per-model results, pulsar vs RFI

Typical-detection AUC with 98.3% intervals; FPR at 90% of sources found.

| Model | AO327 AUC | AO327 FPR | GBNCC AUC | GBNCC FPR |
|---|---|---|---|---|
| E1 | 0.979 [0.930, 1.000] | 0.015 | 0.881 [0.802, 0.947] | 0.260 |
| E2 | 0.981 [0.938, 1.004] | 0.027 | 0.884 [0.812, 0.943] | 0.226 |
| E3 | 0.944 [0.873, 0.987] | 0.182 | 0.820 [0.733, 0.895] | 0.286 |
| C1 | 0.978 [0.938, 0.998] | 0.028 | 0.918 [0.853, 0.966] | 0.159 |
| Duty, CNN input | 0.969 [0.913, 0.999] | 0.001 | 0.791 [0.703, 0.869] | 0.464 |
| Duty, native | 0.967 [0.904, 1.000] | 0.000 | 0.858 [0.777, 0.922] | 0.326 |
| Duty + DMc forest | 0.973 [0.923, 1.001] | 0.000 | 0.918 [0.857, 0.963] | 0.244 |
| HTRU1 model | 0.540 [0.376, 0.718] | 0.965 | 0.375 [0.264, 0.483] | 0.950 |
| Period + DM forest | 0.991 [0.961, 1.004] | 0.005 | 0.994 [0.969, 1.005] | 0.000 |
| All metadata forest | 0.990 [0.944, 1.005] | 0.000 | 0.989 [0.964, 1.000] | 0.000 |

Populations: AO327 has 42 sources against 86 RFI units; GBNCC has 62 sources against
69 RFI units. The frozen bars reproduce exactly (duty 0.969 / 0.967 / HTRU1 0.540 on
AO327; 0.791 / 0.858 / 0.375 on GBNCC).

Metadata forests reach 0.99–1.00. That measures confounding in this sample. It is not
a bar the CNN has to clear to be useful.

GBNCC typical AUC by assignment:

| | a0 | a1 | a2 |
|---|---|---|---|
| E1 | 0.869 | 0.885 | 0.887 |
| E2 | 0.883 | 0.883 | 0.884 |
| E3 | 0.817 | 0.817 | 0.826 |
| C1 | 0.923 | 0.923 | 0.907 |

### Claim A — image learning: inconclusive, leaning negative

| GBNCC pulsar vs RFI | E1 | E2 | E3 |
|---|---|---|---|
| Minus C1 | −0.037 [−0.113, +0.023] | −0.034 [−0.088, +0.008] | −0.098 [−0.181, −0.033] |
| Minus duty (CNN input) | +0.089 [−0.019, +0.193] | +0.092 [−0.006, +0.191] | +0.029 [−0.078, +0.138] |
| Minus C1, fundamental detections only | −0.018 [−0.092, +0.048] | −0.019 [−0.084, +0.032] | −0.099 [−0.187, −0.036] |
| Minus C1, uncertain rows excluded | −0.029 [−0.095, +0.029] | −0.026 [−0.074, +0.014] | −0.110 [−0.194, −0.042] |
| AO327: minus duty (CNN input) | +0.010 [−0.044, +0.070] | +0.013 [−0.035, +0.078] | −0.025 [−0.086, +0.034] |
| GBNCC AUC drop when test rows are shuffled | 0.058 | 0.037 | −0.013 |

- Success needs some arm to beat both C1 and duty on GBNCC. None does.
- Genuine failure needs every arm's upper bound against C1 below +0.02. E1's is
  +0.023, so the verdict is inconclusive by 0.003.
- Including E3 does not change the verdict; it sits clearly below C1. The
  pre-specified E1/E2 verdict and the E3-inclusive one are the same.
- Component-block resampling agrees (E1 − C1: −0.037 [−0.117, +0.026]).

### Claim B — cross-survey transfer: inconclusive, leaning negative

Models trained on one survey and tested on the other; the 2 shared sources removed.

| Target | L1 | L2 | Duty, CNN input | L1 − duty | L2 − duty |
|---|---|---|---|---|---|
| AO327 (40 sources) | 0.855 | 0.902 | 0.967 | −0.112 [−0.219, −0.015] | −0.066 [−0.244, +0.080] |
| GBNCC (60 sources) | 0.741 | 0.642 | 0.787 | −0.046 [−0.181, +0.087] | −0.144 [−0.283, −0.012] |

All four point estimates are below duty, and two are clearly below it. Failure needs
every arm clearly below duty in both directions, which two intervals prevent.
GBNCC-trained models label 93–99% of AO327 rows as RFI in assignment 2, the same
collapse seen in the pilot.

### Claim C — added ranking value: success against the weaker baseline only

Logistic stacking fitted on each run's validation rows, scored on its test rows.

| Combination | Population | Baseline → combined | Difference |
|---|---|---|---|
| E1 + duty (CNN input) | GBNCC RFI | 0.791 → 0.895 | +0.104 [+0.005, +0.208] **success** |
| E2 + duty (CNN input) | GBNCC RFI | 0.791 → 0.895 | +0.104 [+0.011, +0.199] **success** |
| E1 + duty + DMc | GBNCC RFI | 0.915 → 0.927 | +0.012 [−0.057, +0.082] inconclusive |
| E2 + duty + DMc | GBNCC RFI | 0.915 → 0.927 | +0.012 [−0.052, +0.081] inconclusive |
| E1 / E2 + either | AO327 RFI | 0.966–0.969 → 0.981–0.986 | +0.013 to +0.020, all spanning 0, inconclusive |
| E3 + either | Both | — | All inconclusive |

The CNN adds information that duty on its own 32-bin input lacks. Duty plus the
position of the DM-curve peak captures about the same information.

### Claim D: not assessable with current data

## Additional observations (exploratory)

These were not pre-specified as decision criteria and support no claim.

- **Noise is provisional.** On GBNCC pulsar vs noise, E1 beats C1 (+0.049
  [+0.006, +0.100]), the reverse of RFI. Each survey's noise comes from one night.
- **Interventions.** Average GBNCC pulsar-vs-RFI AUC drop when trained models score
  altered test images:

  | Intervention | E1 | E2 | E3 |
  |---|---|---|---|
  | Remove FvP | 0.250 | 0.177 | 0.146 |
  | Remove TvP | 0.248 | 0.183 | 0.326 |
  | Shuffle whole array | 0.177 | 0.162 | 0.088 |
  | Fill blank (masked) rows | 0.159 | 0.129 | 0.150 |
  | Shuffle within rows | 0.058 | 0.037 | −0.013 |
  | Roll phase | −0.007 | −0.005 | −0.028 |
  | Rebin phase | −0.005 | 0.010 | −0.027 |

  Phase rolling and rebinning change little, as expected. The large effect of
  filling masked rows suggests the models partly key on masking patterns. That would
  be an acquisition-side shortcut, not morphology, and is worth testing directly.
- **One-detection sensitivity.** For every CNN arm, the random-row and strongest-row
  AUCs are within 0.016 of the typical-detection AUC. On GBNCC the arms keep the same
  order under every metric (C1 above E1 and E2, above E3), so the Claim A picture
  does not rest on repeat detections.

## Amendments made during the study

Both are recorded in the protocol's Amendments section, dated 2026-11-03, before any
assignment 1–2 test metric was computed.

1. **Score precision.** CNN rows are ranked by full-precision pulsar log-odds from the
   stored logits, not by `p_pulsar` rounded to 8 decimals. This moved pilot AUCs by
   at most 0.008 and left the pilot decision unchanged. Claim C does not depend on
   the accompanying removal of probability clipping (combined AUC differs by
   ±0.0001).
2. **E3 completion.** The 10 E3 runs for assignments 1–2 were added after the pilot.
   As stated above, they change no verdict.

## Limitations

- All three assignments reuse the same 2,746 rows, so the assignments are not
  independent replications.
- Label decisions rest on non-expert plot inspection, and 9 uncertain rows await an
  experienced reviewer. Excluding them changes no conclusion.
- GBNCC RFI is selected by DM < 1, and AO327 RFI in the pulsar period band is mostly
  one ~671 ms mode. Both make metadata nearly sufficient in this sample.
- AO327 is at its ceiling, so it cannot discriminate between models.
- HTRU1 may contain some of the same pulsars. E2 (random initialization) performs
  like E1, so this has no visible effect here.
- PSC authorization for the cached data is not documented in the repository.

## Reproduce

```bash
python scripts/verify_exp009_protected.py
python scripts/train_exp009.py --plan pilot --skip-completed
python scripts/train_exp009.py --plan extension --skip-completed
python scripts/train_exp009.py --plan e3-completion --skip-completed
python scripts/run_exp009_baselines.py
python scripts/evaluate_exp009.py --stage full --name <new_directory>
```

The evaluation refuses to overwrite an existing report directory.
