# Corrections to EXP-005 and EXP-008

Those records are left as written. Read them with these corrections, which come from
the repository audit and the ATNF cross-match (EXP-009).

## Corrected facts

| Record | As written | Corrected | Evidence |
|---|---|---|---|
| EXP-005 scope note, `CLAUDE.md` | Backbone frozen | Not frozen | `frozen_backbone: false`; convolutional weights moved by up to 0.30 |
| EXP-005 split | 929 / 304 / 315 | 907 / 326 / 315 | `psc_threeclass_metrics.json`, and a rebuild |
| EXP-008 | "Arecibo did not degrade — it improved slightly" | No evidence of a difference | Different test sets; 69 vs 70 correct on the 72 rows in both |
| EXP-008 | "Beats the baseline on a telescope other than the one it was tuned for" | GBNCC was in training | Manifest and split |
| EXP-008 | GBNCC noise spans 2 nights | 1 night | Manifest MJD is rounded; JSON MJD gives one UTC day |
| `CLAUDE.md` | Accuracy 0.9684 set against duty-cycle 0.9726 | Accuracy vs ROC-AUC, not comparable | Compare 0.9961 with 0.9726 instead |

## Independence

`source_group` does not capture source identity. Checked against verified identities:

| | EXP-005 | EXP-008 AO327 | EXP-008 GBNCC |
|---|---|---|---|
| Test pulsar rows | 110 | 92 | 48 |
| From sources also in train/val | 103 | 83 | 41 |
| Strictly independent test rows, total | 14 of 315 | 34 of 543, both surveys | |

Do not quote 0.9587, 0.9684, or 0.8458 as independent held-out performance. The metrics
do reproduce from the checkpoints; the problem is what the test sets measure.

## Still usable

The cross-telescope numbers from the Arecibo-only model (0.2905 accuracy on GBNCC, RFI
recall 0.1357, noise recall 0.0) remain valid in-sample descriptions: that model never
trained on GBNCC, apart from 2 shared pulsars. They are selected data, and EXP-006's 19
rows cover one RFI type.
