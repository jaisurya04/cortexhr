# Baseline comparison note (Day 3)

**Setup.** 1906 training rows / 477 held-out test rows (stratified 80/20, seed 42); the test set is untouched here. Stratified 5-fold CV (seed 42) on the training rows, **identical folds for every model**. Out-of-fold probabilities come from `cross_val_predict(method="predict_proba")`; preprocessing is refit inside every fold. Threshold metrics use 0.5; undefined precision/F1 (no positive predictions) are reported as 0 (`zero_division=0`) and counted in `folds_with_undefined_precision`.

## Results vs the supplied benchmark (full pipeline: attrition features + attendance/payroll aggregates)
| model | roc_auc | roc_auc_sd | benchmark_roc_auc | delta_roc_auc | pr_auc | precision | recall | f1 | accuracy |
|---|---|---|---|---|---|---|---|---|---|
| Dummy (always stays) | 0.500 | 0.000 | 0.500 | 0.000 | 0.146 | 0.000 | 0.000 | 0.000 | 0.854 |
| Logistic Regression (balanced) | 0.698 | 0.041 | 0.713 | -0.015 | 0.312 | 0.248 | 0.573 | 0.346 | 0.681 |
| Decision Tree (depth 4, balanced) | 0.630 | 0.033 | 0.650 | -0.020 | 0.237 | 0.216 | 0.538 | 0.306 | 0.637 |
| Random Forest (300, balanced) | 0.703 | 0.013 | 0.715 | -0.012 | 0.308 | 0.469 | 0.050 | 0.090 | 0.852 |
| LEAKY Logistic Regression (adds exit information) | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Same models, attrition-table features only (ablation)
| model | roc_auc | roc_auc_sd | benchmark_roc_auc | delta_roc_auc | pr_auc | recall |
|---|---|---|---|---|---|---|
| Dummy (always stays) | 0.500 | 0.000 | 0.500 | 0.000 | 0.146 | 0.000 |
| Logistic Regression (balanced) | 0.705 | 0.027 | 0.713 | -0.008 | 0.308 | 0.595 |
| Decision Tree (depth 4, balanced) | 0.624 | 0.044 | 0.650 | -0.026 | 0.232 | 0.531 |
| Random Forest (300, balanced) | 0.698 | 0.012 | 0.715 | -0.017 | 0.303 | 0.043 |

The 3-month attendance/payroll aggregates exist for only 383 of 2383 rows (the 2,000 historical rows have no monthly history), so they cannot move the overall score much.

## Five-sentence interpretation
1. Logistic regression reaches a cross-validated ROC-AUC of 0.698 (+/-0.041), 0.015 below the benchmark 0.713; a gap this small is believable because my CV uses 80% of the rows and different fold boundaries, and the fold-to-fold spread is itself about 0.04.
2. The always-stay Dummy scores 0.854 accuracy but ROC-AUC 0.500 and recall 0, so accuracy alone would have called it a success while it finds no leavers.
3. The balanced logistic regression catches 57% of leavers at precision 25% (benchmark 63% / 26%), meaning most flagged people are false alarms: that is the price of finding more leavers on a 14.6% base rate.
4. The depth-4 tree (0.630) is weaker than the logistic regression and the forest (0.703) is about equal in ranking but almost never says 'will leave' at a 0.5 threshold (recall 5%); an unlimited tree scores 1.00 AUC on its own training rows but only 0.55 out-of-fold because it memorises noise.
5. No honest model exceeds 0.90 (the highest is 0.703), so the leakage alarm is quiet, while the deliberately leaky demo hits 1.00 and would have been sent back for investigation; SMOTE gave ROC-AUC 0.682 versus 0.698 with class weights, so it brings no ranking gain here.

## Imbalance handling: class weights vs SMOTE
| model | roc_auc_mean | pr_auc_mean | precision_mean | recall_mean | f1_mean |
|---|---|---|---|---|---|
| LR balanced class weights | 0.698 | 0.312 | 0.248 | 0.573 | 0.346 |
| LR no weights | 0.700 | 0.316 | 0.440 | 0.118 | 0.183 |
| LR + SMOTE (train folds only) | 0.682 | 0.301 | 0.238 | 0.538 | 0.329 |

SMOTE is applied only to training folds, never to validation rows. It was implemented in-house (imbalanced-learn was not installable in the build sandbox); because it interpolates between one-hot columns it is closer to plain SMOTE than SMOTE-NC, a known limitation.

## Why a decision tree can look perfect on training data but weaker in cross-validation
An unlimited tree keeps splitting until it has memorised every training row (in-sample AUC 1.00), including noise. On rows it has not seen it scores 0.55. Only the out-of-fold number tells you how it will behave on next year's employees.

## What the balanced logistic regression trades
`class_weight="balanced"` makes a missed leaver cost about 6x a false alarm, so the model flags many more people and finds more leavers (recall 57%) but most flags are wrong (precision 25%). That is acceptable for a ranked list a human reviews, and not acceptable for any automatic action.

## Leakage alarm (ROC-AUC > 0.90 triggers an investigation)
| model | roc_auc_mean | leakage_check |
|---|---|---|
| Dummy (always stays) | 0.500 | ok |
| Logistic Regression (balanced) | 0.698 | ok |
| Decision Tree (depth 4, balanced) | 0.630 | ok |
| Random Forest (300, balanced) | 0.703 | ok |
| LEAKY Logistic Regression (adds exit information) | 1.000 | INVESTIGATE: ROC-AUC > 0.90 (leakage/evaluation bug?) |

The last row is a controlled demonstration (a model that only sees whether the exit fields are filled in). It is **not** part of the real pipeline.

## Run metadata
Seed 42; model configs and package versions in `artifacts/metadata/baseline_run.json`. Python warnings captured during the run: none.
