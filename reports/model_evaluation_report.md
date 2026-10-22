# Model evaluation report (Day 4)

Model: balanced logistic regression on the leakage-safe pipeline. Scores for training rows are **out-of-fold**; the held-out test set (477 rows) was scored once by a model fit on training rows only.

## Metrics in plain words
- **ROC-AUC**: pick one person who left and one who stayed at random; ROC-AUC is the chance the model gives the leaver the higher score (0.5 = coin flip, 1.0 = perfect).
- **PR-AUC (average precision)**: how pure the flagged list stays as you flag more people, which matters when only 14.7% leave. A useless model scores about the base rate (14.7%).
- **Precision**: of the people flagged, how many really left. **Recall**: of the people who left, how many were flagged.
- **Top-10%** (official `binary_report`): review only the 47 highest-scored employees (int(n x 0.10)); precision = share of them who left, recall = share of all leavers they contain.

## Metrics table
| evaluation | n | positives | base_rate | roc_auc | pr_auc | precision | recall | f1 | accuracy | tn | fp | fn | tp | top_k | top_precision | top_recall | top_lift | brier |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Training out-of-fold (5-fold) | 1906 | 279 | 0.146 | 0.697 | 0.294 | 0.247 | 0.573 | 0.345 | 0.681 | 1138 | 489 | 119 | 160 | 191 | 0.377 | 0.258 | 2.575 | 0.204 |
| Held-out test (used once) | 477 | 70 | 0.147 | 0.682 | 0.321 | 0.259 | 0.629 | 0.367 | 0.681 | 281 | 126 | 26 | 44 | 48 | 0.375 | 0.257 | 2.555 | 0.213 |
| Held-out test @ threshold 0.772 (top-10% cut from training OOF) | 477 | 70 | 0.147 | 0.682 | 0.321 | 0.333 | 0.257 | 0.290 | 0.816 | 371 | 36 | 52 | 18 | 48 | 0.375 | 0.257 | 2.555 | 0.213 |

## Official helper (`05_Starter/cortexhr_helpers.py`) and parity with my extended report
| evaluation | roc_auc | pr_auc | precision_at_top | recall_at_top | f1_at_top | base_rate |
|---|---|---|---|---|---|---|
| training out-of-fold | 0.697 | 0.294 | 0.379 | 0.258 | 0.307 | 0.146 |
| held-out test | 0.682 | 0.321 | 0.383 | 0.257 | 0.308 | 0.147 |

Parity (held-out test): ROC-AUC, PR-AUC and base rate are identical; the top group differs by one row because my extended report uses ceil(n x 10%) with a stable tie-break and the official helper uses int(n x 10%).

| metric | official_helper | my_extended_report | difference |
|---|---|---|---|
| roc_auc | 0.6822 | 0.6822 | 0.0000 |
| pr_auc | 0.3209 | 0.3209 | 0.0000 |
| base_rate | 0.1468 | 0.1468 | 0.0000 |
| top-10% group size | 47.0000 | 48.0000 | 1.0000 |
| precision in top group | 0.3830 | 0.3750 | -0.0080 |
| recall in top group | 0.2571 | 0.2571 | 0.0000 |

## Calibration
On the test set the raw scores average 0.40 against an actual leaver rate of 0.147: `class_weight=balanced` inflates probabilities, so a raw '0.40' does **not** mean 4 in 10. After Platt recalibration fitted on training out-of-fold scores only, the average becomes 0.150 and the Brier score moves from 0.213 to 0.117. Chart: `figures/calibration_curve.png`. Calibration is fitted on training data only and never on test labels; because it is a monotone mapping, ranking metrics and risk bands do not change.

## Risk bands: official company rule (`risk_bands`: High = score at or above the 90th percentile, Medium = at or above the 70th, Low = the rest)
Held-out test:

| band | employees | leavers | share_of_employees_pct | leaver_rate_pct | share_of_all_leavers_pct | lift_vs_base_rate |
|---|---|---|---|---|---|---|
| High | 48 | 18 | 10.063 | 37.500 | 25.714 | 2.555 |
| Medium | 95 | 20 | 19.916 | 21.053 | 28.571 | 1.435 |
| Low | 334 | 32 | 70.021 | 9.581 | 45.714 | 0.653 |

Training out-of-fold:

| band | employees | leavers | share_of_employees_pct | leaver_rate_pct | share_of_all_leavers_pct | lift_vs_base_rate |
|---|---|---|---|---|---|---|
| High | 191 | 72 | 10.021 | 37.696 | 25.806 | 2.575 |
| Medium | 381 | 71 | 19.990 | 18.635 | 25.448 | 1.273 |
| Low | 1334 | 136 | 69.990 | 10.195 | 48.746 | 0.696 |

## Lift
The official top-10% group of test scores holds **25.7%** of all leavers versus 10% from random picking (precision 38.3% vs base rate 14.7%, lift 2.61x); on training out-of-fold scores it holds 25.8% (lift 2.59x). The test set is small (70 leavers), so treat its numbers as +/- several points. Chart: `figures/gains_lift_chart.png`.

## Honest limits
Only 70 leavers in the test set; ROC-AUC differences below about 0.03 are noise. The model sees data up to the snapshot date; 12% of leavers had already resigned before the snapshot. 84% of rows have no attendance/payroll history. Scores support a human review and must not decide any employment action.
