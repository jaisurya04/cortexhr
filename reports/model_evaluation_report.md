# Model evaluation report (Day 4)

Model: balanced logistic regression on the leakage-safe pipeline. Scores for training rows are **out-of-fold**; the held-out test set (477 rows) was scored once by a model fit on training rows only.

## Metrics in plain words
- **ROC-AUC**: pick one person who left and one who stayed at random; ROC-AUC is the chance the model gives the leaver the higher score (0.5 = coin flip, 1.0 = perfect).
- **PR-AUC (average precision)**: how pure the flagged list stays as you flag more people, which matters when only 14.7% leave. A useless model scores about the base rate (14.7%).
- **Precision**: of the people flagged, how many really left. **Recall**: of the people who left, how many were flagged.
- **Top-10%**: review only the 48 highest-scored employees; precision = share of them who left, recall = share of all leavers they contain. Ties are broken by row position, so the list is deterministic.

## Metrics table
| evaluation | n | positives | base_rate | roc_auc | pr_auc | precision | recall | f1 | accuracy | tn | fp | fn | tp | top_k | top_precision | top_recall | top_lift | brier |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Training out-of-fold (5-fold) | 1906 | 279 | 0.146 | 0.697 | 0.294 | 0.247 | 0.573 | 0.345 | 0.681 | 1138 | 489 | 119 | 160 | 191 | 0.377 | 0.258 | 2.575 | 0.204 |
| Held-out test (used once) | 477 | 70 | 0.147 | 0.682 | 0.321 | 0.259 | 0.629 | 0.367 | 0.681 | 281 | 126 | 26 | 44 | 48 | 0.375 | 0.257 | 2.555 | 0.213 |
| Held-out test @ threshold 0.772 (top-10% cut from training OOF) | 477 | 70 | 0.147 | 0.682 | 0.321 | 0.333 | 0.257 | 0.290 | 0.816 | 371 | 36 | 52 | 18 | 48 | 0.375 | 0.257 | 2.555 | 0.213 |

## Calibration
On the test set the raw scores average 0.40 against an actual leaver rate of 0.147: `class_weight=balanced` inflates probabilities, so a raw '0.40' does **not** mean 4 in 10. After Platt recalibration fitted on training out-of-fold scores only, the average becomes 0.150 and the Brier score moves from 0.213 to 0.117. Chart: `figures/calibration_curve.png`. Calibration is fitted on training data only and never on test labels; because it is a monotone mapping, ranking metrics and risk bands do not change.

## Risk bands (High = top 10%, Medium = next 20%, Low = remaining 70%)
Held-out test:

| band | employees | leavers | share_of_employees_pct | leaver_rate_pct | share_of_all_leavers_pct | lift_vs_base_rate |
|---|---|---|---|---|---|---|
| High | 48 | 18 | 10.063 | 37.500 | 25.714 | 2.555 |
| Medium | 96 | 21 | 20.126 | 21.875 | 30.000 | 1.491 |
| Low | 333 | 31 | 69.811 | 9.309 | 44.286 | 0.634 |

Training out-of-fold:

| band | employees | leavers | share_of_employees_pct | leaver_rate_pct | share_of_all_leavers_pct | lift_vs_base_rate |
|---|---|---|---|---|---|---|
| High | 191 | 72 | 10.021 | 37.696 | 25.806 | 2.575 |
| Medium | 381 | 71 | 19.990 | 18.635 | 25.448 | 1.273 |
| Low | 1334 | 136 | 69.990 | 10.195 | 48.746 | 0.696 |

## Lift
The top 10% of test scores contains **25.7%** of all leavers versus 10% from random picking (lift 2.57x); on training out-of-fold scores the top decile captures 25.8% (lift 2.58x). The test set is small (70 leavers), so treat its numbers as +/- several points. Chart: `figures/gains_lift_chart.png`.

## Honest limits
Only 70 leavers in the test set; ROC-AUC differences below about 0.03 are noise. The model sees data up to the snapshot date; 12% of leavers had already resigned before the snapshot. 84% of rows have no attendance/payroll history. Scores support a human review and must not decide any employment action.
