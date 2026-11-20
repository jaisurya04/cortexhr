# Model Card: CortexHR headcount and exits forecast   (CX-20)

**Purpose** projects company and department headcount and company exits for the next six months with an 80% range.  ·  **Not for** individual decisions, budgets set without review, or forecasts beyond 12 months.

**Data** `monthly_headcount_history.csv` (81 months, 2026-09 last) and `workforce_events.csv` (9 events, none after 2026-04). Department D00 (1 person) is not forecast.

**Method** Baselines: *seasonal naive + yearly drift* for headcount and *same-month average of 3 years* for exits. Improved model: events + yearly seasonality regression with AR(1) errors (statsmodels SARIMAX) on the monthly net change; interval from the model's own error recursion. The improved model replaces the baseline only if its rolling-origin MAE is at least 1% lower.

**Metrics**

**Improved-model status in the saved run: run.**

| metric | model | method | scope | n | mae | mape_pct | bias | coverage_80 |
|---|---|---|---|---|---|---|---|---|
| headcount_end | baseline | seasonal naive + yearly drift | holdout_last_6_months | 6 | 4.333 | 1.008 | 2.0 | 1.0 |
| headcount_end | baseline | seasonal naive + yearly drift | rolling_origin | 162 | 9.809 | 2.556 | -5.364 | 0.679 |
| headcount_end | improved | events + seasonality regression with AR(1) errors (statsmodels SARIMAX) | holdout_last_6_months | 6 | 3.958 | 0.936 | -0.928 | 0.833 |
| headcount_end | improved | events + seasonality regression with AR(1) errors (statsmodels SARIMAX) | rolling_origin | 162 | 7.153 | 1.859 | -6.082 | 0.753 |
| exits | baseline | same-month average of last 3 years | holdout_last_6_months | 6 | 1.833 | 55.093 | -0.5 | 1.0 |
| exits | baseline | same-month average of last 3 years | rolling_origin | 162 | 1.965 | 47.327 | 1.047 | 0.852 |
| exits | improved | events + seasonality regression with AR(1) errors (statsmodels SARIMAX) | holdout_last_6_months | 6 | 1.314 | 30.412 | 0.411 | 0.833 |
| exits | improved | events + seasonality regression with AR(1) errors (statsmodels SARIMAX) | rolling_origin | 162 | 2.041 | 43.673 | 1.386 | 0.63 |
| exits | improved | nan | rolling_origin_raw_same_rows | 114 | nan | nan | nan | 0.658 |
| exits | improved | nan | rolling_origin_calibrated_walk_forward | 114 | nan | nan | nan | 0.895 |
| headcount_end | improved | nan | rolling_origin_raw_same_rows | 150 | nan | nan | nan | 0.747 |
| headcount_end | improved | nan | rolling_origin_calibrated_walk_forward | 150 | nan | nan | nan | 0.813 |

holdout = last 6 known months hidden and forecast; rolling_origin = the same test from many past starting points. Coverage should be near 0.80.

Baseline interval calibration check (`interval_calibration_study.csv`, mean coverage across series):

| metric | interval_method | coverage_80 |
|---|---|---|
| exits | signed quantiles | 0.759 |
| exits | symmetric absolute error | 0.852 |
| headcount_end | signed quantiles | 0.671 |
| headcount_end | symmetric absolute error | 0.763 |


**Explanations** Past events enter the improved model as 0/1 regressors (hiring freeze, hiring boost, policy change). Plain-language explanation in `docs/forecasting_report.md`.

**Fairness** Not applicable: the forecast uses counts, not people's attributes.

**Limits & risks** Only one hiring freeze (2020) is available to learn its effect; intervals ignore parameter uncertainty; the baseline 80% interval covered clearly less than 80% in rolling tests, so treat ranges as optimistic.

**Version & owner** `cortexhr-forecast-1.0.0` · card written 2026-11-20 · Jaisurya Marisamy (GM00134), AI/ML Engineer, Growmar

_Every output of this model is AI-generated and needs human review. Numbers in this card are copied from the evaluation files named in the text by `scripts/build_docs.py`._
