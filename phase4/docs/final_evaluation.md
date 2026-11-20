# Day 20 - Final evaluation (re-run 2026-11-20)

Phase 4 evaluations (forecast backtests, monitoring, tests) were re-run for this release. Phase 1-3 model evaluations are **read from their saved evaluation files**, not re-trained here (the restricted validation vault and the pinned scikit-learn 1.8.0 are needed to re-train them).

## Monitoring status now: ALERT (coverage partial)

* Drift: 8 features checked, max PSI 3.835 (months_since_last_raise); 3 above the WATCH line: months_since_last_raise 3.835, annual_fixed_pay 0.159, comp_ratio 0.104
* Note: months_since_last_raise has collapsed rather than drifted: 77% of the current population share the value 5 (reference mean 11.0, current mean 5.0). Check whether a company-wide event (for example a raise cycle) or a pipeline change explains it before trusting scores that use it.
* Twin resumes: 75 comparisons, max gap 0.000000 (tolerance 0.001); status OK.
* Pay gap by career_gap: model gap 0.0085 vs history 0.0368; status OK.
* Pay gap by gender: restricted CX-15 file not found (set CORTEXHR_VAULT_DIR); gender pay gap NOT checked

## Attrition v2 (cross-validation)

| model | roc_auc_mean | pr_auc_mean | brier_mean | roc_auc_sd |
|---|---|---|---|---|
| HistGradientBoosting default | 0.68 | 0.294 | 0.179 | 0.017 |
| HistGradientBoosting tuned | 0.695 | 0.31 | 0.198 | 0.021 |
| Logistic Regression (balanced) [baseline] | 0.709 | 0.325 | 0.2 | 0.021 |
| Random Forest (300, balanced) [baseline] | 0.7 | 0.316 | 0.123 | 0.025 |
| attrition-v2 candidate: LR + tuned HGB blend | 0.716 | 0.325 | 0.189 | 0.017 |
| LightGBM trial 1 | 0.692 | 0.319 | 0.196 | 0.025 |
| LightGBM trial 2 | 0.691 | 0.304 | 0.186 | 0.016 |
| LightGBM trial 3 | 0.689 | 0.304 | 0.188 | 0.023 |

**Gain over the simple baseline is NOT proven; the baseline is not beaten (Phase 3 Task 13).**

## Resume ranker

| engine | precision@5 | recall@5 | hit_rate@5 | ndcg@5 | mrr | average_precision | r_precision |
|---|---|---|---|---|---|---|---|
| final engine (v2, masked) | 0.94 | 0.549 | 1.0 | 0.923 | 1.0 | 0.96 | 0.882 |
| random ordering | 0.166 | 0.089 | 0.592 | 0.266 | 0.355 | 0.223 | 0.168 |

## Pay-band model

| model | MAE_rupees | RMSE_rupees | R2 | median_abs_pct_error | MAE_log |
|---|---|---|---|---|---|
| baseline: average offered salary by role | 115094.237 | 167820.719 | 0.933 | 12.769 | 0.14 |
| baseline 2: role average adjusted for market index | 55656.281 | 88203.649 | 0.982 | 5.521 | 0.066 |
| Ridge regression (job-related features) | 53924.807 | 84330.025 | 0.983 | 5.397 | 0.065 |
| Gradient boosting (job-related features, tuned) | 57504.058 | 93722.1 | 0.979 | 5.731 | 0.068 |
| Ridge regression + career_gap_months | 53160.626 | 83054.131 | 0.984 | 5.451 | 0.064 |
| Gradient boosting + career_gap_months | 56955.579 | 92902.114 | 0.98 | 5.657 | 0.067 |

## Headcount forecast (company)

| metric | model | scope | n | mae | mape_pct | coverage_80 |
|---|---|---|---|---|---|---|
| headcount_end | baseline | holdout_last_6_months | 6 | 4.333 | 1.008 | 1.0 |
| headcount_end | baseline | rolling_origin | 162 | 9.809 | 2.556 | 0.679 |
| headcount_end | improved | holdout_last_6_months | 6 | 3.958 | 0.936 | 0.833 |
| headcount_end | improved | rolling_origin | 162 | 7.153 | 1.859 | 0.753 |
| exits | baseline | holdout_last_6_months | 6 | 1.833 | 55.093 | 1.0 |
| exits | baseline | rolling_origin | 162 | 1.965 | 47.327 | 0.852 |
| exits | improved | holdout_last_6_months | 6 | 1.314 | 30.412 | 0.833 |
| exits | improved | rolling_origin | 162 | 2.041 | 43.673 | 0.63 |
| exits | improved | rolling_origin_raw_same_rows | 114 | nan | nan | 0.658 |
| exits | improved | rolling_origin_calibrated_walk_forward | 114 | nan | nan | 0.895 |
| headcount_end | improved | rolling_origin_raw_same_rows | 150 | nan | nan | 0.747 |
| headcount_end | improved | rolling_origin_calibrated_walk_forward | 150 | nan | nan | 0.813 |
