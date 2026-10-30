# CortexHR Phase 1 - Data Audit Report (Day 1)

Written **before** any model was trained and **before** the validation vault was opened (see `vault_comparison_cx01.md` for the later comparison).

## 1. What the data is
- Table `attrition_dataset` in `03_Data/CortexHR_Dataset.xlsx`: **2,389 rows x 42 columns** (matches the 2,389 rows stated in the brief).
- **One row = one employee at one observation date** (a snapshot of their features). The target **`left_within_12m`** is 1 if that employee left within 12 months after `observation_date`, else 0.
- Column groups: identifiers/dates, role & pay, history (raises, promotions), engagement/performance, attendance/leave (3-6 month windows), department context, the label, and six exit columns.
- The `source` column shows {'historical_extension': 2000, 'master_employee_snapshot': 389}: 383 real employees (30 Sep 2025 snapshot, with 6 repeated rows) plus 2,000 historical extension rows from 2021-2025.

## 2. Class imbalance (plain language)
349 of 2,389 people left = **14.61%**. A "model" that always says *will stay* is right 85.39% of the time - it looks excellent - yet it finds **zero** of the 349 leavers, so HR learns nothing. That is why this project reports ROC-AUC, PR-AUC, recall and precision instead of accuracy alone. Chart: `figures/class_balance.png`.

## 3. Data-quality results
**Missing values (predictors only, exit columns are blank for stayers by design):**

| column | missing | missing_pct |
|---|---|---|
| months_since_last_promotion | 1459 | 61.070 |
| last_engagement_score | 528 | 22.100 |
| last_raise_pct | 202 | 8.460 |
| last_performance_rating | 182 | 7.620 |
| commute_distance_km | 55 | 2.300 |
| highest_education | 15 | 0.630 |

**Duplicates:** 6 exact duplicate rows (employee ids ['EMP-0060', 'EMP-0237', 'EMP-0238', 'EMP-0325', 'EMP-0483', 'EMP-0570']). `employee_id` has 6 repeated ids, all from these rows.

**Impossible values found (rules that matched at least one row):**

| rule | columns | rows |
|---|---|---|
| negative tenure | tenure_months | 4 |
| negative total experience | total_experience_years | 10 |
| experience shorter than tenure | total_experience_years, tenure_months | 140 |
| attendance_rate_3m above 100% | attendance_rate_3m | 2 |
| wfh_share_3m above 100% | wfh_share_3m | 23 |
| wfh_share_3m below 0% | wfh_share_3m | 190 |

There is **no `age` column** in this table, so "age under 18" could not be tested (age lives in a separate protected-attributes sheet, which must be used for fairness checks only).

**Outliers - IQR rule (1.5 x IQR)** (top columns):

| column | lower | upper | iqr_outliers |
|---|---|---|---|
| avg_overtime_hours_per_month_3m | 0.000 | 0.000 | 446 |
| internal_transfers | 0.000 | 0.000 | 189 |
| commute_distance_km | -7.862 | 29.637 | 141 |
| promotions_received | -1.500 | 2.500 | 138 |
| job_level | 0.500 | 4.500 | 126 |
| training_hours_12m | -13.800 | 61.400 | 99 |

**Outliers - Z-score (|z| > 3)** (top columns):

| column | z_outliers_abs_gt_3 | max_abs_z |
|---|---|---|
| internal_transfers | 189 | 6.633 |
| manager_changes_12m | 97 | 8.364 |
| avg_overtime_hours_per_month_3m | 61 | 8.969 |
| promotions_received | 50 | 6.477 |
| commute_distance_km | 47 | 5.621 |
| training_hours_12m | 47 | 6.523 |

> Outliers are *investigation candidates*, not automatic errors. A very high `annual_fixed_pay` is a senior person; many `internal_transfers` or long commutes can be real. They are kept (with scaling / log-pay) unless they also break a logical rule above. Chart: `figures/outliers_boxplot.png`.

## 4. Leakage review (proof, not guesses)
Leakage = a column that secretly contains the answer, usually because it is only known after the person leaves.

**Confirmed leakage (dropped from all model inputs):**

| column | verdict | evidence |
|---|---|---|
| exit_date | CONFIRMED LEAKAGE | filled for 100.0% of leavers and 0.0% of stayers (filled-in flag AUC=1.0); every non-null date is after observation_date in 349/349 rows |
| resignation_date | CONFIRMED LEAKAGE | filled for 92.0% of leavers and 0.0% of stayers (filled-in flag AUC=0.9599); every non-null date is after observation_date in 279/321 rows |
| exit_reason | CONFIRMED LEAKAGE | filled for 100.0% of leavers and 0.0% of stayers (filled-in flag AUC=1.0) |
| exit_interview_done | CONFIRMED LEAKAGE | filled for 100.0% of leavers and 0.0% of stayers (filled-in flag AUC=1.0) |
| fnf_settlement_amount | CONFIRMED LEAKAGE | filled for 100.0% of leavers and 0.0% of stayers (filled-in flag AUC=1.0) |
| voluntary_left_within_12m | CONFIRMED LEAKAGE | derived from the exit event; 1 in 321 rows, all of them target=1 (rows with voluntary=1 and target=0: 0); cannot exist before the outcome |

**Suspected / excluded for safety:**

| column | verdict | evidence |
|---|---|---|
| source | SUSPECTED (excluded) | dataset-origin tag, not an employee attribute; label rates {'historical_extension': 0.15, 'master_employee_snapshot': 0.129}; single-column AUC=0.506 |
| observation_date | SUSPECTED (excluded) | time stamp of the snapshot (2021-2025 for historical rows); single-column AUC=0.504; excluded to avoid time proxies |

**Legitimate predictors:** 31 columns pass the checks. No single legitimate column predicts the label strongly (best single-column AUC = 0.600 for `months_since_last_raise`), which is consistent with an honest model scoring about 0.7. Chart: `figures/leakage_evidence.png`.

## 5. Audit table
17 distinct, evidence-backed findings (full table: `tables/audit_table.csv`, screenshot-ready image: `figures/audit_table.png`).

| problem | affected_column_or_location | rows_affected | severity | proposed_treatment |
|---|---|---|---|---|
| Post-outcome (leakage) columns | exit_date, resignation_date, exit_reason, exit_interview_done, fnf_settlement_amount, voluntary_left_within_12m | 349 | Critical | Drop from every model input |
| Exact duplicate snapshots | employee_id | 6 | High | Keep first occurrence, drop repeats |
| Impossible value: negative tenure | tenure_months | 4 | High | Set to missing, impute on training data only |
| Impossible value: negative total experience | total_experience_years | 10 | High | Set to missing, impute on training data only |
| Impossible value: experience shorter than tenure | total_experience_years, tenure_months | 140 | High | Set to missing, impute on training data only |
| Impossible value: attendance_rate_3m above 100% | attendance_rate_3m | 2 | Medium | Clip to the valid 0-1 range |
| Impossible value: wfh_share_3m above 100% | wfh_share_3m | 23 | Medium | Clip to the valid 0-1 range |
| Impossible value: wfh_share_3m below 0% | wfh_share_3m | 190 | Medium | Clip to the valid 0-1 range |
| Inconsistent city spellings | work_location | 13 | Medium | Map to canonical names |
| Missing values in predictors | months_since_last_promotion (61.07%), last_engagement_score (22.1%), last_raise_pct (8.46%), last_performance_rating (7.62%), commute_distance_km (2.3%), highest_education (0.63%) | 1459 | Medium | Median/most-frequent imputation fitted on training only + missing-indicator |
| Structural missingness: never promoted | months_since_last_promotion | 1459 | Low | Derive never_promoted flag; impute number |
| Class imbalance | left_within_12m | 349 | High | Stratified splits/folds, class weights, PR-AUC/recall instead of accuracy |
| Statistical outliers (IQR and /z/>3) | avg_overtime_hours_per_month_3m, internal_transfers, commute_distance_km | 446 | Low | Keep and investigate; use log pay/scaling; do not auto-delete |
| Mixed row origins and time span | source, observation_date | 2000 | High | Exclude source/date as inputs; report coverage of monthly features; ablate with/without them |
| Leaver had already resigned before the snapshot | resignation_date vs observation_date | 42 | Medium | Keep rows (cannot be detected from safe columns); report a sensitivity run excluding them |
| Duplicate payroll employee-month rows | payroll_monthly_features | 15 | Medium | Drop exact repeats before aggregating |
| Monthly records after the observation date | attendance/payroll months up to 2026-09 | 4985 | High | Aggregate only complete months up to observation_date |

## 6. Decisions carried to Day 2
Every column was classified drop / retain / transform (`tables/column_roles.csv`). Summary: {'RETAIN': 20, 'TRANSFORM': 11, 'DROP': 10, 'TARGET': 1}.

*Honest limits:* the audit only reflects checks that could be tested from the data. Anything the checks cannot see is handled in the vault comparison.
