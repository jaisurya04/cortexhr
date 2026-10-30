# Feature definitions

All learned statistics are fitted on training rows only. Raw attrition-table predictors (retain/transform): `job_level`, `tenure_months`, `total_experience_years`, `previous_employers_count`, `commute_distance_km`, `annual_fixed_pay`, `comp_ratio`, `raises_received`, `last_raise_pct`, `months_since_last_raise`, `promotions_received`, `months_since_last_promotion`, `internal_transfers`, `last_performance_rating`, `last_engagement_score`, `manager_changes_12m`, `training_hours_12m`, `attendance_rate_3m`, `late_arrivals_3m`, `avg_overtime_hours_per_month_3m`, `avg_hours_worked_3m`, `wfh_share_3m`, `unpaid_leave_days_6m`, `paid_leave_days_6m`, `dept_attrition_rate_prior_12m`, `department_id`, `role_id`, `work_location`, `work_mode`, `employment_type`, `highest_education`.

Dropped from every model input (never enter the pipeline): `left_within_12m`, `row_id`, `employee_id`, `observation_date`, `exit_date`, `resignation_date`, `exit_reason`, `exit_interview_done`, `fnf_settlement_amount`, `voluntary_left_within_12m`, `source`.

## Derived and aggregated features

| feature | source | definition | temporal cutoff |
|---|---|---|---|
| pay_vs_dept_median | derived | annual pay / median pay of same department+job level; medians fitted on training rows only | snapshot at observation_date |
| raise_stagnation_ratio | derived | months_since_last_raise / (tenure_months + 1) | snapshot at observation_date |
| never_promoted | derived | 1 if promotions_received == 0 (explains the blank months_since_last_promotion) | snapshot at observation_date |
| log_annual_pay | derived | log1p(annual_fixed_pay) after blanking pay-typos (>3x role mid-point) | snapshot at observation_date |
| att_attendance_rate_mean_3m | attendance_monthly_features | mean attendance_rate over last 3 complete months | months with month_end <= observation_date |
| att_unpaid_absence_sum_3m | attendance_monthly_features | sum unpaid_absence_days over last 3 months | same |
| att_late_arrivals_sum_3m | attendance_monthly_features | sum late_arrivals over last 3 months | same |
| att_overtime_mean_3m | attendance_monthly_features | mean overtime_hours over last 3 months | same |
| att_avg_hours_mean_3m | attendance_monthly_features | mean avg_hours_worked over last 3 months | same |
| att_wfh_share_3m | attendance_monthly_features | sum wfh_days / sum present_days over last 3 months | same |
| att_half_days_sum_3m | attendance_monthly_features | sum half_days_unpaid over last 3 months | same |
| pay_variable_pay_mean_3m | payroll_monthly_features | mean variable_pay over last 3 months (exact duplicate rows removed first) | same |
| pay_lop_days_sum_3m | payroll_monthly_features | sum loss-of-pay days over last 3 months | same |
| pay_overtime_pay_mean_3m | payroll_monthly_features | mean overtime_pay over last 3 months | same |
| pay_variable_share_3m | payroll_monthly_features | sum variable_pay / sum gross_pay over last 3 months | same |
| pay_net_to_gross_3m | payroll_monthly_features | sum net_pay / sum gross_pay over last 3 months | same |

## Pipeline steps
1. `RawCleaner`: stateless fixes (canonical city, negative tenure/experience and experience<tenure -> missing, rates clipped to 0-1).
2. `Derive`: pay-typo blanking (pay > 3x training role mid-point), department+level relative pay (training medians), stagnation ratio, never-promoted flag, log pay.
3. `ColumnTransformer`: numeric -> median impute (+missing indicators) -> StandardScaler; monthly aggregates -> median impute -> scale; categorical -> 'Unknown' impute -> one-hot (unseen/rare categories bucketed, never an error).

**Temporal cutoff for monthly aggregates:** only complete months whose last day is on/before `observation_date`, last 3 such months. 2,000 historical rows have no monthly history (NaN -> training-median impute); only 383 rows have it.
