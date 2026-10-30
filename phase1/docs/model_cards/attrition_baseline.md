# Model Card: CortexHR attrition baseline (balanced logistic regression)   (CX-20 draft from Phase 1)
**Purpose:** rank employees by estimated risk of leaving within 12 months so HR can prioritise conversations. **Not for:** hiring, firing, promotion, pay or any automatic employment decision; not for use on individuals without human review.

**Data:** sheet `attrition_dataset` (2,389 rows, snapshot dates 2021-01-01 to 2025-09-30); 6 duplicate rows removed (2,383 used); removed as inputs: 6 post-outcome columns (exit_date, resignation_date, exit_reason, exit_interview_done, fnf_settlement_amount, voluntary_left_within_12m), `source`, `observation_date`, ids; no protected attributes in the table. Attendance/payroll aggregates (last 3 complete months before the snapshot) exist for only 383 rows.

**Method:** leakage-safe `build_pipeline()` (clean, derived features incl. pay vs department median, raise-recency buckets, overtime flag; median impute + indicators, scale, one-hot); LogisticRegression(class_weight=balanced); stratified 80/20 split (seed 42) and stratified 5-fold CV on the 80%.

**Metrics** (official `cortexhr_helpers.binary_report`; compared with `baseline_benchmarks`):

| Measure | Mine | Benchmark |
|---|---|---|
| ROC-AUC (5-fold CV) | 0.709 | 0.713 |
| PR-AUC (5-fold CV) | 0.325 | 0.332 |
| ROC-AUC (held-out test) | 0.691 | |
| PR-AUC (held-out test, base rate 0.147) | 0.355 | |
| Precision / recall in the top-10% group (test) | 0.404 / 0.271 | |
| High band (test): leavers / employees | 19 / 48 (39.6%), holds 27% of all leavers | |

**Explanations:** strongest coefficients (standardised inputs):

| feature | coefficient |
|---|---|
| role_id_R012 | -1.696 |
| highest_education_M.A. Social Work | -1.546 |
| role_id_R015 | -1.507 |
| role_id_R044 | -1.386 |
| raise_recency_bucket_18m+ | 1.369 |
| role_id_R018 | 1.205 |
| role_id_R042 | -1.153 |
| department_id_D10 | -1.140 |

Worked example (highest-scored test employee, score 0.99): raw values {'tenure_months': 22.0, 'comp_ratio': 0.91, 'months_since_last_raise': 22.0, 'last_engagement_score': 1.4, 'avg_overtime_hours_per_month_3m': 0.0}. Largest contributions to the score: `last_engagement_score` (+1.84), `raise_recency_bucket_18m+` (+1.37), `work_mode_Onsite` (+0.72), `job_level` (+0.64).

**Fairness:** the table has no protected attributes, so group checks use available groups with the official `group_gap`: work_mode largest gap in mean score 0.189; department 0.218. Gaps are warnings, not proof; real fairness testing needs the protected-attributes data (age, gender), which is deliberately kept out of the model.

**Limits & risks:** only 70 test leavers (noise of several points); 84% of rows lack monthly history; 12% of leavers had already resigned before the snapshot; raw probabilities are inflated by class weights (use risk bands or calibrated scores); drift signs: score PSI above 0.25, prevalence far from 14.6%.

**Version & owner:** attrition-lr-balanced-v1.0 (2026-10-23); owner: Jaisurya Marisamy.
