# Model Card: CortexHR attrition v2 (12-month leave risk)   (CX-20)

**Purpose** estimates the chance that an active employee leaves within 12 months so HR can prioritise conversations.  ·  **Not for** dismissal, pay, promotion, hiring or any decision about a person without human review; not as a certainty.

**Data** `attrition_dataset.csv` (2,383 rows after removing 6 exact duplicates; snapshots 2021-01 to 2025-09); post-outcome columns, ids and `source` are removed as inputs; no protected attribute is an input. Scoring population: `attrition_scoring_2026.csv`, 430 active employees (as of 2026-09-30). Source evidence: `phase3/reports/task13/`.

**Method** Phase 1 leakage-safe pipeline, then an equal-weight blend of balanced logistic regression and tuned HistGradientBoosting, calibrated with a sigmoid learned inside the folds; stratified 5-fold CV (seed 42) plus one untouched test split. A separate LightGBM challenger was tested on the same folds.

**Metrics**

Cross-validated, same folds (`phase3/reports/task13/cv_summary_with_lightgbm.csv`):

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

Paired fold differences against the baseline (`paired_differences.csv`):

| metric | a | b | mean_diff | sd_diff | boot_ci_low | boot_ci_high | folds_a_better | n_folds |
|---|---|---|---|---|---|---|---|---|
| roc_auc | attrition-v2 candidate: LR + tuned HGB blend | Logistic Regression (balanced) [baseline] | 0.007 | 0.012 | -0.002 | 0.016 | 3 | 5 |
| pr_auc | attrition-v2 candidate: LR + tuned HGB blend | Logistic Regression (balanced) [baseline] | 0.001 | 0.028 | -0.018 | 0.025 | 2 | 5 |
| roc_auc | HistGradientBoosting tuned | Logistic Regression (balanced) [baseline] | -0.015 | 0.018 | -0.029 | -0.0 | 1 | 5 |
| pr_auc | HistGradientBoosting tuned | Logistic Regression (balanced) [baseline] | -0.015 | 0.06 | -0.052 | 0.036 | 1 | 5 |

**Verdict (Phase 3):** the gain over the simple logistic baseline is **not proven** (the 95% interval includes 0). Test split (used once): ROC-AUC 0.704, PR-AUC 0.382, Brier 0.113, ECE 0.028. Risk bands on the 2026 population: {'Low': 301, 'Medium': 86, 'High': 43}.

**Explanations** Top permutation drivers (AUC drop):

| feature | auc_drop | sd |
|---|---|---|
| last_engagement_score | 0.0644 | 0.0137 |
| months_since_last_raise | 0.056 | 0.0091 |
| annual_fixed_pay | 0.031 | 0.0124 |
| manager_changes_12m | 0.0306 | 0.0051 |
| total_experience_years | 0.0261 | 0.0083 |

Worked example `EMP-0002` (D08): risk 11.0% (Low); reasons shown to HR: The latest engagement score (2.2 out of 5) raised the predicted risk.; The role the person holds raised the predicted risk; this describes common patterns in the role, not the individual.; The attendance pattern over the last 3 months (attendance 96%, 1 unpaid absence days) raised the predicted risk.. Reasons come from the real `shap` library (PermutationExplainer 0.54.1) on 23 HR groups and show only factors that raised the score.

**Fairness** Group tables from `task13/group_gender.csv` and `group_age.csv` (test predictions, n per group):

| gender | n | actual_leave_rate | mean_predicted_risk | recall_of_leavers_in_top10 | auc |
|---|---|---|---|---|---|
| Female | 1011 | 0.155 | 0.153 | 0.261 | 0.705 |
| Male | 1311 | 0.139 | 0.155 | 0.275 | 0.722 |
| Other | 61 | 0.164 | 0.153 | 0.3 | 0.747 |

| age_band | n | actual_leave_rate | mean_predicted_risk | recall_of_leavers_in_top10 | auc |
|---|---|---|---|---|---|
| 25-34 | 1468 | 0.144 | 0.153 | 0.275 | 0.723 |
| 35-44 | 627 | 0.155 | 0.153 | 0.258 | 0.7 |
| 45+ | 28 | 0.107 | 0.1 | 0.0 | nan |
| <25 | 260 | 0.146 | 0.165 | 0.289 | 0.726 |

Mean predicted risk is similar across groups; the age band 45+ has only 28 people so no conclusion is possible. Similar averages do not prove equal treatment of individuals. Gender and age are never model inputs.

**Limits & risks** AUC around 0.70 means many leavers are missed and many stayers flagged. **Input drift (2026):** the latest monitoring run reports an ALERT for `months_since_last_raise` (see `/v1/monitoring`), so 2026 scores that rely on this input should be read with extra caution until the cause is understood. Class-weighted models overstate risk until calibrated.

**Version & owner** `attrition-v2` · card written 2026-11-20 · Jaisurya Marisamy (GM00134), AI/ML Engineer, Growmar

_Every output of this model is AI-generated and needs human review. Numbers in this card are copied from the evaluation files named in the text by `scripts/build_docs.py`._
