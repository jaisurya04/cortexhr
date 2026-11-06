# Data dictionary: Phase 3 outputs

**reports/task14/attrition_risk_2026.csv** (430 rows, same columns as `sample_attrition_risk.csv`)
| Column | Meaning |
|---|---|
| prediction_id | AR-0001 .. AR-0430, in the order of the scoring file |
| employee_id, department_id | from attrition_scoring_2026 |
| as_of_date | observation date of the snapshot (2026-09-30) |
| risk_score_12m | calibrated probability of leaving within 12 months (0 to 1) |
| risk_band | Low / Medium / High from the supplied risk_bands(): High = top 10%, Medium = next 20% |
| top_reason_1..3 | the three groups of inputs that raised this score most compared with an average employee, as plain sentences |
| model_version | attrition-v2 |
| model_note | human-review note and how to read bands and reasons |

**reports/task14/group_contributions.csv**: contribution of each of 23 explanation groups to each employee's score (probability points), the baseline value and the prediction; contributions plus base equal the prediction.

**reports/task11/top5_per_job_all_resumes.csv**: job_id, rank, resume_id, score, tfidf_scaled, required_coverage, nice_coverage, experience_score, matched_required, missing_required, total_years, relevant_years, recommendation, human_review.

**reports/task12/twin_pair_results.csv**: pair_id, job_id, score_original, score_twin, gap (twin minus original), abs_gap, relative_gap_pct, ranks, twin_ranked_higher, attribute_changed, screener (v1 before / v2 after), plus the hidden attributes of both twins when the restricted file was available.

**reports/task15/band_recommendations_test.csv**: hire_id, role_id, job_level, hire_year, recommended_low / mid / high (rupees per year), actual_offer, inside_band.

**artifacts/metadata/*.json**: model version, creation time, configuration, features, metrics, environment versions and sha256 of each saved model.
