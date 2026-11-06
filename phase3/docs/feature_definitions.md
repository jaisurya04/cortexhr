# Feature definitions

## Resume ranking (per resume x job)
| Feature | Definition | Range |
|---|---|---|
| tfidf_cosine | cosine similarity between the masked resume document and the job document | 0..1 |
| tfidf_scaled | cosine scaled by the job's min and max over the corpus | 0..1 |
| required_coverage | matched required skills / required skills | 0..1 |
| nice_coverage | matched nice-to-have skills / nice-to-have skills | 0..1 |
| total_years | sum of role durations found in the text (duplicates ignored) | years |
| relevant_years | years in roles whose title overlaps the job title | years |
| experience_score | 0.5 x min(1, total / minimum) + 0.5 x min(1, relevant / minimum) | 0..1 |
| score | weighted sum (weights in config) | 0..1 |

## Attrition v2
The 43 raw inputs of the Phase 1 pipeline (HR snapshot fields plus cut-off-safe 3-month attendance and payroll aggregates). Derived in the pipeline: pay versus department median, raise-stagnation ratio, overtime flag, missing indicators, scaling, one-hot encoding. See `docs/phase1` notes in the Phase 1 folder. For explanations the inputs are folded into 23 groups (`src/cortexhr/phase3/explanations/groups.py`).

## Salary
| Feature | Definition |
|---|---|
| log_band_mid | natural log of the 2026 band mid-point of the role and level |
| log_market_index | natural log of the market index at the hiring date |
| job_level, department_id, role_id, work_location, highest_education | as in the file |
| prior_experience_years, skills_count | as in the file |
| pay_vs_band (for fairness only) | offered salary / (band mid-point x market index) |
| career_gap_months | NOT a model input in the chosen model; used in the comparison model only |
