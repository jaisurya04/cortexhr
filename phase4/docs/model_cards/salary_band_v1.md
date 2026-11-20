# Model Card: CortexHR pay-band advisor v1   (CX-20)

**Purpose** suggests a low / mid / high fixed-pay range for a role so a manager and HR can discuss an offer.  ·  **Not for** setting an individual's pay, or any decision made without a person.

**Data** `salary_training.csv` (past offers, 2,934 rows). Inputs are job-related only: department, role, level, band mid-point, market index, experience, skills, location, education. Gender, age, career gap, hire date and source are never inputs.

**Method** Ridge regression on log salary (chosen over gradient boosting); the band is the prediction scaled by out-of-fold error quantiles so the range is honest; test hold-out used once.

**Metrics**

| model | MAE_rupees | R2 | median_abs_pct_error |
|---|---|---|---|
| baseline: average offered salary by role | 115094.237 | 0.933 | 12.769 |
| baseline 2: role average adjusted for market index | 55656.281 | 0.982 | 5.521 |
| Ridge regression (job-related features) | 53924.807 | 0.983 | 5.397 |
| Gradient boosting (job-related features, tuned) | 57504.058 | 0.979 | 5.731 |
| Ridge regression + career_gap_months | 53160.626 | 0.984 | 5.451 |
| Gradient boosting + career_gap_months | 56955.579 | 0.98 | 5.657 |

Band coverage on the test set: **0.802** (target 0.80).

**Explanations** Worked example (example row from the training file): low ₹206,100, mid ₹229,500, high ₹254,200.

**Fairness** Mean (predicted pay / band reference) gap between best and worst group (`task15/gap_bin_gaps.csv`):

| measure | group_gap |
|---|---|
| actual offer / band mid-point (history) | 0.037 |
| model A predicted ratio | 0.046 |
| model B predicted ratio (chosen) | 0.009 |
| model A actual/predicted | 0.021 |
| model B actual/predicted | 0.039 |

The model that is *given* career gap would widen the career-gap gap (0.046); the chosen model, which never sees it, has a gap of 0.0085. The history itself contains a career-gap penalty, which the model must not copy.

**Limits & risks** Past offers may contain past bias; only offers of the last years; the gender comparison needs a restricted file that is not in the repository.

**Version & owner** `salary-band-v1` · card written 2026-11-20 · Jaisurya Marisamy (GM00134), AI/ML Engineer, Growmar

_Every output of this model is AI-generated and needs human review. Numbers in this card are copied from the evaluation files named in the text by `scripts/build_docs.py`._
