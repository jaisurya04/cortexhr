# Model card: salary-band-v1
**Intended use.** Advice to a manager and HR: a low / mid / high annual fixed salary band for a role at hiring. The final decision stays with people. **Not for:** an automatic offer, or setting pay of existing staff.
**Model.** Ridge regression (job-related features) on the log salary, inputs job_level, log_band_mid, log_market_index, prior_experience_years, skills_count, department_id, role_id, work_location, highest_education. Excluded as not job-related: career_gap_months. Never inputs: hire_id, hire_date, source, gender, age_years, age_band, offered_annual_fixed.
**Accuracy (test set, used once).** MAE Rs 53,925, R2 0.983; baseline 'average by role' MAE Rs 115,094, R2 0.933; role average adjusted for market index MAE Rs 55,656.
**Band.** mid x exp(-0.108) to mid x exp(0.102) (10th and 90th percentile of out-of-fold log error); coverage on the test set 0.802.
**Fairness.** See `reports/task15/README.md` (gender and career-gap gaps, measured with a restricted file).
**Limits.** Advice for a manager and HR, never an offer. The target is historical offers and may contain past bias. See `docs/limitations.md`.