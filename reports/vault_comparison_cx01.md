# Vault comparison for CX-01 (opened AFTER the audit was committed)

Sequence proof: git commit `0e1d5ce` ("Day 1: data audit (completed BEFORE opening 04_Validation_Vault)") contains the finished audit table. Only then was `unlock_CX-01_answer_key_notes.csv` opened.

**Material actually available:** only the notes file. The row-level `answer_key_planted_dirt` file was **not** in the Drive vault, so the comparison is at the level of counts and categories, not row by row.

| Vault statement | My original audit | Verdict |
|---|---|---|
| Leakage: exit_date, resignation_date, exit_reason, exit_interview_done, fnf_settlement_amount | All five flagged CONFIRMED with proof (filled-in-flag AUC 0.96-1.00, timing) | Caught |
| (not listed) voluntary_left_within_12m | Flagged as derived from the exit event and dropped | Extra catch (logically post-outcome) |
| dept_attrition_rate_prior_12m, comp_ratio, months_since_last_raise, attendance windows are legitimate | Classified legitimate; best single-column AUC 0.60 | Agree |
| 6 exact duplicate rows | 6 found (EMP-0060, 0237, 0238, 0325, 0483, 0570) | Caught |
| 13 inconsistent categories | 13 rows / 16 spellings -> 9 cities | Caught |
| 4 negative tenure | 4 found | Caught |
| 2 impossible values | attendance_rate_3m above 100%: 2 rows | Caught |
| 70 missing values (planted) | Reported 6 predictor columns with blanks. 15 (highest_education) + 55 (commute_distance_km) = 70 are the planted ones | Partly missed: I did not separate accidental blanks from structural blanks (never promoted, no survey reply) |
| 3 salary typo outliers | Flagged skewed pay as "real senior staff, keep" | **MISSED** |
| Class imbalance ~85% stay | 85.39% accuracy of always-stay | Caught |

## What I missed and why
1. **Salary typos (3 rows).** I looked at `annual_fixed_pay` on its own scale, saw long-tailed pay, and wrongly decided every extreme was a genuine senior salary. The check that finds them uses two columns together: `annual_fixed_pay / comp_ratio` is the implied role mid-point and should be near constant inside a role. Rows 15, 217 and 345 are about 11x their role's median (a monthly-vs-annual style typo). *Lesson:* test consistency between related columns, not just each column alone. *Fix applied:* the pipeline now blanks pay that is more than 3x the training-set role mid-point and imputes it (`features.Derive`), and the audit module gained this check.
2. **Planted vs natural blanks.** I counted blanks but did not ask which are errors. *Lesson:* classify missingness as structural / accidental before choosing a treatment.

## Things I found that the vault does not list
- `wfh_share_3m` outside 0-1 (213 rows), `total_experience_years` negative or shorter than tenure (150 rows), 42 leavers who had already resigned before the snapshot, 15 duplicate payroll rows and 4,985 attendance/payroll rows dated after the snapshot. These come from the generator's noise or from table design, not from the planted list, but they still change how features must be cleaned and cut off, so they were kept in the audit and handled in the pipeline.

The vault answers were used only for this comparison, never as features or labels.
