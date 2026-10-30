# Vault comparison for CX-01 (opened AFTER the audit was committed)

**Sequence proof.** git commit `0e1d5ce` ("Day 1: data audit (completed BEFORE opening 04_Validation_Vault)") holds the finished audit table. Only then were the vault files opened: first `unlock_CX-01_answer_key_notes.csv`, later (once supplied) the row-level `unlock_CX-01_answer_key_planted_dirt.csv`. The answers were never used as model inputs.

## Row-level result (planted dirt vs my audit code)
| issue_type | planted_rows | found_by_current_audit_code | recall | found_in_ORIGINAL_audit |
|---|---|---|---|---|
| DUPLICATE_ROW | 6 | 6 | 1.000 | yes |
| IMPOSSIBLE_VALUE | 2 | 2 | 1.000 | yes |
| INCONSISTENT_CATEGORY | 13 | 13 | 1.000 | yes |
| MISSING_VALUE | 70 | 70 | 1.000 | yes |
| NEGATIVE_TENURE | 4 | 4 | 1.000 | yes |
| SALARY_TYPO_OUTLIER | 3 | 3 | 1.000 | no (missed) |

*Found by current audit code* = the same row ids are flagged by the audit/cleaning functions in `src/cortexhr/audit.py` and `data_loading.py`.
Every planted issue is caught at row level **except the 3 salary typos, which my original audit missed**.

## Counts from the notes file
| Vault statement | My original audit | Verdict |
|---|---|---|
| Leakage: exit_date, resignation_date, exit_reason, exit_interview_done, fnf_settlement_amount | all five flagged CONFIRMED with proof | Caught |
| (not listed) voluntary_left_within_12m | flagged as derived from the exit event | Extra catch |
| dept_attrition_rate_prior_12m, comp_ratio, months_since_last_raise, attendance windows are legitimate | classified legitimate (best single-column AUC 0.60) | Agree |
| 6 duplicates / 13 inconsistent categories / 4 negative tenure / 2 impossible values | all found | Caught |
| 70 missing values (15 highest_education + 55 commute_distance_km) | reported, but lumped with structural blanks | Partly missed (did not separate accidental from structural) |
| 3 salary typo outliers | called "real senior pay, keep" | **MISSED** |

## What I missed and why
1. **Salary typos (3 rows).** I looked at `annual_fixed_pay` on its own scale and decided every extreme was real senior pay. The check that finds them compares two related columns: `annual_fixed_pay / comp_ratio` is the role mid-point and should be near constant inside a role; rows 15, 217 and 345 are about 11x their role (a monthly-vs-annual style typo). *Lesson:* test consistency between related columns, not just each column alone. *Fix:* `audit.pay_consistency_outliers` and the pipeline blanks pay above 3x the training role mid-point (`features.Derive`).
2. **Planted vs structural blanks.** I counted blanks but did not ask which are errors and which carry information (never promoted, no survey reply). *Lesson:* classify missingness before choosing a treatment.

## Found by me, not in the planted list
| check | rows |
|---|---|
| wfh_share_3m outside 0-1 | 213 |
| total_experience below tenure/negative | 150 |
| leavers who had already resigned before the snapshot | 42 |

These come from generator noise or table design rather than the planted list, but they still change how features must be cleaned and cut off.
