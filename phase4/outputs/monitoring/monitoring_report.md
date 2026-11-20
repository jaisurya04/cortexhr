```
==============================================================================
CortexHR monitoring report   2026-11-18T08:54:53Z   status: ALERT   coverage: partial
==============================================================================

FEATURE DRIFT (PSI) - attrition_dataset.csv vs attrition_scoring_2026.csv
feature                                PSI  status    ref mean   cur mean
last_engagement_score                0.063  OK            3.44       3.52
months_since_last_raise              3.835  ALERT        10.99       5.01
annual_fixed_pay                     0.159  WATCH   1029199.02 1262075.47
manager_changes_12m                  0.001  OK            0.31       0.34
total_experience_years               0.051  OK            8.26       9.11
comp_ratio                           0.104  WATCH         1.03       1.01
last_performance_rating              0.008  OK            3.32       3.42
commute_distance_km                  0.017  OK           12.41      11.77

FAIRNESS RE-CHECKS
  Note: months_since_last_raise has collapsed rather than drifted: 77% of the current population share the value 5 (reference mean 11.0, current mean 5.0). Check whether a company-wide event (for example a raise cycle) or a pipeline change explains it before trusting scores that use it.
  Twin resumes: 75 comparisons, max gap 0.000000 (tolerance 0.001); status OK.
  Pay gap by career_gap: model gap 0.0085 vs history 0.0368; status OK.
  Pay gap by gender: restricted CX-15 file not found (set CORTEXHR_VAULT_DIR); gender pay gap NOT checked

NOT CHECKED: pay_gap/gender

WHY: Drift: 8 features checked, max PSI 3.835 (months_since_last_raise); 3 above the WATCH line: months_since_last_raise 3.835, annual_fixed_pay 0.159, comp_ratio 0.104

Limits: The masked screener removes name, e-mail, city and college-tier text, so a zero gap is expected by construction. It does not test other proxies (for example hobbies or employer names).; Compares group means of predicted pay relative to the role band; it does not control for every legitimate factor and small groups are not compared. A small gap does not prove the offers are fair.

ALERT - NEXT STEPS
1. Confirm it is real: check the data pipeline, units and column mapping before blaming the world.
2. Locate it: which features and which departments moved, and since which month?
3. Measure impact: compare recent model scores and, once outcomes arrive, accuracy and calibration against the baseline.
4. Decide and record: recalibrate, retrain or roll back, keep a human override in place, and write the decision in the audit log.
==============================================================================
```
