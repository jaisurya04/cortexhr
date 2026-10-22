# Attendance anomaly report (Day 5)

## What was built
- Data: `attendance_monthly_features`, 7,339 employee-months for 528 employees. **No labels are read at any point of training or flagging.**
- **Model:** `IsolationForest(n_estimators=300, contamination=0.03, random_state=42)`. It builds random trees; an unusual row is separated from the others in very few splits, so it gets a high score. Contamination = the share of rows we choose to flag (3% -> 221 rows).
- **Features:** the 10 attendance columns (`scheduled_days, present_days, half_days_unpaid, paid_leave_days, unpaid_absence_days, longest_unpaid_streak_days, late_arrivals, avg_hours_worked, overtime_hours, attendance_rate`) plus three label-free comparisons with the same employee's own median month (attendance rate, late arrivals, hours) so that a *sudden* change for one person stands out even if the absolute value looks normal.
- **Rules (explainable, domain logic, no labels):** R1 zero attendance for a whole month (50 rows); R2 implausible hours, i.e. above 12 h/day or below 3 h on days present (44 rows).
- **Combination:** a row is flagged if the model OR a rule fires: 252 rows (3.4%): 221 model, 94 rule, 63 both. Rules add certainty for obvious cases; the model finds patterns nobody wrote a rule for.

## Ranking flags (think for yourself: who checks 1,000 rows?)
Flags are ranked by `priority_score` = model score + 1.0 if a rule fired, so rule-confirmed rows come first and the rest follow by how unusual they are. Reviewers work top-down within their capacity (`reports/tables/attendance_review_queue_top100.csv`, `review_rank` column in `attendance_flags.csv`). Two production ideas: suppress repeat flags for the same employee-month pattern, and add leave status so legitimate long absences (e.g. maternity leave, which the monthly table cannot show) are not escalated as anomalies; that is the main false-positive trap for the zero-attendance rule.

## Measured precision and recall (labels used for measurement only: `/home/claude/cortexhr_phase1/04_Validation_Vault/unlock_CX-05_attendance_anomaly_labels.csv`)

| detector | flagged | injected | caught | precision | recall |
|---|---|---|---|---|---|
| IsolationForest (raw features only) | 221 | 150 | 88 | 0.398 | 0.587 |
| IsolationForest (+own-history features) | 221 | 150 | 95 | 0.430 | 0.633 |
| Rules only | 94 | 150 | 50 | 0.532 | 0.333 |
| Combined (IF + rules) | 252 | 150 | 126 | 0.500 | 0.840 |
| POST-HOC v2 combined (optimistic) | 278 | 150 | 150 | 0.540 | 1.000 |

Reference (anomaly_baselines): precision 0.403, recall 0.593, 221 flagged, 150 injected.

### If reviewers can only check N rows per month
| review_capacity | flagged | caught | precision | recall |
|---|---|---|---|---|
| 50 | 50 | 15 | 0.300 | 0.100 |
| 100 | 100 | 56 | 0.560 | 0.373 |
| 150 | 150 | 94 | 0.627 | 0.627 |
| 221 | 221 | 124 | 0.561 | 0.827 |

### Anomaly types: which are hardest
| anomaly_type | injected | caught | recall | missed | model_only_recall | rule_only_recall |
|---|---|---|---|---|---|---|
| ABSENCE_STREAK | 40 | 40 | 1.000 | 0 | 1.000 | 0.000 |
| IMPOSSIBLE_HOURS | 35 | 35 | 1.000 | 0 | 0.114 | 1.000 |
| LATE_SPIKE | 30 | 6 | 0.200 | 24 | 0.200 | 0.000 |
| SUDDEN_DROP | 30 | 30 | 1.000 | 0 | 1.000 | 0.000 |
| ZERO_ATTENDANCE | 15 | 15 | 1.000 | 0 | 1.000 | 1.000 |

**Hardest type(s) for v1:** `LATE_SPIKE` (recall 0.20, 24 of 30 missed).

**Why (my reading):** a late-arrival spike is an extreme in ONE column, but the Isolation Forest scores combinations of columns: with only 3% of rows to flag, single-column spikes were out-ranked by rows that were odd on several columns, and no rule covered late arrivals.

**False alarms:** 126 of 252 flags were not injected anomalies. About 35 came from the zero-attendance rule: those months are fully covered by paid leave (e.g. maternity or long leave), which is legitimate. This is the known false-positive trap; leave status must be used.

**Post-hoc v2** (written after seeing this table, so its numbers are optimistic): R1 ignores months covered by paid leave and a new R3 flags late arrivals above the 99.5th percentile of the column (a label-free threshold). Compare the v2 row in the table above; do not present v2 as an honest test.


## Supplementary check on the labelled DAILY sample (a different table, measurement only)
`attendance_daily_examples` (2,000 daily records with labels, shipped inside the main workbook) was used only to measure a daily version of the same ideas; it does **not** replace the monthly measurement above. Anomaly types there: ODD_HOUR_LOGIN, PRESENT_WITH_ZERO_HOURS, IMPOSSIBLE_HOURS, CHECKOUT_BEFORE_CHECKIN (30 each).

| detector | flagged | injected | caught | precision | recall |
|---|---|---|---|---|---|
| Daily IsolationForest | 60 | 120 | 47 | 0.783 | 0.392 |
| Daily rule (hours/clock) | 93 | 120 | 90 | 0.968 | 0.750 |
| Daily combined | 107 | 120 | 93 | 0.869 | 0.775 |

Reading: simple clock/hour rules catch most daily anomalies with high precision; the Isolation Forest alone is weaker at 3% contamination. On the monthly table (measured above) the same pattern holds: the hours rule fully covers impossible-hours rows while the model covers streaks and sudden drops.

## Checklist
- Precision and recall reported: yes (monthly labels)
- Hard anomaly types noted: yes
- Never trained on labels: yes (`assert_no_labels`, test `test_attendance_never_uses_labels`)
- Rules and model explained: yes
