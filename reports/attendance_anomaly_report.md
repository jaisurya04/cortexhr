# Attendance anomaly report (Day 5)

## What was built
- Data: `attendance_monthly_features`, 7,339 employee-months for 528 employees. **No labels are read at any point of training or flagging.**
- **Model:** `IsolationForest(n_estimators=300, contamination=0.03, random_state=42)`. It builds random trees; an unusual row is separated from the others in very few splits, so it gets a high score. Contamination = the share of rows we choose to flag (3% -> 221 rows).
- **Features:** the 10 attendance columns (`scheduled_days, present_days, half_days_unpaid, paid_leave_days, unpaid_absence_days, longest_unpaid_streak_days, late_arrivals, avg_hours_worked, overtime_hours, attendance_rate`) plus three label-free comparisons with the same employee's own median month (attendance rate, late arrivals, hours) so that a *sudden* change for one person stands out even if the absolute value looks normal.
- **Rules (explainable, domain logic, no labels):** R1 zero attendance for a whole month (50 rows); R2 implausible hours, i.e. above 12 h/day or below 3 h on days present (44 rows).
- **Combination:** a row is flagged if the model OR a rule fires: 252 rows (3.4%): 221 model, 94 rule, 63 both. Rules add certainty for obvious cases; the model finds patterns nobody wrote a rule for.

## Ranking flags (think for yourself: who checks 1,000 rows?)
Flags are ranked by `priority_score` = model score + 1.0 if a rule fired, so rule-confirmed rows come first and the rest follow by how unusual they are. Reviewers work top-down within their capacity (`reports/tables/attendance_review_queue_top100.csv`, `review_rank` column in `attendance_flags.csv`). Two production ideas: suppress repeat flags for the same employee-month pattern, and add leave status so legitimate long absences (e.g. maternity leave, which the monthly table cannot show) are not escalated as anomalies; that is the main false-positive trap for the zero-attendance rule.

## Measured precision and recall

**NOT MEASURED YET.** The hidden attendance labels (`unlock_CX-05` / `attendance_anomaly_labels`) were not in the supplied `04_Validation_Vault` Drive folder, which contained only `unlock_CX-01_answer_key_notes.csv`. No number below is invented. To measure: put the labels file in `04_Validation_Vault/` and run `python scripts/detect_attendance_anomalies.py` (it auto-detects the file). The flags were already saved to `reports/tables/attendance_flags.csv` before any label is read, so measuring cannot change them.

Reference to beat (anomaly_baselines): precision 0.403, recall 0.593, 221 flagged, 150 injected.

## Supplementary check on the labelled DAILY sample (a different table, measurement only)
`attendance_daily_examples` (2,000 daily records with labels, shipped inside the main workbook) was used only to measure a daily version of the same ideas; it does **not** replace the monthly measurement above. Anomaly types there: ODD_HOUR_LOGIN, PRESENT_WITH_ZERO_HOURS, IMPOSSIBLE_HOURS, CHECKOUT_BEFORE_CHECKIN (30 each).

| detector | flagged | injected | caught | precision | recall |
|---|---|---|---|---|---|
| Daily IsolationForest | 60 | 120 | 47 | 0.783 | 0.392 |
| Daily rule (hours/clock) | 93 | 120 | 90 | 0.968 | 0.750 |
| Daily combined | 107 | 120 | 93 | 0.869 | 0.775 |

Reading: simple clock/hour rules catch most daily anomalies with high precision; the Isolation Forest alone is weaker at 3% contamination. The hypotheses to verify on the monthly labels: hours rules cover impossible-hours cases, whereas absence streaks, late spikes and sudden drops depend on the model and own-history features and are the likelier misses. Treat these as hypotheses, not results.

## Checklist
- Precision and recall reported: daily proxy only; monthly labels missing from the supplied vault
- Hard anomaly types noted: hypotheses only until monthly labels are supplied
- Never trained on labels: yes (`assert_no_labels`, test `test_attendance_never_uses_labels`)
- Rules and model explained: yes
