# Day 18 - Monitoring policy

## What is checked
| Check | Method | OK / WATCH / ALERT |
|---|---|---|
| Input drift | PSI between `attrition_dataset` (training reference) and `attrition_scoring_2026` on the **8 most important numeric features** (Phase 3 permutation importance, `auc_drop`), using the project's `psi()` helper | PSI < 0.10 / 0.10 to 0.25 / > 0.25 |
| Twin-resume gap | 25 twin pairs (75 comparisons) scored by the masked ranker; largest absolute score gap | 0 gaps OK; any tiny gap WATCH; gap > 0.001 ALERT |
| Pay gap by career gap | mean (predicted pay / band reference) per career-gap group, out-of-sample predictions as in Phase 3; groups under 30 people are not compared; fewer than 2 comparable groups is WATCH, never OK | gap < 0.02 / >= 0.02 / >= 0.05 |
| Pay gap by gender | same, needs the restricted CX-15 file (`CORTEXHR_VAULT_DIR`) | if the file is absent the check is **SKIPPED and reported as "not checked"** |

The overall status is the worst judged status. Skipped or failed checks are listed under `not_checked` / `errors` and make `coverage` "partial"; a failed check counts as WATCH, never OK.

## What a result does and does not mean
* A PSI alert shows that inputs moved, not that the model is wrong. A passed fairness check shows that this test found no gap above the threshold, **not that the system is fair**. The twin gap is zero after masking by construction.
* Thresholds for PSI come from the task. Fairness thresholds (0.02 / 0.05 / 0.001) are this project's policy choices, written here so they can be challenged.

## Schedule and records
* `python scripts/monitor.py` prints the report and exits 0 / 1 / 2. `deploy/github-actions-monitor.yml` runs it weekly; copy it to `.github/workflows/`.
* Every run appends one line to `outputs/monitoring/monitoring_history.jsonl` (timestamp, status, component statuses, max PSI) and rewrites `monitoring_latest.json` and `monitoring_report.md`.

## Think for yourself: "If PSI shows an ALERT, what would you do first?" (4-step plan)
1. **Confirm it is real.** Check the data pipeline, column mapping and units before blaming the world.
2. **Locate it.** Which features, which departments, since which month? (The report shows the shift and, when one value dominates, says so.)
3. **Measure the impact.** Compare recent scores, and once outcomes arrive, accuracy and calibration against the baseline.
4. **Decide and record.** Recalibrate, retrain or roll back; keep a human override in place; write the decision in the audit log.

## The latest finding (from the saved run)
`months_since_last_raise` is the second most important attrition input. In the 2026 scoring table 77% of employees share one value (5 months), so its distribution **collapsed rather than drifted** (PSI 3.84; exact numbers in `outputs/monitoring/monitoring_report.md`). Five months before the 30 Sep 2026 snapshot is April 2026, the month of the only 2026 event in `workforce_events.csv` (EV-09, a performance bonus paid with April payroll). The events file does **not** say that raises were given then, so this is a hypothesis to verify with HR (step 1 of the plan): did an April increment cycle reset the "last raise" date, or did the pipeline change how it is calculated? Until that is answered, read 2026 attrition scores that lean on this input with extra caution.
