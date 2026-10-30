# Daily Log (Week 1 draft: replace the dates and edit so it is truly yours)

**Date:** Day 1  **Done today:** set up repo and venv; audited attrition_dataset (17 findings, 6 leakage columns, class balance 14.6%); committed the audit before opening the vault.  **Doing next:** feature pipeline.  **Blocked by:** nothing.  **One thing I learned:** compare related columns (pay / comp_ratio), not only each column alone.

**Date:** Day 2  **Done today:** build_pipeline() with train-only fitting, attendance/payroll aggregates with a date cutoff, 3-row smoke test.  **Doing next:** baselines.  **Blocked by:** nothing.  **One thing I learned:** monthly aggregates exist for only 383 of 2,383 rows, so they barely move the score.

**Date:** Day 3  **Done today:** Dummy, logistic regression, tree, forest on identical folds; compared with benchmark; SMOTE comparison.  **Doing next:** evaluation.  **Blocked by:** nothing.  **One thing I learned:** a deliberately leaky model scores 1.00; anything above 0.90 means "find the leak".

**Date:** Day 4  **Done today:** official binary_report and risk_bands, calibration (raw balanced scores are inflated), lift chart, manager note.  **Doing next:** anomalies.  **Blocked by:** nothing.  **One thing I learned:** show risk bands, not raw probabilities.

**Date:** Day 5  **Done today:** IsolationForest + two rules; measured with hidden labels after saving flags (precision 0.50, recall 0.84).  **Doing next:** Phase 2 payroll.  **Blocked by:** nothing.  **One thing I learned:** a rule fired on months fully covered by paid leave: legitimate absences are the false-positive trap.
