# Week 1 demo script (2 minutes) - follows WEEKLY_DEMO_CHECKLIST.md
Draft; say it in your own words and be ready to explain every line.

1. **Who:** "I am Jaisurya Marisamy, AI/ML track. This week I finished CX-01 to CX-05: audit, pipeline, baselines, evaluation, attendance anomalies."
2. **Show it working (not slides):** run `python scripts/run_pipeline.py` (3-row smoke test passes) and open `reports/figures/audit_table.png` and `reports/figures/gains_lift_chart.png`.
3. **One number that proves it:** logistic regression ROC-AUC 0.709 vs benchmark 0.713; the leaky demo scored 1.00 (that is why I removed 6 columns); High risk band: 19 of 48 left (40% vs 15% overall). Attendance detector: precision 0.50, recall 0.84 vs benchmark 0.40 / 0.59.
4. **A problem I hit:** I called all salary extremes "real senior pay" in my audit and missed 3 salary typos; the vault showed pay/comp_ratio was ~11x its role. Lesson: compare related columns, not just single columns. (Also: late-arrival spikes were the hardest anomaly type; zero-attendance rule fired on paid leave.)
5. **Next week:** payroll anomalies (Phase 2), reusing the same measurement discipline.
