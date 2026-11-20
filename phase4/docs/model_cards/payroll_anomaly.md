# Model Card: CortexHR payroll anomaly detector   (CX-20)

**Purpose** flags payslips that look unusual so a payroll clerk reviews them before payout.  ·  **Not for** proof of error or fraud, deductions, discipline or any automatic payroll action.

**Data** `payroll_monthly_features.csv`: 7,355 payslips, 18 monthly runs (2025-04 to 2026-09), synthetic. Training is unsupervised (no labels). Evidence: `phase2/reports/phase2/workflow38/payroll_model_documentation.md`.

**Method** 23 numeric features (ratios, identity gaps, amounts); median imputation, scaling; Isolation Forest (300 trees, contamination 0.03, seed 42) combined with rule checks (negative / zero / duplicate pay, gross-minus-deductions identity). Temporal split: train 2025-04..2026-03, validation 2026-04..06, test 2026-07..09.

**Metrics**

Test split (evaluated once, against synthetically injected errors, not the vault labels): precision 0.541, recall 0.958, F1 0.692, ROC-AUC 0.982, PR-AUC 0.702, flag share 6.5%. Production operating point flags 5.68% of real rows.

**Explanations** Each flag carries reason codes and an anomaly score; the API returns them with every result (`/v1/anomaly/payroll`).

**Fairness** Phase 2 fairness screen: 3 groups flagged more than 2x the overall rate (overall 5.68%). Flags are review candidates; reviewers should check the over-flagged groups.

**Limits & risks** Precision depends on how common errors really are; synthetic data; money-at-risk is a proxy; known weak spot: inflated variable pay. Monthly PSI/flag-rate monitoring raised alerts in 2025-04, 2025-10, 2026-02 and 2026-04.

**Version & owner** `payroll-iforest-rules-v2` · card written 2026-11-20 · Jaisurya Marisamy (GM00134), AI/ML Engineer, Growmar

_Every output of this model is AI-generated and needs human review. Numbers in this card are copied from the evaluation files named in the text by `scripts/build_docs.py`._
