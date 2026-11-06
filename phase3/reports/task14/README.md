# Task 14: explain every person's risk
Notebooks 10 and 11. 430 active employees scored with attrition-v2; bands {'Low': 301, 'Medium': 86, 'High': 43}; mean risk 0.127.
**Reasons in the 430-row report:** real `shap` library (PermutationExplainer, shap 0.54.1, all 430 employees, 50-row background) folded into 23 HR groups; additivity error 4.4e-16. Produced by `scripts/run_shap_full430.py` (or notebook 10) through `run_task14`; run data in `explanation_run.json`.
**Format checks:** 14 of 14 passed (columns, order, ranges, bands, three distinct reasons, human-review note, wording).
**Output:** `attrition_risk_2026.csv` (reasons from the shap library). Manual review of 10 rows and the wording changes it caused are in notebook 11.