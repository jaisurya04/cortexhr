# CortexHR Phase 2 — Payroll Anomaly Detection & HR Chatbot

**Name:** Jaisurya Marisamy  
**Track:** AI/ML  
**Phase:** 2  
**Repository:** `<paste your GitHub repository URL>`  
**Git tag:** `phase-2` (create after final review)

## Scope and status

| Task | Deliverable | Status | Evidence |
|---|---|---|---|
| 6 | Payroll feature engineering, rule baselines, Isolation Forest, per-type evaluation | Done; per-type CX-06 metrics measured (overall precision 0.316, recall 0.95), see `payroll_cx06_per_type.csv` | `05_Starter/notebooks/phase2/07`–`11`; `reports/phase2/` |
| 7 | Explanations, reason codes/text, money-at-risk ranking, sample-schema output | Implemented; inspect at least 20 flagged rows before submission | Notebook `12`; `payroll_anomaly_explanations.csv` |
| 8 | FastAPI attendance/payroll anomaly endpoints, validation and health check | Implemented; run tests and curl checks | `05_Starter/api_skeleton/app/main.py`; `tests_phase2/` |
| 9 | TF-IDF + Logistic Regression intent classifier, held-out test, confusion matrix, FAQ/fallback | Implemented; record executed metrics after running notebook | Notebook `14`; `chatbot_classification_report.csv` |

## Validation results

Run from repository root:

```bash
pip install -r 05_Starter/requirements.txt
pytest -q tests_phase2
uvicorn app.main:app --app-dir 05_Starter/api_skeleton --reload
```

Record the actual test summary and held-out classifier metrics here after execution. Never paste estimated or fabricated scores. Per-error-type payroll precision/recall is only valid when the authorized CX-06 label file is available and mapped to its real columns.

## Key design decisions

- Payroll labels are never accepted by the model-fit API; Isolation Forest is fit on engineered numeric features only.
- Identifiers are retained for joining output to records but excluded from the feature matrix.
- Rule-based and model-based flags are review signals, not proof of fraud or misconduct.
- `money_at_risk` is explicitly a prioritization proxy in this demo, not an audited financial-loss estimate.
- Chatbot confidence fallback and a hard privacy override prevent disclosure of another employee's salary.

## Learnings (complete after running)

1. Explain the most useful payroll ratio features and their edge cases.
2. Compare transparent rules with unsupervised detection and discuss false positives.
3. Describe the biggest intent-confusion pairs and the impact of confidence thresholding.
4. Explain how session slots and role checks change multi-turn behavior.

## Problems and fixes

Document reproducible problems encountered, the root cause, the exact fix, and the test/evidence that verifies the fix.

## Next phase

Integrate authorized mock services, improve entity resolution/disambiguation, validate financial exposure calculations with payroll owners, and add deployment/monitoring controls before any real-world use.
