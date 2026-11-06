# CortexHR

AI/ML project by **Jaisurya Marisamy** (Growmar, AI/ML track), built one phase per week.
All companies, people and datasets in this programme (Aura Workforce, Growmar, Zenith Nova) are fictional.

## Repository layout

```
cortexhr/
├── README.md        <- this file
├── .gitignore
├── phase1/          <- Phase 1: Data foundations and baseline models   (tag: phase-1)
├── phase2/          <- Phase 2: Core development                       (tag: phase-2)
└── phase3/          <- Phase 3: Intelligence                           (tag: phase-3)
```

Each phase folder is **self-contained**: it has its own `src/`, `tests/`, `reports/`, `requirements.txt` and README,
and runs on its own. Phase 2 builds on Phase 1, so `phase2/` also contains the Phase 1 code it reuses. Phase 3 imports the Phase 1 pipeline and folds.
Paths in each phase report are relative to that phase's folder.

| Phase | Folder | Git tag | What it delivers |
|---|---|---|---|
| 1 | [`phase1/`](phase1/) | `phase-1` | Data audit and leakage checks, feature pipeline, honest attrition baselines, risk bands, unsupervised attendance anomaly detection |
| 2 | [`phase2/`](phase2/) | `phase-2` | Payroll outlier detector (per-error-type results), reason codes and money at risk, FastAPI anomaly service, HR intent chatbot with slots, memory and permissions |
| 3 | [`phase3/`](phase3/) | `phase-3` | Resume ranking, fairness audit of the screener, attrition model v2, risk report with reasons for all 430 active employees, fair pay bands |

## Quick start (any phase)

```powershell
cd phase3                     # or phase2 / phase1
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pytest -q tests               # Phase 3 and Phase 1 tests (use `pytest -q tests_phase2` for Phase 2)
```

Run the Phase 2 services (two terminals, from `phase2/`):

```powershell
uvicorn mock_server:app --app-dir 05_Starter/mock_services --port 9000
uvicorn app.main:app --app-dir 05_Starter/api_skeleton --port 8000
```

## Phase 1 highlights

- Data audit: 17 evidence-backed findings and 6 confirmed leakage columns with proof; a deliberately leaky demo model hits ROC-AUC 1.00 and is flagged, while no honest model goes above 0.90.
- Leakage-aware feature pipeline: fits on training rows only, with cutoff-safe attendance and payroll aggregates.
- Honest attrition baselines (identical folds): logistic regression CV ROC-AUC 0.709 (benchmark 0.713). Held-out test, scored once: ROC-AUC 0.691, PR-AUC 0.355, top-10% lift 2.75x.
- Risk bands: High band caught 19 of 48 leavers (40%) vs 10% in the Low band; High band holds 27% of all leavers.
- Attendance anomaly detector (IsolationForest + rules, no labels used for training): precision 0.500, recall 0.840 (benchmark 0.403 / 0.593).
- Full classification workflow: 37 items covered in one notebook, with limitations stated openly.

## Phase 2 highlights

- Payroll detector: rules + IsolationForest, measured on all 8 error types (overall precision 0.316, recall 0.95; rules alone precision 0.860).
- Every flag has a reason code, a plain-language sentence and `money_at_risk`, in the official output format.
- FastAPI service with validated inputs (422 on bad input), models loaded once at start-up, versioned models.
- Chatbot: TF-IDF + Logistic Regression, 33 intents, held-out accuracy 0.8106, safe fallback, role-based answers; another employee's salary is always refused.
- Tests: 69 passed in `tests_phase2` (59 service-layer + 10 API).

## Phase 3 highlights

- Resume ranking (TF-IDF + skill overlap + capped experience, identifiers masked): precision@5 0.940, NDCG@5 0.923, MRR 1.000 on 555 labelled pairs across 10 jobs.
- Fairness audit: 25 twin pairs, 75 comparisons. Non-zero gap in 74 of 75 before masking; zero after masking (zero by construction, other proxies untested).
- Attrition v2 (LR + gradient boosting blend): CV ROC-AUC 0.716 vs 0.709 baseline. The gain is **not proven** (paired 95% interval includes 0). Calibrated test ECE 0.028.
- Risk report: 430 active employees scored (301 Low, 86 Medium, 43 High), each with a calibrated score, three plain-English reasons and a human-review note. Native SHAP cross-check on all 430 agrees on the top reason for 68.1%.
- Pay bands: Ridge on job-related features only (gender, age and career gap are never inputs), test MAE Rs 53,925, R2 0.983, band coverage 0.802 (target 0.80).
- Tests: 72 passed, 51 of 51 requirements traced. 15 notebooks with saved outputs.

## Reports

- Phase 1 report: see `phase1/reports/`
- Phase 2 report: `phase2/reports/phase2/AIML_Phase2_Report_Jaisurya_Marisamy.pdf`
- Phase 3 report: `phase3/reports/AIML_Jaisurya_Marisamy_Phase3_Report.pdf`

Restricted validation material (`04_Validation_Vault/` ) is never committed to this repository.
