# CortexHR

AI/ML project by **Jaisurya Marisamy** (Growmar, AI/ML track), built one phase per week.  
All companies, people and datasets in this programme (Aura Workforce, Growmar, Zenith Nova) are fictional.

## Repository layout

```text
cortexhr/
├── README.md
├── .gitignore
├── phase1/          # Phase 1: Data foundations and baseline models
├── phase2/          # Phase 2: Core development
└── phase3/          # Phase 3: Intelligence
```

All three phases are maintained in this **single GitHub repository**.

Each phase has its own source code, dependencies, tests, reports and README. Phase 2 includes the Phase 1 code it reuses so that Phase 2 can run independently. Phase 3 reuses the Phase 1 feature pipeline and fold definitions.

Paths in each phase report are relative to that phase's folder.

| Phase | Folder | Git tag | What it delivers |
| --- | --- | --- | --- |
| 1 | [`phase1/`](phase1/) | `phase-1` | Data audit and leakage checks, feature pipeline, honest attrition baselines, risk bands, unsupervised attendance anomaly detection |
| 2 | [`phase2/`](phase2/) | `phase-2` | Payroll outlier detector, reason codes and money at risk, FastAPI anomaly service, HR intent chatbot with slots, memory and permissions |
| 3 | [`phase3/`](phase3/) | `phase-3` | Resume ranking, fairness audit, attrition model v2, SHAP risk explanations and fair pay bands |

## Notebooks and learning workflow

Notebooks are maintained inside their respective phase folders.

**Phase 1 and Phase 2 notebooks are under** `05_Starter/notebooks/`. **Phase 3 notebooks are under** `notebooks/`.

| Phase | Notebook location | Main contents |
| --- | --- | --- |
| 1 | [`phase1/05_Starter/notebooks/`](phase1/05_Starter/notebooks/) | Data audit, feature pipeline, baseline models, model evaluation, attendance anomalies and complete classification workflow |
| 2 | [`phase2/05_Starter/notebooks/phase2/`](phase2/05_Starter/notebooks/phase2/) | Payroll audit, anomaly detection, error-type evaluation, money-at-risk explanations, API validation, chatbot NLU, FAQ, slots, memory and permissions |
| 3 | [`phase3/notebooks/`](phase3/notebooks/) | 15 notebooks covering Tasks 11–15: resume ranking, fairness, attrition v2, SHAP explanations and salary modelling |

Phase 2 also includes task-specific notebooks (`P2_T06` to `P2_T10`) and a master workflow notebook.

The `_original_starter/` folders in Phases 1 and 2 preserve starter/reference notebooks.

## Quick start

Use the relevant phase folder as the working directory.

Example for Phase 3 (replace `phase3` with `phase1` or `phase2` as needed):

```powershell
cd phase3
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Run tests

From the repository root:

```powershell
# Phase 1
cd phase1
pytest -q tests

# Phase 2
cd ..\phase2
pytest -q tests_phase2
pytest -q tests

# Phase 3
cd ..\phase3
python tests/run_all.py
```

### Open notebooks

Install Jupyter in the relevant phase environment if it is not already installed:

```powershell
jupyter lab
```

Open the notebook directory listed in the table above.

### Run Phase 2 services

Run these commands from `phase2/` in two separate terminals.

Terminal 1 – Mock HR service:

```powershell
uvicorn mock_server:app --app-dir 05_Starter/mock_services --port 9000
```

Terminal 2 – Anomaly and chatbot API:

```powershell
uvicorn app.main:app --app-dir 05_Starter/api_skeleton --port 8000
```

API documentation: `http://127.0.0.1:8000/docs`

## Phase 1 highlights

- Data audit: 17 evidence-backed findings and 6 confirmed leakage columns with proof.
- Leakage-aware feature pipeline: fits on training rows only, with cutoff-safe attendance and payroll aggregates.
- Honest attrition baselines: logistic regression CV ROC-AUC 0.709. Held-out test ROC-AUC 0.691, PR-AUC 0.355 and top-10% lift 2.75x.
- Risk bands: High band caught 19 of 48 leavers (40%).
- Attendance anomaly detector (IsolationForest + rules): precision 0.500, recall 0.840.
- Full classification workflow: 37 items covered in one notebook.

## Phase 2 highlights

- Payroll detector: rules + IsolationForest, measured on all 8 error types (overall precision 0.316, recall 0.95; rules alone precision 0.860).
- Every flag has a reason code, plain-language explanation and `money_at_risk`.
- FastAPI service with validated inputs, models loaded once at startup and versioned models.
- Chatbot: TF-IDF + Logistic Regression, 33 intents, held-out accuracy 0.8106, safe fallback and role-based answers.
- Another employee's salary is always refused.
- Tests: 69 passed in `tests_phase2` (59 service-layer + 10 API); Phase 1 regression tests are also maintained under `tests/`.

## Phase 3 highlights

- Resume ranking: precision@5 0.940, NDCG@5 0.923 and MRR 1.000 on 555 labelled pairs across 10 jobs.
- Fairness audit: 25 twin pairs and 75 comparisons. Non-zero gap in 74 of 75 before masking; zero after masking (zero by construction, other proxies untested).
- Attrition v2: CV ROC-AUC 0.716 vs 0.709 baseline. The gain is **not proven** (paired 95% interval includes 0).
- Risk report: 430 active employees scored (301 Low, 86 Medium, 43 High), with calibrated scores, three plain-English reasons and human-review notes.
- Native SHAP cross-check on all 430 employees agrees on the top reason for 68.1%.
- Pay bands: Ridge on job-related features only. Test MAE Rs 53,925, R2 0.983, band coverage 0.802.
- Tests: 72 passed, 51 of 51 requirements traced. 15 notebooks with saved outputs.

**Audit note:** The Phase 3 final audit documents that a clean-kernel full rebuild of all 15 notebooks in one pass was not performed. See [`phase3/docs/final_audit_status.md`](phase3/docs/final_audit_status.md) for the exact execution scope and limitations.

## Reports

- Phase 1 report: [`phase1/reports/`](phase1/reports/)
- Phase 2 report: [`phase2/reports/phase2/AIML_Phase2_Report_Jaisurya_Marisamy.pdf`](phase2/reports/phase2/AIML_Phase2_Report_Jaisurya_Marisamy.pdf)
- Phase 3 report: [`phase3/reports/AIML_Jaisurya_Marisamy_Phase3_Report.pdf`](phase3/reports/AIML_Jaisurya_Marisamy_Phase3_Report.pdf)

## Security

Restricted validation material (`04_Validation_Vault/`) is never committed to this repository.
