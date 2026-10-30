# CortexHR

AI/ML project by **Jaisurya Marisamy** (Growmar, AI/ML track), built one phase per week.
All companies, people and datasets in this programme (Aura Workforce, Growmar, Zenith Nova) are fictional.

## Repository layout

```
cortexhr/
├── README.md        <- this file
├── .gitignore
├── phase1/          <- Phase 1: Data foundations and baseline models   (tag: phase-1)
└── phase2/          <- Phase 2: Core development                       (tag: phase-2)
```

Each phase folder is **self-contained**: it has its own `src/`, `tests/`, `reports/`, `requirements.txt` and README,
and runs on its own. Phase 2 builds on Phase 1, so `phase2/` also contains the Phase 1 code it reuses.
Paths in each phase report are relative to that phase's folder.

| Phase | Folder | Git tag | What it delivers |
|---|---|---|---|
| 1 | [`phase1/`](phase1/) | `phase-1` | Data audit and leakage checks, feature pipeline, honest attrition baselines, risk bands, unsupervised attendance anomaly detection |
| 2 | [`phase2/`](phase2/) | `phase-2` | Payroll outlier detector (per-error-type results), reason codes and money at risk, FastAPI anomaly service, HR intent chatbot with slots, memory and permissions |

## Quick start (any phase)

```powershell
cd phase2                     # or phase1
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pytest -q tests_phase2        # Phase 2 tests (use `pytest -q tests` for Phase 1)
```

Run the Phase 2 services (two terminals, from `phase2/`):

```powershell
uvicorn mock_server:app --app-dir 05_Starter/mock_services --port 9000
uvicorn app.main:app --app-dir 05_Starter/api_skeleton --port 8000
```

## Phase 2 highlights

- Payroll detector: rules + IsolationForest, measured on all 8 error types (overall precision 0.316, recall 0.95; rules alone precision 0.860).
- Every flag has a reason code, a plain-language sentence and `money_at_risk`, in the official output format.
- FastAPI service with validated inputs (422 on bad input), models loaded once at start-up, versioned models.
- Chatbot: TF-IDF + Logistic Regression, 33 intents, held-out accuracy 0.8106, safe fallback, role-based answers; another employee's salary is always refused.
- Tests: 69 passed in `tests_phase2` (59 service-layer + 10 API).

## Reports

- Phase 1 report: see `phase1/reports/`
- Phase 2 report: `phase2/reports/phase2/AIML_Phase2_Report_Jaisurya_Marisamy.pdf`

Restricted validation material (`04_Validation_Vault/`) is never committed to this repository.
