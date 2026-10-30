# CortexHR · Phase 2: Core Development

Payroll anomaly detection, reason codes with money at risk, a validated anomaly API and an HR chatbot with memory and role-based permissions.
Built for the Aura Workforce by Growmar AI/ML track (Tasks 6 to 10). All companies, people and data are fictional.

Phase 2 builds on [`../phase1`](../phase1) (audit, feature pipeline, baselines, attendance anomalies). The Phase 1 code it reuses is included here, so this folder runs on its own.

## What was built

| Task | Deliverable | Where |
|---|---|---|
| 6 Payroll outlier detector | Ratio features, readable rules (duplicate, negative/zero net, gross vs expected, net arithmetic, PF missing) plus IsolationForest; precision/recall for each of the 8 error types | `src/cortexhr/phase2/payroll.py`, `reports/phase2/payroll_cx06_per_type.csv` |
| 7 Reason codes and money at risk | One reason code and plain sentence per flag, `money_at_risk`, official 12-column output, ranked by money at risk | `reports/phase2/payroll_anomaly_explanations.csv`, `payroll_review_sample_20.csv` |
| 8 Anomaly API | FastAPI + Pydantic, models loaded once at start-up, validated input (422), versioned models, 10 API tests | `05_Starter/api_skeleton/app/main.py`, `models/`, `tests_phase2/test_api.py` |
| 9 Intent chatbot | TF-IDF + Logistic Regression on 858 messages / 33 intents, FAQ answers, confidence fallback | `src/cortexhr/phase2/chatbot.py`, `reports/phase2/chatbot_*.csv` |
| 10 Slots, memory, permissions | Month and employee slots, follow-ups, mock HR service via `X-Demo-User`, `salary_of_other` always refused | `reports/phase2/chat_transcripts.md`, `chatbot_dialogue_results.csv` |

## Results

| Item | Value |
|---|---|
| Payroll rows scored / flagged | 7,355 / 301 (107 by rules, 194 by model only) |
| Payroll detection vs injected errors (CX-06) | precision 0.316, recall 0.95 (rules alone: 0.860 / 0.92) |
| Hardest error type | OVERTIME_SPIKE (recall 0.73) |
| Chatbot held-out accuracy | 0.8106 (107 of 132) |
| Official dialogues | 12 scenarios, 28 turns, all intents correct |
| Tests | 69 in `tests_phase2` (59 service-layer + 10 API); 32 Phase 1 tests in `tests` |

Labels were used only to measure the detectors after the predictions were frozen, never to train them.
The CX-06 label file is restricted and is not part of this repository.

## Run it

Python 3.11 or newer is required. Run everything from this folder (`phase2/`).

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install -r 05_Starter/requirements.txt     # FastAPI, httpx, uvicorn (API tests and service)
pytest -q tests_phase2                         # 69 tests
pytest -q tests                                # Phase 1 tests, 32
```

Start the services (two terminals):

```powershell
# terminal 1: mock HR service
uvicorn mock_server:app --app-dir 05_Starter/mock_services --port 9000
# terminal 2: anomaly + chat API
uvicorn app.main:app --app-dir 05_Starter/api_skeleton --port 8000
```

Interactive docs: `http://127.0.0.1:8000/docs`

Models are stored in `models/` with `model_registry.json` (version, rows, library version, SHA-256). If a model file is missing, the API retrains it; `python scripts/train_phase2_models.py` rebuilds them.

## API examples (PowerShell)

```powershell
curl.exe -s http://127.0.0.1:8000/v1/health

# bad input returns 422
'{"employee_id":"E1","scheduled_days":20,"present_days":25,"avg_hours_worked":8}' | Set-Content bad.json -Encoding ascii
curl.exe -s -o NUL -w "HTTP %{http_code}`n" -X POST http://127.0.0.1:8000/v1/anomaly/attendance -H "content-type: application/json" -d "@bad.json"

# another employee's salary is refused (no service call)
'{"message":"what is Priya Nair salary","session_id":"s1","demo_user":"EMP-0002"}' | Set-Content chat.json -Encoding ascii
curl.exe -s -X POST http://127.0.0.1:8000/v1/chat -H "content-type: application/json" -d "@chat.json"
```

Endpoints: `GET /v1/health`, `GET /v1/models`, `POST /v1/anomaly/payroll`, `POST /v1/anomaly/payroll/batch` (1 to 500 rows, duplicates inside the batch detected), `POST /v1/anomaly/attendance`, `POST /v1/anomaly/attendance/batch`, `POST /v1/chat`.

Example payroll response: `flagged: "Y"`, `reason_code: "NEGATIVE_NET_PAY|DEDUCTION_TOTAL_MISMATCH|DEDUCTIONS_EXCEED_GROSS"`, a plain sentence in `reason_text`, `money_at_risk: 47500.0`, `model_version: "payroll-iforest-rules-v2"`.

## Design notes

- **Rules first, model second:** readable rules cover hard facts (duplicates, negative pay); IsolationForest adds review candidates the rules do not cover.
- **Single-row scoring:** scores are scaled with min/max learned on training data and saved with the detector, so one row scores the same as in a batch.
- **money_at_risk:** gap against expected gross (fixed pay x payable days + overtime + variable), duplicate net pay, PF owed, overtime over the cap. It is a ranking estimate, not an audited loss.
- **Role security:** the role comes from the demo identity directory; a `user_role` sent by the client is ignored. Chit-chat never triggers a data call.
- **Chatbot:** FAQ answers use TF-IDF retrieval over `hr_faq` with similarity and margin thresholds; fallback threshold 0.30 was chosen on training data only (`reports/phase2/chatbot_threshold_sweep_cv.csv`).

## Limits

- Precision is 0.316: 206 flagged payslips are not injected errors (many are high earners the model finds unusual). Every flag needs human review before any action.
- The duplicate check only knows payslips from the training file or the same batch.
- Salary band, attrition, headcount forecast and resume scoring answer "not connected in this version yet" (Phase 3).
- The mock HR service returns only the signed-in user's own data; "last month" is anchored to 2026-09 because the mock data ends there.

## Notebooks and reports

- Task notebooks: `05_Starter/notebooks/phase2/P2_T06` to `P2_T10`; supporting notebooks 07 to 16 and `P2_MASTER_37_chatbot_classification_workflow.ipynb`.
- Phase 2 report: `reports/phase2/AIML_Phase2_Report_Jaisurya_Marisamy.pdf`
- Scripts to regenerate reports and models are in `scripts/`; documentation in `docs/`.
