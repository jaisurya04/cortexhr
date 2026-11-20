# CortexHR v1.0.0 - Release notes (draft; the tag is created only after the checks in docs/deployment.md pass)

## Included
* **Forecasting (Day 16):** six-month headcount (company + 10 departments), monthly exits and attrition rate with 80% intervals; seasonal baselines; improved statsmodels model with workforce events; last-6-months hold-out and rolling-origin backtests; scenario (what-if) events; plain-language explanation; outputs in the `sample_forecast` layout.
* **Unified API (Day 17):** versioned `/v1` FastAPI service for attrition, payroll and attendance anomalies, HR chatbot, resume ranking, pay-band advice, forecast and monitoring; Pydantic contracts; models loaded once; consistent JSON errors; optional API key; structured logs; `smoke_test.py`; Docker and Render configuration.
* **Monitoring (Day 18):** PSI drift on the 8 most important attrition features, twin-resume gap and career-gap pay-gap re-checks, OK / WATCH / ALERT, scheduled `monitor.py`, `GET /v1/monitoring`, audit history.
* **Insights Console (Day 19):** Streamlit console in the approved CortexHR design (welcome popup, KPI cards, charts, search, alerts), Data Foundation evidence page, model cards and saved employee-level SHAP explanations. Includes role-aware navigation.
* **Security and ingestion update:** PBKDF2-SHA256 password hashes, HMAC-signed 8-hour bearer sessions, API-level Admin / HR Manager / Recruiter / Analyst authorization, token sign-out revocation, and in-memory PDF/DOCX resume ingestion with PII redaction and file limits.
* **Workforce map:** the decorative reference image is replaced with interactive city-level markers using active employee work-location counts and approximate city-centre coordinates.
* **Handover (Day 20):** seven model cards, architecture picture, traceability matrix, final evaluation, limitations, deployment guide, demo script.

## Not included
* Automatic employment decisions of any kind. Retraining of Phase 1-3 models. Enterprise SSO / identity-provider integration, durable session revocation across API restarts, built-in distributed rate limiting, antivirus scanning, durable resume storage, and external HRIS/payroll/time-clock integration. Department-level exit forecasts. The restricted-vault gender pay-gap check unless you provide the vault locally.

## Known limits (details in `docs/known_limitations.md`)
* Attrition v2 is not proven better than the baseline; 2026 inputs show a PSI ALERT for `months_since_last_raise`.
* Forecast ranges are optimistic (baseline 80% interval covered about 68% in rolling tests); one hiring freeze is the only evidence for that effect.
* The FastAPI layer started and 7/10 live HTTP smoke checks passed in the current sandbox; payroll is unavailable under the sandbox pandas 2.2.3 and requires the pinned pandas 3.0.2 environment. Streamlit is not installed here and the browser suite could not launch Chromium, so the console still needs runtime and desktop/mobile verification. The saved forecast outputs are baseline-only until the pinned forecast pipeline is rerun.
* Public deployment, phone test, scheduled run and demo recording are your actions.

## Compatibility
Python 3.12; pinned libraries in `phase4/requirements.txt` (pandas 3.0.2, numpy 2.4.4, scikit-learn 1.8.0, statsmodels 0.14.6, fastapi 0.141.1, pydantic 2.13.5, uvicorn 0.52.4, streamlit 1.66.0). Phase 1-3 are unchanged.
