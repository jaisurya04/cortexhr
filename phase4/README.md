# CortexHR Phase 4 - Release (Days 16-20)

Forecast the future, put every model behind one live API, watch for drift and unfairness, give people a console, and hand the project over.
**All Phase 4 work lives in this folder. Phases 1, 2 and 3 are read-only inputs and were not changed** (proof: `python scripts/verify_phase1_3_unchanged.py`).

> Synthetic data. Every output is **AI-generated, for human review**: decision support, never a decision about a person.

## What was verified, and what still needs your machine

| Part | Current verification |
|---|---|
| Phase 1–3 suites | 32 + 69 + 72 tests passed; 736 Phase 1–3 files remained unchanged in the integrated audit |
| Phase 4 integrated suite | 92 passed, 0 failed, 8 skipped, 8 blocked; see root `TEST_RESULTS.md` |
| FastAPI + Pydantic | Focused API tests passed; local live HTTP smoke test passed 7/10 checks (payroll model unavailable in unpinned pandas runtime) |
| Model registry | Six model components loaded; original payroll artifact requires pinned pandas 3.0.2 and is unavailable under sandbox pandas 2.2.3 |
| Monitoring | PSI, twin-resume and pay-gap checks are covered by passing service tests; restricted-vault checks remain not checked without the vault |
| Resume ingestion | Focused DOCX upload, PII-redaction and memory-only retention test passed; PDF upload path uses pypdf and still needs a real-world document check |
| Workforce map | Data-backed city aggregation and dashboard HTML generation test passed; full browser interaction remains unverified |
| Streamlit console | Not launched in this sandbox because Streamlit is not installed |
| Browser suite | Desktop/mobile, chart, popup and menu tests skipped because Chromium could not be launched |
| Forecast release output | Saved output remains baseline-only; rerun the forecast pipeline in the pinned environment before treating SARIMAX as released |
| Public deployment, phone test, scheduled run, demo video, `v1.0.0` tag | Not done; these require your hosting accounts, device and release sign-off |

The sandbox uses Python 3.13.5 and several library versions that differ from the pinned Python 3.12 release stack. `scripts/check_environment.py` fails closed when the exact versions are not installed. The integrated runner exits 2 while blocked/skipped checks and user-owned release actions remain; no production-ready claim is made.

## Quick start (from the repository root, with the Phase 1-3 virtual environment)

```powershell
pip install -r phase4/requirements.txt          # pinned: statsmodels, fastapi, pydantic, uvicorn, streamlit + the Phase 2/3 pins
Copy-Item phase4/.env.example .env             # Windows PowerShell; use cp phase4/.env.example .env on macOS/Linux
python phase4/scripts/create_user.py           # create password hash + random signing secret; set them in .env for production auth
python phase4/scripts/check_environment.py      # are the pins installed and does SARIMAX fit?
python phase4/scripts/run_forecast.py           # Day 16: backtests + 6-month forecast -> phase4/outputs/forecast/
python phase4/scripts/monitor.py                # Day 18: drift + fairness report, exit code 0/1/2 = OK/WATCH/ALERT
$env:PYTHONPATH="phase4/src"; uvicorn cortexhr_phase4.api.app:app --port 8000     # Day 17: the API (docs at /docs)
python phase4/scripts/smoke_test.py --base-url http://127.0.0.1:8000             # /v1/health + one real call per model
cd phase4; $env:CORTEXHR_API_URL="http://127.0.0.1:8000"; streamlit run src/cortexhr_phase4/console/app.py   # Day 19: console
```
Or run `.\phase4\run_local.ps1` (Windows) / `sh phase4/run_local.sh`. Tests: `python phase4/tests/run_all.py` (or `pytest phase4/tests`). For the browser suite, install `pip install -r phase4/requirements-dev.txt` and run `python -m playwright install chromium` once on the test machine.

## Folder map

| Path | What it holds |
|---|---|
| `src/cortexhr_phase4/forecasting/` | data checks, workforce events, baselines, statsmodels model, intervals, backtests, pipeline, EDA |
| `src/cortexhr_phase4/monitoring/` | PSI drift, twin-resume and pay-gap re-checks, OK/WATCH/ALERT rules, monitor, audit history |
| `src/cortexhr_phase4/services/` | `Registry` (loads every model once), request handlers, errors, sample inputs |
| `src/cortexhr_phase4/api/` | the `/v1` FastAPI app and the Pydantic contracts |
| `src/cortexhr_phase4/console/` | dashboard canvas (reference design), charts, API client, Streamlit app, theme |
| `scripts/` | `run_forecast.py`, `monitor.py`, `smoke_test.py`, `check_environment.py`, `build_docs.py`, `verify_phase1_3_unchanged.py`, `render_dashboard_preview.py` |
| `tests/` | all tests (`run_all.py` prints PASS / FAIL / SKIP honestly) |
| `docs/` | requirements traceability, model cards (7), architecture, API reference, hosting comparison, monitoring policy, deployment, limitations, release notes inputs |
| `outputs/` | generated forecast files, monitoring reports and history, dashboard screenshots |
| `deploy/` | Dockerfile, Render blueprint, GitHub Actions monitor workflow (copy instructions inside) |
| `assets/` | the CortexHR logo and illustrations (cropped from the approved reference screenshot, see "Logo" below) |

## Endpoints (all under `/v1`; full reference in `docs/api_reference.md`)

| Method | Path | Purpose |
|---|---|---|
| GET | `/v1/health` | service + per-model status (open, no key needed) |
| POST | `/v1/auth/login`, `/v1/auth/logout` | password sign-in and token sign-out |
| GET | `/v1/models` | every model: status (`loaded` / `fallback` / `unavailable`), version, source, detail |
| POST | `/v1/attrition/predict` | 12-month leave risk, band and three reasons for one employee |
| GET | `/v1/attrition/employees` | risk list with band / department filters |
| POST | `/v1/anomaly/payroll` and `/batch` | payslip anomaly flags |
| POST | `/v1/anomaly/attendance` and `/batch` | attendance anomaly flags |
| POST | `/v1/chat` | HR assistant |
| POST | `/v1/resume/rank` | rank saved resumes for a job (protected fields are refused) |
| POST | `/v1/resume/upload` | in-memory PDF/DOCX ingestion and temporary ranking |
| POST | `/v1/payband/recommend` | low / mid / high pay band |
| GET | `/v1/forecast` | six-month forecast with 80% interval (`metric`, `department_id`) |
| POST | `/v1/forecast/scenario` | what-if workforce event (needs the improved model) |
| GET | `/v1/monitoring` | drift + fairness status OK / WATCH / ALERT (`?refresh=true` re-runs it) |
| GET | `/v1/overview`, `/v1/performance`, `/v1/payband/options`, `/v1/resume/jobs` | data for the console |

Errors are always `{"error": "<code>", "details": ["..."]}` with a meaningful HTTP status (401, 403, 404, 413, 415, 422, 503, 500).

## Authentication and roles

The console supports password sign-in through `POST /v1/auth/login`. Passwords are stored as PBKDF2-SHA256 hashes (never plaintext); the signing secret must contain at least 32 bytes; successful login returns an 8-hour HMAC-signed bearer token. Sign-out revokes the token for the lifetime of the current API process; restarting the API clears its in-memory revocation list. The API validates the token and enforces role permissions on every protected `/v1/*` route. Roles: `admin`, `hr_manager`, `recruiter`, `analyst`. Admin and HR Manager can use the core HR modules; Recruiter is restricted to overview, assistant, resume and pay-band workflows; Analyst can use workforce analytics, anomalies, forecasts, models and monitoring but not resume ranking. The exact matrix is enforced by the API route policy.

1. Generate a user record: `python phase4/scripts/create_user.py`.
2. Add the generated JSON record to `CORTEXHR_AUTH_USERS` and the generated signing secret to `CORTEXHR_AUTH_SECRET` in your private environment configuration.
3. Set `CORTEXHR_ENV=production`, configure a restrictive `CORTEXHR_ALLOWED_ORIGINS`, and set a shared `CORTEXHR_API_KEY` if your deployment requires it.
4. Never commit `.env`, passwords, password hashes, signing secrets, API keys, or restricted fairness-vault data.

When authentication is not configured, local development mode explicitly shows a demo-only sign-in and grants a local admin demo session. Production mode refuses to serve protected endpoints until auth is configured. Production deployments should also enforce HTTPS, reverse-proxy rate limiting, secret rotation, backups, and centralized audit-log retention; this project does not provide a managed identity provider or distributed rate limiter.

## Configuration (environment variables; nothing secret is in the repository)

Copy `.env.example` to the repository-root `.env`; `python-dotenv` loads it without overriding explicitly exported variables. Key ones: `CORTEXHR_API_URL` (console and smoke test), `CORTEXHR_API_KEY` (optional shared API key), `CORTEXHR_AUTH_SECRET` and `CORTEXHR_AUTH_USERS` (password login and role authorization), `CORTEXHR_VAULT_DIR` (restricted files, enables the gender pay-gap check), `CORTEXHR_MOCK_SERVICE_URL` (Phase 2 mock HR service for leave/payslip chatbot intents).

## Authentication endpoints

- `POST /v1/auth/login` — username/password exchange for a signed bearer session.
- `POST /v1/auth/logout` — revokes the current token for the lifetime of the API process.

## Resume document ingestion

The upload API route is `POST /v1/resume/upload` (multipart fields: `file`, `job_id`, `top_k`).

The Resume Screener accepts text-based PDF and DOCX files up to 8 MB (PDF up to 20 pages). Files are parsed in memory; uploaded content is not written to disk. The uploaded candidate is added to an in-memory ranking corpus and disappears when the API restarts. Common email, phone, city, college-tier, explicitly labelled demographic lines and a confidently detected first-line name are redacted before scoring; document parsing is not an antivirus scan, so production deployments must add malware scanning at the upload gateway. The engine refits the TF-IDF corpus for this upload, so its score must **not** be compared directly with the saved offline benchmark metrics. Scanned/image-only PDFs need OCR and are rejected when no usable text can be extracted. Human review remains mandatory; no automatic rejection is performed.

## Workforce location map

The dashboard map is an interactive city-level plot built from `work_location` counts in the supplied active-employee scoring dataset. Markers use approximate city-centre coordinates; they are not office addresses or employee-level tracking. The panel reports active employees by location, not hiring volume.

## Model loading and provenance
Models are loaded **once at start-up** (`services/registry.py`). Phase 2 files are checked against the SHA-256 in `phase2/models/model_registry.json`. The Phase 3 attrition model is loaded live and compared with the saved Phase 3 scores; if it cannot be loaded (for example a different scikit-learn version) the API serves the saved scores of the same model and says so in `/v1/models`.

## Logo and illustrations
No logo file exists in the repository, so `assets/cortexhr_logo_sidebar.png`, `cortexhr_logo_mark.png` and `brain_promo.png` were **cropped from the approved reference screenshot** (no new logo was invented). They are low resolution. To use your original logo, replace those files (same names) with higher-resolution versions. The smiling sun with its rising, rotating-rays and drifting-cloud animation is vector artwork drawn in the same style.

## Troubleshooting
* `Cannot reach the CortexHR API` in the console: start the API, check `CORTEXHR_API_URL`, then `python phase4/scripts/smoke_test.py`.
* `/v1/models` shows `fallback` for attrition: the pinned scikit-learn 1.8.0 is not installed; install `phase4/requirements.txt`.
* Forecast says `improved model NOT RUN`: `pip install statsmodels==0.14.6`, then re-run `scripts/run_forecast.py`.
* 401 from the API: check the shared `CORTEXHR_API_KEY` and sign in again if the bearer session expired or was revoked.
* Render free instance out of memory: switch to the $7 Starter instance (see `docs/hosting_comparison.md`).
* More: `docs/deployment.md`, `docs/known_limitations.md`.
