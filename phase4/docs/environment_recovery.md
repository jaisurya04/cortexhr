# Recover the pinned local development environment

This is the supported recovery path for the current release pins. Run from the repository root in Windows PowerShell. Do not install into the global Python environment.

## 1. Install and select Python 3.12

Install a 64-bit Python 3.12 release, then open a new PowerShell window and verify:

```powershell
py -0p
py -3.12 --version
```

The second command must report Python 3.12.x. If `py -3.12` is not found, finish the Python installation or repair the Windows Python Launcher before continuing.

## 2. Create a clean virtual environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install --upgrade pip
python -m pip install -r phase4\requirements-dev.txt
```

`requirements-dev.txt` includes the exact runtime pins plus pytest, HTTPX, Flask and Playwright. Do not use `pip install -U pandas` or selectively upgrade model dependencies: that can invalidate the saved model artifact compatibility.

## 3. Verify the environment before loading models

```powershell
python phase4\scripts\check_environment.py
```

Every pinned dependency should show `ok`, and the SARIMAX smoke fit should pass. Stop here if any line says `DIFF`, `MISSING` or `FAIL`; capture the full output rather than changing the model artifact.

## 4. Install the browser used by the UI tests

```powershell
python -m playwright install chromium
```

If Windows reports missing browser system dependencies, install the Microsoft Visual C++ runtime and rerun the command. Then verify:

```powershell
python -c "from playwright.sync_api import sync_playwright; p=sync_playwright().start(); b=p.chromium.launch(); print(b.version); b.close(); p.stop()"
```

## 5. Configure local credentials (optional for development; required in production)

```powershell
Copy-Item phase4\.env.example .env
python phase4\scripts\create_user.py
```

Copy the generated JSON user record into `CORTEXHR_AUTH_USERS` in the root `.env`, and the generated signing secret into `CORTEXHR_AUTH_SECRET`. Keep `.env` private and out of Git. For a local smoke test using password auth, set the credentials only in the current shell:

```powershell
$env:CORTEXHR_SMOKE_USERNAME = "your-login"
$env:CORTEXHR_SMOKE_PASSWORD = "your-password"
```

If no auth users are configured and `CORTEXHR_ENV=development`, the API's local demo principal is used. Do not use demo mode in production.

## 6. Rebuild generated forecast outputs and run all tests

```powershell
python phase4\scripts\run_forecast.py
python phase4\scripts\monitor.py
python phase4\tests\run_all.py
```

Read the final summary. A test marked SKIP or BLOCKED is not a pass. Investigate every FAIL and retain the output as release evidence.

## 7. Start API and run all ten live checks

Open PowerShell window A:

```powershell
$env:PYTHONPATH = "$PWD\phase4\src"
uvicorn cortexhr_phase4.api.app:app --host 127.0.0.1 --port 8000
```

Open PowerShell window B, activate the same `.venv`, then:

```powershell
# If using password authentication:
$env:CORTEXHR_SMOKE_USERNAME = "your-login"
$env:CORTEXHR_SMOKE_PASSWORD = "your-password"
python phase4\scripts\smoke_test.py --base-url http://127.0.0.1:8000

# Or use an already issued bearer token:
python phase4\scripts\smoke_test.py --base-url http://127.0.0.1:8000 --token $env:CORTEXHR_SMOKE_TOKEN
```

The smoke test now supports the same Bearer-token login used by the console, as well as the optional `X-API-Key`. When auth is configured, use a user whose role is allowed to call every smoke-tested endpoint (Admin or HR Manager). A Recruiter or Analyst will correctly receive 403 for some endpoints and cannot be used for the full all-endpoint smoke test.

Expected result: 10/10 checks. In particular, `/v1/models` should report the payroll model as loaded, and the payroll anomaly request should return one result. If it still reports unavailable, stop and capture the exact `/v1/models` response and API startup log; do not regenerate or replace the original payroll model artifact.

## 8. Launch the Streamlit console

In a third PowerShell window (or after stopping the API window, if running API separately):

```powershell
Set-Location phase4
$env:CORTEXHR_API_URL = "http://127.0.0.1:8000"
streamlit run src\cortexhr_phase4\console\app.py
```

Sign in with the configured user. Verify every page and the role-specific navigation.

## 9. Run the desktop/mobile browser suite

From the repository root, with API and Streamlit running where required by the test setup:

```powershell
python -m pytest phase4\tests\test_dashboard.py phase4\tests\test_ai_label.py -q
```

The browser tests render the dashboard in Chromium at desktop and mobile viewports. Also manually test the real Streamlit app at desktop width and a phone-sized viewport. Record screenshots and browser console/network errors.

## Release rule

Do not deploy until the environment checker is clean, all relevant tests pass with no unexplained skips/blocks, the API smoke test is 10/10, Streamlit works, and desktop/mobile browser checks are complete. The original Phase 1–3 and payroll model artifacts must remain unchanged unless a separately reviewed model-retraining task explicitly authorizes replacement.
