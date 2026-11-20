# Deployment guide (Day 17 steps 5-6, Day 19 steps 5-6)

None of the steps below has been performed for you; they need your accounts and devices. Do them in this order and keep the evidence.

## 0. Before deploying
1. `pip install -r phase4/requirements.txt` then `python phase4/scripts/check_environment.py` (every line must say `ok`).
2. `python phase4/scripts/run_forecast.py` (this runs the statsmodels model; it replaces the baseline-only saved outputs) and `python phase4/scripts/monitor.py`.
3. `python phase4/tests/run_all.py`: nothing may FAIL; read every SKIP line; with the full requirements installed there should be none for libraries.
4. Local end-to-end: start the API, run `python phase4/scripts/smoke_test.py --base-url http://127.0.0.1:8000` (all 10 checks must pass; if payroll reports model_unavailable, verify the exact pinned pandas 3.0.2 runtime), start the console and click every page.
5. Commit and push the Phase 4 files (and the regenerated `phase4/outputs/forecast/`).

## 1. API on Render (free to start)
1. render.com > New > Web Service > connect `jaisurya04/cortexhr`.
2. Root directory: repository root. Build: `pip install -r phase4/requirements.txt`. Start: `uvicorn cortexhr_phase4.api.app:app --host 0.0.0.0 --port $PORT`. Health check path `/v1/health`. (Or copy `deploy/render.yaml` to the repository root.)
3. Environment: `PYTHONPATH=phase4/src`, `PYTHON_VERSION=3.12.3`, `CORTEXHR_ENV=production`, and optionally `CORTEXHR_API_KEY` (a long random value). **Set every setting as an environment variable in the dashboard; no secret goes in the repository.**
4. If the log shows an out-of-memory kill, change the instance to Starter ($7).
5. Open `https://<service>.onrender.com/v1/health` and `/docs`.

## 2. Smoke test against the live address, from your phone (step 6)
* Easiest: open `https://<service>.onrender.com/docs` on the phone, run `GET /v1/health` and `POST /v1/attrition/predict` with `{"employee_id": "EMP-0002"}`.
* Full: on Android install Termux (`pkg install python`), then `python smoke_test.py --base-url https://<service>.onrender.com --api-key <key>` (the script uses only the standard library; copy the file or the repository).
* The first call after 15 idle minutes can take about a minute (free instance waking up). Screenshot the output as evidence.

## 3. Monitoring schedule (Day 18 step 5)
Copy `phase4/deploy/github-actions-monitor.yml` to `.github/workflows/cortexhr-monitor.yml` (a new file outside `phase4/`; this is the only step that touches a path outside this folder, so it needs your approval), push, then Actions > CortexHR monitor > Run workflow. Screenshot the finished run. Then redeploy the API so `GET /v1/monitoring` is live (step 6).

## 4. Console on Streamlit Community Cloud (Day 19 step 5)
1. share.streamlit.io > New app > repository `jaisurya04/cortexhr`, branch `main`.
2. Main file: `phase4/src/cortexhr_phase4/console/app.py`. Dependencies are read from `phase4/src/cortexhr_phase4/console/requirements.txt`.
3. Advanced settings > Secrets: `CORTEXHR_API_URL = "https://<service>.onrender.com"` (and `CORTEXHR_API_KEY` if you set one on the API).
4. Deploy. Open it on a phone and ask someone who has never seen it to find an employee at high risk and ask the assistant one question without your help (step 6). Write down where they got stuck.
5. The Streamlit config file lives in `phase4/.streamlit/config.toml`; Community Cloud may read it only from the repository root. If the colours look default, add the same `[theme]` block under *App settings > Theme*, or copy the file to `.streamlit/config.toml` at the root (a new file outside `phase4/`).

## 5. Release (Day 20)
Only after every check above is green and the deployed API and console work: write the Final Challenge notes (`docs/final_challenge_and_judgement.md`), record the 5-minute demo (`docs/demo_script_5min.md`), then
```
git tag v1.0.0
git push origin v1.0.0
```
Do **not** tag earlier: the tag should point at a commit whose tests you have run on your machine.
