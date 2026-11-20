# MY ACTION REQUIRED - everything only you can do

| # | Action | Why it cannot be done for you | Where the steps are |
|---|---|---|---|
| 1 | `pip install -r requirements-dev.txt -r requirements.txt`; run the full test runner so FastAPI, statsmodels and Streamlit tests actually execute | build sandbox had no package network | README |
| 2 | `python scripts/run_forecast.py` (SARIMAX), then `python scripts/monitor.py`, then re-run `tests/run_all.py`; read the new coverage honestly | statsmodels missing here | docs/forecasting_report.md |
| 3 | Start the real API (`uvicorn cortexhr_phase4.api.app:app`) and Streamlit console; look at all five tabs on desktop and phone | not installable here | docs/console_guide.md |
| 4 | Create a Render account, deploy the API, set env vars, run the smoke test on the live URL from your phone | your account | docs/deployment.md |
| 5 | Deploy the console (Streamlit Cloud), set `CORTEXHR_API_URL` | your account | docs/deployment.md |
| 6 | Copy `deploy/github-actions-monitor.yml` to `.github/workflows/`, enable Actions, wait for one successful scheduled run, screenshot it | outside phase4/, needs your repo | docs/monitoring_policy.md |
| 7 | Restricted vault: run the gender pay-gap check with `CORTEXHR_VAULT_DIR` | restricted data | docs/monitoring_policy.md |
| 8 | Stranger usability test of the console; write what happened | needs a person | docs/console_guide.md |
| 9 | Take 3 tab screenshots from the live console | live URL | - |
| 10 | Record the demo | you | docs/demo_recording_checklist.md |
| 11 | Final Challenge + the "model I would not release" sentence in your own words | judgement | docs/final_challenge_workflow.md |
| 12 | Tag v1.0.0 only when the conditions pass | release decision | docs/release_tag_instructions.md |
