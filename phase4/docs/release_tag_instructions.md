# v1.0.0 tag - NOT created. Prepared only.

Do not tag until every item below is true and its evidence is saved.

| # | Condition | Evidence to keep |
|---|---|---|
| 1 | `CORTEXHR_PRISTINE_REPO=<untouched copy> python tests/run_all.py` on your machine: 0 failed, 0 skipped, 0 blocked, Phase 1-3 unchanged | the printed summary |
| 2 | `python scripts/run_forecast.py` run with statsmodels installed; `improved_model_status` is not "NOT RUN" | `outputs/forecast/forecast_run.json` |
| 3 | Real FastAPI served with uvicorn; `python scripts/smoke_test.py` green locally | smoke output |
| 4 | Public API URL live; smoke test green from the live URL and from your phone | output + photo/screenshot |
| 5 | Monitoring workflow enabled and one real scheduled run succeeded; `GET /v1/monitoring` on the live URL returns it | screenshot of the run + endpoint response |
| 6 | Console live, five tabs checked on desktop and phone, 3 tab screenshots, stranger test done | screenshots + note |
| 7 | Model cards match the live `/v1/performance` (`tests/test_model_cards.py` green) | test output |
| 8 | Demo recorded, link opened in a private window | link |
| 9 | No secret in the repo | `test_secrets_are_not_in_the_repository` green |

Commands (run from the repository root on a clean `main`):
```
git status                      # must be clean
git tag -a v1.0.0 -m "CortexHR v1.0.0"
git push origin v1.0.0
```
Then put `https://github.com/<you>/cortexhr/releases/tag/v1.0.0` in the report. Status today: **USER ACTION REQUIRED**.
