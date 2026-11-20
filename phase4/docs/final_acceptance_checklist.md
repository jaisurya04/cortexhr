# Final acceptance checklist (brief section 12 and Day 20)

Legend: [x] done and evidenced here, [~] written but needs your machine to run, [ ] your action.

## Requirements and protection
- [x] Every requirement of the five task images is accounted for: `docs/requirements_traceability.md` (66 rows, unclear text listed)
- [x] Phase 1-3 unchanged: byte-identical to the received ZIP (`scripts/verify_phase1_3_unchanged.py --pristine <copy>`); `tests/run_all.py` hashes 726 files before and after every run. **On your repository run:** `python phase4/scripts/verify_phase1_3_unchanged.py` (compares with the `phase-3` tag)
- [x] All new work is inside `phase4/`; two optional files outside it (`.github/workflows/...`, `render.yaml`) are **not created** and need your approval

## Day 16 - forecasting
- [x] Six-month headcount and exits forecast with 80% intervals, sample_forecast columns, reproducible (hash test)
- [x] Baseline, hold-out backtest, rolling backtests, events, explanation, EDA
- [~] Improved statsmodels model: run `python phase4/scripts/run_forecast.py` and read `outputs/forecast/model_comparison.csv`

## Day 17 - API
- [x] Handlers for every endpoint run against the real saved models; `smoke_test.py` 10/10 over HTTP against a stand-in; load test and its fix
- [~] FastAPI + Pydantic layer: `pip install -r phase4/requirements-dev.txt && python phase4/tests/run_all.py -k fastapi`
- [ ] Deploy (Render), set environment variables, run the smoke test from your phone, keep the live link

## Day 18 - monitoring
- [x] PSI on 8 features, OK/WATCH/ALERT, twin-resume and career-gap pay-gap re-checks, history, 4-step plan
- [ ] Schedule it (GitHub Actions workflow in `deploy/`), screenshot the run; redeploy with `/v1/monitoring`

## Day 19 - console
- [x] Dashboard canvas in the reference design, popup and animation, charts, search, menus; browser-tested at three sizes
- [~] Streamlit shell: `cd phase4 && streamlit run src/cortexhr_phase4/console/app.py`, click every sidebar page, compare with `outputs/console/` screenshots
- [ ] Deploy the console, test on a phone with a stranger, screenshot three tabs

## Day 20 - handover
- [x] Seven model cards, architecture picture, final evaluation, release notes draft, demo script, limitations
- [ ] Re-read every number after your forecast re-run (`python phase4/scripts/build_docs.py` regenerates the cards and reports)
- [ ] Record the demo, finish the Phase 4 report, write the Final Challenge and your "which model would I not release" answer
- [ ] Only then: `git tag v1.0.0` and push the tag

## Disclosures that must stay in the report
Attrition v2 not proven better than baseline; PSI ALERT on `months_since_last_raise`; forecast intervals optimistic; gender pay gap not checked without the vault; zero twin gap is by construction; FastAPI/statsmodels/Streamlit were not run in the build sandbox.
