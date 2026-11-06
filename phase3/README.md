# CortexHR · Phase 3: Intelligence

Resume ranking, a fairness audit of the screener, attrition model v2, a risk report with reasons for every active employee, and fair pay bands.
Built for the Aura Workforce by Growmar AI/ML track (Tasks 11 to 15). All companies, people and data are fictional.

Phase 3 reuses the Phase 1 pipeline and folds (imported, not copied). It is self-contained: this folder runs on its own.

## What was built

| Task | Deliverable | Where |
|---|---|---|
| 11 Resume ranking | TF-IDF + skill overlap + capped experience, identifiers masked; top-5 per job with matched and missing skills; quality against human labels; misranking analysis; API contract | `src/cortexhr/phase3/resume/`, `reports/task11/`, `docs/api_notes_resume_ranking.md` |
| 12 Fairness audit | 25 twin pairs, gap per pair, before and after masking, ablation, fairness statement | `src/cortexhr/phase3/fairness/`, `reports/task12/`, `docs/fairness/fairness_statement.md` |
| 13 Attrition v2 | Gradient-boosting candidates on the Task 3 folds, tuning log, calibration, honest verdict, saved `attrition-v2` | `src/cortexhr/phase3/attrition/`, `reports/task13/`, `artifacts/models/attrition_v2.joblib` |
| 14 Risk explanations | 430-row CSV in the sample format: calibrated score, band, three plain-English reasons, human-review note | `src/cortexhr/phase3/explanations/`, `reports/task14/attrition_risk_2026.csv` |
| 15 Pay bands | Role baseline vs Ridge vs gradient boosting, low / mid / high band, gaps by gender and career gap | `src/cortexhr/phase3/salary/`, `reports/task15/` |

All 15 notebooks (`notebooks/01` to `15`) cover the 38-stage workflow of Phase 1 and 2 and include saved outputs; stages that do not apply say why. The final audit structurally validated all 15 notebooks, but did not perform a clean-kernel full rebuild of every notebook in one pass (see `docs/final_audit_status.md`).

## Headline results

See `reports/taskNN/README.md` for each task (generated from the saved results) and `docs/phase3_quality_gate.md` for the test summary.

## Run it

Python 3.11 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt -r requirements-optional.txt
python tests/run_all.py                    # or: pytest -q tests
python scripts/execute_notebooks.py        # re-run all 15 notebooks (about 12 minutes on one CPU)
python scripts/make_docs.py                # regenerate model cards, task reports, traceability
python scripts/run_quality_gate.py         # tests + traceability gate -> docs/phase3_quality_gate.md
```

`lightgbm` and `shap` are optional extras for reproducing the separate LightGBM challenger and the native-SHAP cross-check. Install `requirements-optional.txt` to reproduce those audit outputs. The production Task 13 model remains the LR + HGB blend; the Task 14 full-population explanations remain grouped Monte Carlo Shapley (see `docs/limitations.md`).

## Honest notes

- The attrition v2 gain over the simple baseline is **not proven** (Task 13 says so and why).
- The 430 employee-level Task 14 explanations are produced by the grouped Monte Carlo Shapley estimator. SHAP 0.50.0 was subsequently installed and its `PermutationExplainer` was run as a separate cross-check on 10 employees only; it was not used to generate the full 430-row explanation report.
- Group fairness measurements use two restricted files (`04_Validation_Vault/unlock_CX-12...`, `unlock_CX-15...`) that are **never committed**; without them those cells say they were skipped.
- Everything here is decision support. A human decides.

## Documentation

`docs/architecture.md`, `methodology.md`, `feature_definitions.md`, `limitations.md`, `experiment_log.md`, `final_audit_status.md`, `model_cards/`, `fairness/`, `requirements_traceability.md`, `phase3_quality_gate.md`.


Answers to the five "Think for yourself" questions: `docs/think_for_yourself.md`. `python tests/run_all.py` prints installed vs pinned versions and reports any test it cannot run as SKIP with the reason.

Optional strict check: `python scripts/run_shap_full430.py` runs the real `shap` library on all 430 employees and writes `reports/task14/shap_full430_*` comparing its top-3 reasons with the Monte Carlo ones. It needs `shap` (requirements-optional.txt).

**Native SHAP on all 430 employees (run on the author's machine, 2026-11-06):** shap 0.54.1 PermutationExplainer on all 430 employees (50-row background): additivity error 5.6e-16; mean top-3 overlap with the grouped Monte Carlo reasons 0.82; all three reasons identical for 50.0% of employees; same #1 reason for 68.1%; median per-person correlation of group contributions 0.954. Output files: `reports/task14/shap_full430_group_values.csv`, `shap_full430_comparison.csv`, `shap_full430_summary.json` (rows = 430). The 430-row HR report still uses the grouped Monte Carlo reasons; the two methods agree on all three reasons for half of the employees, and differences are expected because the Monte Carlo method treats related columns as grouped players.
