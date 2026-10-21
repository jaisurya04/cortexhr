# CortexHR AI/ML - Phase 1: Data Foundations & Baseline Models

Audited HR attrition dataset -> leakage-aware reusable feature pipeline -> honest baselines -> evaluation and risk bands -> attendance anomaly detector. Built from `03_Data/CortexHR_Dataset.xlsx` (fictional data, Zenith Nova Technologies).

## Headline results (all from real runs; seed 42)
| Item | Result |
|---|---|
| Data audit | 2,389 rows x 42 cols, 349 leavers (14.61%), **17 evidence-backed findings**, 6 confirmed leakage columns |
| Logistic regression (balanced), 5-fold CV on training rows | ROC-AUC **0.698** (benchmark 0.713), PR-AUC 0.312, recall 0.57, precision 0.25 |
| Held-out test (used once) | ROC-AUC 0.682, PR-AUC 0.321, top-10% precision 0.38 / recall 0.26, lift 2.56x |
| High risk band (test) | 18 of 48 left (37.5%), holds 26% of all leavers |
| Attendance anomalies (monthly) | 252 rows flagged (IsolationForest 3% + 2 rules). **Precision/recall vs hidden labels NOT measured: labels file was not in the supplied vault** (see below) |
| Tests | 25 tests (`pytest`) |

## Quick start (local)
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
bash scripts/run_all.sh                                # audit, baselines, evaluation, anomalies, notebooks, tests
jupyter lab                                            # then open 05_Starter/notebooks/
pytest                                                 # preferred test runner
```
## Docker
```bash
docker compose up --build        # JupyterLab at http://localhost:8888/?token=cortexhr
docker compose run --rm cortexhr python scripts/run_audit.py
```
## GitHub (private repo `cortexhr`) - not created for you, needs your account
```bash
git remote add origin git@github.com:jaisurya04/cortexhr.git   # create the PRIVATE repo "cortexhr" on GitHub first
git push -u origin main
```
`04_Validation_Vault/` is git-ignored and docker-ignored on purpose.

## Layout
`src/cortexhr/` modules (`data_loading, audit, features, modeling, evaluation, attendance, reporting, runs, config`) · `scripts/` one script per task (`run_audit.py`, `run_pipeline.py`, `train_baselines.py`, `evaluate_models.py`, `detect_attendance_anomalies.py`; `run_all.sh` runs all) · `05_Starter/notebooks/` five executed notebooks, one per task · `reports/` markdown reports, `tables/`, `figures/` · `artifacts/` fitted pipeline + run metadata · `tests/` · `docs/requirements_traceability.md` · `docs/feature_definitions.md`.

## Notebooks (one per task, each follows the same 12-stage workflow)
Data Collection, Data Understanding, Data Quality Audit, EDA, Data Cleaning, Feature Engineering, Train-Test Split, Model Building, Model Evaluation (with confusion matrices), Hyperparameter Tuning, Model Interpretation, Final Conclusions and Recommendations.

| Task | Notebook |
|---|---|
| 1 Audit | `01_data_audit.ipynb` |
| 2 Pipeline | `01b_feature_pipeline.ipynb` |
| 3 Baselines | `02_baselines.ipynb` |
| 4 Evaluation | `03_model_evaluation.ipynb` |
| 5 Anomalies | `04_attendance_anomalies.ipynb` |

Each ends with a code-checked completion checklist (`[PASS]` / `[MANUAL]` for things only you can do, such as the GitHub link or the Day 5 labels).

## Reports to read
`reports/data_audit_report.md` · `reports/vault_comparison_cx01.md` · `reports/baseline_model_comparison.md` · `reports/model_evaluation_report.md` · `reports/manager_summary.md` · `reports/attendance_anomaly_report.md`

## Honest limitations (read these)
1. **Official starter files were not supplied**: `05_Starter/cortexhr_helpers.py`, the starter notebooks, starter `requirements.txt`, and `03_Data/csv/`. Nothing was fabricated as "official": `binary_report`, `risk_bands`, `catch_rate` are my own implementations in `src/cortexhr/evaluation.py`; the notebooks are written from scratch.
2. **Vault**: Drive's `04_Validation_Vault` held only `unlock_CX-01_answer_key_notes.csv`. Row-level `answer_key_planted_dirt` and the attendance labels (`unlock_CX-05`) were missing, so the CX-01 comparison is count-level and the Day 5 monthly precision/recall is **pending**; drop the labels file into `04_Validation_Vault/` and rerun `python scripts/detect_attendance_anomalies.py`. A supplementary labelled *daily* sample (in the main workbook) was measured separately and is labelled as such.
3. **Build sandbox had no internet**: pytest, JupyterLab, nbformat, imbalanced-learn and Docker could not be installed/run. Tests ran via `tests/run_all.py` (pytest-style asserts); notebooks were executed by `scripts/execute_notebooks.py`; **Dockerfile is untested**; SMOTE is an in-house implementation. Tested on Python 3.12.3, not 3.11.
4. **Attendance/payroll features cover only 383 of 2,383 rows** (the 2,000 historical rows have no monthly history), so they barely change model scores (ablation reported).
5. Test set has only 70 leavers: small-sample noise of several points is expected.

## References
scikit-learn https://github.com/scikit-learn/scikit-learn · pandas https://github.com/pandas-dev/pandas · JupyterLab https://github.com/jupyterlab/jupyterlab · pytest https://github.com/pytest-dev/pytest · imbalanced-learn https://github.com/scikit-learn-contrib/imbalanced-learn · Matplotlib https://github.com/matplotlib/matplotlib · PyOD https://github.com/yzhao062/pyod
