# CortexHR AI/ML - Phase 1: Data Foundations & Baseline Models
An audited HR attrition dataset, a leakage-aware reusable feature pipeline, honest baseline models, a risk-band evaluation for managers and an attendance anomaly detector, built for Zenith Nova Technologies (part of Aura Workforce, fictional data). For HR analysts and the AI/ML training evaluation.

## Live demo
Not applicable in Phase 1 (deployment is Phase 4). The working evidence is the executed notebooks in `05_Starter/notebooks/` and the reports in `reports/`.

## Features
- **Audit (CX-01):** 17 evidence-backed findings, 6 confirmed leakage columns with proof, row-level vault comparison.
- **Pipeline (CX-02):** `build_pipeline()` fits on training rows only; cutoff-safe attendance/payroll aggregates; level-ups: raise-recency buckets, overtime flag, department-relative pay.
- **Baselines (CX-03):** Dummy, logistic regression, tree, forest, SMOTE; identical folds; compared with `baseline_benchmarks`.
- **Evaluation (CX-04):** official `binary_report`/`risk_bands`, calibration, lift, manager note.
- **Anomalies (CX-05):** IsolationForest + rules, measured against hidden labels.
- **Full workflow:** all 37 items of the classification workflow in `05_complete_classification_workflow.ipynb` (see `docs/workflow_coverage.md`).

## Tech stack
Python 3.11+, pandas, NumPy, scikit-learn, Matplotlib, joblib, openpyxl, JupyterLab, pytest, Docker (see `requirements.txt`).

## Run it locally
1. `unzip cortexhr_phase1.zip && cd cortexhr_phase1`
2. `python -m venv .venv` then activate (`source .venv/bin/activate`, Windows: `.venv\Scripts\activate`)
3. `pip install -r requirements.txt && pip install -e .`
4. `bash scripts/run_all.sh` (or run the five scripts one by one, see `docs/requirements_traceability.md`)
5. `pytest -v`
6. `jupyter lab --no-browser`, open `05_Starter/notebooks/`, **Kernel -> Restart and Run All**
7. Optional measurement: copy the vault files into `04_Validation_Vault/` (restricted, git-ignored) and rerun `python scripts/detect_attendance_anomalies.py` and `python scripts/compare_vault_cx01.py`.
Docker: `docker compose up --build` (JupyterLab at http://localhost:8888/?token=cortexhr; not tested in the build environment).

## Architecture
See `docs/architecture.md` (Mermaid diagram). Flow: workbook -> cleaning/join with date cutoff -> audit and pipeline -> models -> official evaluation helpers -> reports; vault files are used for measurement only.

## How I verified it works
| Check | Result |
|---|---|
| Leakage guard | leaky demo ROC-AUC 1.00 (flagged); no honest model above 0.90 |
| Logistic regression, 5-fold CV | ROC-AUC 0.709 vs benchmark 0.713 (PR-AUC 0.325) |
| Held-out test (scored once) | ROC-AUC 0.691, PR-AUC 0.355 (base rate 0.147), top-10% lift 2.75x |
| High risk band (test) | 19 of 48 left (40%) |
| Attendance detector vs hidden labels | precision 0.500, recall 0.840 (benchmark 0.403 / 0.593); flags committed before labels were opened |
| Audit vs planted dirt | every planted issue found at row level except 3 salary typos (missed originally, now caught) |
| Tests | `pytest` (data, features, leakage, split, metrics, bands, official-helper parity, notebooks, script outputs) |

## Limitations and what I would do next
1. Only 383 of 2,383 rows have attendance/payroll history; get it for the 2,000 historical rows.
2. 12% of leavers had already resigned before the snapshot; add a notice flag available before the snapshot.
3. Test set has 70 leavers: differences of a few points are noise; test ROC-AUC sits at about 0.69-0.70.
4. The v2 attendance detector row is post-hoc (written after seeing the labels) and optimistic.
5. Not run in the build environment: Docker, XGBoost/LightGBM/CatBoost, optuna, real JupyterLab (notebooks were executed with `scripts/execute_notebooks.py`); tested on Python 3.12, not 3.11.
6. Scores support human review only and must never decide an employment action (`docs/model_cards/attrition_baseline.md`).
