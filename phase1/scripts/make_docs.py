"""Generates README, architecture, workflow coverage, weekly demo, daily log and the model card from real outputs."""
import sys, pathlib, datetime
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from cortexhr import config as C, runs, evaluation as E, modeling as M, workflow as W, reporting as R
H = E.H; ROOT = C.project_root(); T = ROOT / "reports" / "tables"

def main():
    comp = pd.read_csv(T / "baseline_comparison.csv").set_index("model"); lr = comp.loc["Logistic Regression (balanced)"]
    ev = pd.read_csv(T / "evaluation_metrics.csv"); bands = pd.read_csv(T / "risk_band_counts.csv"); bt = bands[bands.scored_set == "Held-out test"].set_index("band")
    att = pd.read_csv(T / "attendance_detector_metrics.csv").set_index("detector") if (T / "attendance_detector_metrics.csv").exists() else None
    S = runs.load_model_frame(True); Xtr, ytr, Xte, yte = S["Xtr"], S["ytr"], S["Xte"], S["yte"]
    model = M.model_zoo(True)["Logistic Regression (balanced)"].fit(Xtr, ytr); p = model.predict_proba(Xte)[:, 1]
    off = H.binary_report(yte, p); dfj = S["df"]
    gaps = {g: H.group_gap(p, dfj.loc[Xte.index, g].values) for g in ["work_mode", "department_id"]}
    coef = W.logreg_coefficients(model, 8)
    i = int(np.argmax(p)); row = Xte.iloc[[i]]; Z = model.named_steps["features"].transform(row); contrib = (Z.iloc[0] * model.named_steps["model"].coef_[0]).sort_values(ascending=False)
    raw_show = row[["tenure_months", "comp_ratio", "months_since_last_raise", "last_engagement_score", "avg_overtime_hours_per_month_3m"]].iloc[0].round(2).to_dict()
    ver = "attrition-lr-balanced-v1.0 (" + datetime.date.today().isoformat() + ")"
    card = f"""# Model Card: CortexHR attrition baseline (balanced logistic regression)   (CX-20 draft from Phase 1)
**Purpose:** rank employees by estimated risk of leaving within 12 months so HR can prioritise conversations. **Not for:** hiring, firing, promotion, pay or any automatic employment decision; not for use on individuals without human review.

**Data:** sheet `attrition_dataset` (2,389 rows, snapshot dates {S['raw']['observation_date'].min().date()} to {S['raw']['observation_date'].max().date()}); 6 duplicate rows removed (2,383 used); removed as inputs: 6 post-outcome columns (exit_date, resignation_date, exit_reason, exit_interview_done, fnf_settlement_amount, voluntary_left_within_12m), `source`, `observation_date`, ids; no protected attributes in the table. Attendance/payroll aggregates (last 3 complete months before the snapshot) exist for only {S['join_report']['attendance_matched_rows']} rows.

**Method:** leakage-safe `build_pipeline()` (clean, derived features incl. pay vs department median, raise-recency buckets, overtime flag; median impute + indicators, scale, one-hot); LogisticRegression(class_weight=balanced); stratified 80/20 split (seed {C.SEED}) and stratified 5-fold CV on the 80%.

**Metrics** (official `cortexhr_helpers.binary_report`; compared with `baseline_benchmarks`):

| Measure | Mine | Benchmark |
|---|---|---|
| ROC-AUC (5-fold CV) | {lr.roc_auc:.3f} | {lr.benchmark_roc_auc:.3f} |
| PR-AUC (5-fold CV) | {lr.pr_auc:.3f} | {lr.benchmark_pr_auc:.3f} |
| ROC-AUC (held-out test) | {off['roc_auc']:.3f} | |
| PR-AUC (held-out test, base rate {off['base_rate']:.3f}) | {off['pr_auc']:.3f} | |
| Precision / recall in the top-10% group (test) | {off['precision_at_top']:.3f} / {off['recall_at_top']:.3f} | |
| High band (test): leavers / employees | {int(bt.loc['High', 'leavers'])} / {int(bt.loc['High', 'employees'])} ({bt.loc['High', 'leaver_rate_pct']:.1f}%), holds {bt.loc['High', 'share_of_all_leavers_pct']:.0f}% of all leavers | |

**Explanations:** strongest coefficients (standardised inputs):

{R.md_table(coef)}

Worked example (highest-scored test employee, score {p[i]:.2f}): raw values {raw_show}. Largest contributions to the score: {', '.join(f'`{k}` ({v:+.2f})' for k, v in contrib.head(4).items())}.

**Fairness:** the table has no protected attributes, so group checks use available groups with the official `group_gap`: work_mode largest gap in mean score {gaps['work_mode'][1]:.3f}; department {gaps['department_id'][1]:.3f}. Gaps are warnings, not proof; real fairness testing needs the protected-attributes data (age, gender), which is deliberately kept out of the model.

**Limits & risks:** only {int(yte.sum())} test leavers (noise of several points); 84% of rows lack monthly history; 12% of leavers had already resigned before the snapshot; raw probabilities are inflated by class weights (use risk bands or calibrated scores); drift signs: score PSI above 0.25, prevalence far from 14.6%.

**Version & owner:** {ver}; owner: Jaisurya Marisamy.
"""
    (ROOT / "docs" / "model_cards" / "attrition_baseline.md").write_text(card)

    cov = [(1, "Business Understanding", "1"), (2, "Problem Definition", "2"), (3, "Problem Feasibility", "3"), (4, "Data Collection", "4"), (5, "Data Loading and Initial Inspection", "5"), (6, "EDA", "6"),
           (7, "Missing-Value Analysis", "7"), (8, "Duplicate-Record Analysis", "8"), (9, "Outlier Analysis", "9"), (10, "Data Consistency and Validity", "10"), (11, "Data Leakage Investigation", "11"),
           (12, "Target Variable Preparation", "12"), (13, "Feature Identification", "13"), (14, "Train/Validation/Test Split", "14"), (15, "Numerical Preprocessing", "15"), (16, "Categorical Preprocessing", "16"),
           (17, "Datetime and Text Preprocessing", "17"), (18, "Preprocessing Pipeline", "18"), (19, "Feature Engineering", "19"), (20, "Feature Selection", "20"), (21, "Dimensionality Reduction", "21"),
           (22, "Class Imbalance Analysis", "22"), (23, "Imbalance Handling Techniques", "23"), (24, "Establish a Baseline", "24"), (25, "Candidate Algorithms", "25"), (26, "Initial Model Training", "26"),
           (27, "Cross-Validation", "27"), (28, "Hyperparameter Tuning", "28"), (29, "Threshold Optimization", "29"), (30, "Confusion Matrix", "30"), (31, "Classification Metrics", "31"),
           (32, "Metric Selection", "32"), (33, "Probability and Calibration", "33"), (34, "Error Analysis", "34"), (35, "Over/Underfitting", "35"), (36, "Model Comparison and Selection", "36"), (37, "Final Model Training", "37")]
    deep = {1: "01_data_audit (stages 1-3), 05 §1-3", 4: "01_data_audit", 5: "01_data_audit", 6: "01_data_audit stage 4", 7: "01_data_audit", 8: "01_data_audit", 9: "01_data_audit", 10: "01_data_audit stage 5",
            11: "01_data_audit stage 3, 02_baselines leaky demo", 15: "01b_feature_pipeline", 16: "01b_feature_pipeline", 17: "01b_feature_pipeline (cutoff)", 18: "01b_feature_pipeline", 19: "01b_feature_pipeline",
            22: "02_baselines", 23: "02_baselines (SMOTE)", 24: "02_baselines", 25: "02_baselines", 26: "02_baselines", 27: "02_baselines", 28: "02_baselines stage 10", 29: "03_model_evaluation stage 10", 30: "03_model_evaluation stage 9",
            31: "03_model_evaluation stage 9", 33: "03_model_evaluation (calibration)", 34: "03_model_evaluation stage 11", 35: "02_baselines (tree vs CV)", 36: "02_baselines", 37: "03_model_evaluation (test scored once)"}
    t = "# Workflow coverage: your 37-point classification workflow -> where it is implemented\n\nEvery item is a numbered section of `05_Starter/notebooks/05_complete_classification_workflow.ipynb` (requirement-by-requirement coverage table + real code + outputs). Task notebooks go deeper on their own phase.\nThe supplied workflow text ended at item 37 (its first bullet); if more items exist they are not covered yet.\n\n| # | Item | Master notebook section | Also in |\n|---|---|---|---|\n"
    t += "\n".join(f"| {n} | {title} | section {s} | {deep.get(n, '05 only')} |" for n, title, s in cov)
    t += "\n\nAlgorithms not runnable here (libraries not installed in the build environment): XGBoost, LightGBM, CatBoost and Bayesian optimisation (optuna). The cells are guarded: they run where the library is installed and say so where it is not.\n"
    (ROOT / "docs" / "workflow_coverage.md").write_text(t)

    arch = """# Architecture (Phase 1)

```mermaid
flowchart LR
  X[CortexHR_Dataset.xlsx<br/>03_Data] --> L[data_loading<br/>clean_raw, dedupe, join_monthly_features<br/>cutoff = observation_date]
  L --> A[audit.py<br/>quality, leakage proof] --> R1[reports/data_audit_report.md]
  L --> F[features.py build_pipeline<br/>RawCleaner -> Derive -> ColumnTransformer]
  F --> M[modeling.py<br/>stratified split, 5-fold, Dummy/LR/Tree/RF/SMOTE]
  M --> E[evaluation.py + official cortexhr_helpers<br/>binary_report, risk_bands]
  E --> R2[reports: metrics, calibration, bands, manager_summary]
  X --> T[attendance.py<br/>IsolationForest + rules] --> R3[attendance flags, catch_rate vs hidden labels]
  V[(04_Validation_Vault<br/>restricted, git-ignored)] -. measure only .-> R3
  V -. compare only, after audit .-> A
```

Rules the design enforces: nothing learned from the test rows (pipeline fitted on training rows only), no post-outcome columns (`config.FORBIDDEN_AS_INPUT`), monthly features only from complete months before the snapshot, vault files never used as inputs.
"""
    (ROOT / "docs" / "architecture.md").write_text(arch)

    demo = f"""# Week 1 demo script (2 minutes) - follows WEEKLY_DEMO_CHECKLIST.md
Draft; say it in your own words and be ready to explain every line.

1. **Who:** "I am Jaisurya Marisamy, AI/ML track. This week I finished CX-01 to CX-05: audit, pipeline, baselines, evaluation, attendance anomalies."
2. **Show it working (not slides):** run `python scripts/run_pipeline.py` (3-row smoke test passes) and open `reports/figures/audit_table.png` and `reports/figures/gains_lift_chart.png`.
3. **One number that proves it:** logistic regression ROC-AUC {lr.roc_auc:.3f} vs benchmark {lr.benchmark_roc_auc:.3f}; the leaky demo scored 1.00 (that is why I removed 6 columns); High risk band: {int(bt.loc['High','leavers'])} of {int(bt.loc['High','employees'])} left ({bt.loc['High','leaver_rate_pct']:.0f}% vs {100 * yte.mean():.0f}% overall).{'' if att is None else f" Attendance detector: precision {att.loc['Combined (IF + rules)', 'precision']:.2f}, recall {att.loc['Combined (IF + rules)', 'recall']:.2f} vs benchmark 0.40 / 0.59."}
4. **A problem I hit:** I called all salary extremes "real senior pay" in my audit and missed 3 salary typos; the vault showed pay/comp_ratio was ~11x its role. Lesson: compare related columns, not just single columns. (Also: late-arrival spikes were the hardest anomaly type; zero-attendance rule fired on paid leave.)
5. **Next week:** payroll anomalies (Phase 2), reusing the same measurement discipline.
"""
    (ROOT / "docs" / "weekly_demo_week1.md").write_text(demo)

    log = """# Daily Log (Week 1 draft: replace the dates and edit so it is truly yours)

**Date:** Day 1  **Done today:** set up repo and venv; audited attrition_dataset (17 findings, 6 leakage columns, class balance 14.6%); committed the audit before opening the vault.  **Doing next:** feature pipeline.  **Blocked by:** nothing.  **One thing I learned:** compare related columns (pay / comp_ratio), not only each column alone.

**Date:** Day 2  **Done today:** build_pipeline() with train-only fitting, attendance/payroll aggregates with a date cutoff, 3-row smoke test.  **Doing next:** baselines.  **Blocked by:** nothing.  **One thing I learned:** monthly aggregates exist for only 383 of 2,383 rows, so they barely move the score.

**Date:** Day 3  **Done today:** Dummy, logistic regression, tree, forest on identical folds; compared with benchmark; SMOTE comparison.  **Doing next:** evaluation.  **Blocked by:** nothing.  **One thing I learned:** a deliberately leaky model scores 1.00; anything above 0.90 means "find the leak".

**Date:** Day 4  **Done today:** official binary_report and risk_bands, calibration (raw balanced scores are inflated), lift chart, manager note.  **Doing next:** anomalies.  **Blocked by:** nothing.  **One thing I learned:** show risk bands, not raw probabilities.

**Date:** Day 5  **Done today:** IsolationForest + two rules; measured with hidden labels after saving flags (precision 0.50, recall 0.84).  **Doing next:** Phase 2 payroll.  **Blocked by:** nothing.  **One thing I learned:** a rule fired on months fully covered by paid leave: legitimate absences are the false-positive trap.
"""
    (ROOT / "docs" / "daily_log.md").write_text(log)

    readme = f"""# CortexHR AI/ML - Phase 1: Data Foundations & Baseline Models
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
2. `python -m venv .venv` then activate (`source .venv/bin/activate`, Windows: `.venv\\Scripts\\activate`)
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
| Logistic regression, 5-fold CV | ROC-AUC {lr.roc_auc:.3f} vs benchmark {lr.benchmark_roc_auc:.3f} (PR-AUC {lr.pr_auc:.3f}) |
| Held-out test (scored once) | ROC-AUC {off['roc_auc']:.3f}, PR-AUC {off['pr_auc']:.3f} (base rate {off['base_rate']:.3f}), top-10% lift {off['precision_at_top'] / off['base_rate']:.2f}x |
| High risk band (test) | {int(bt.loc['High','leavers'])} of {int(bt.loc['High','employees'])} left ({bt.loc['High','leaver_rate_pct']:.0f}%) |
| Attendance detector vs hidden labels | {'precision %.3f, recall %.3f (benchmark 0.403 / 0.593); flags committed before labels were opened' % (att.loc['Combined (IF + rules)', 'precision'], att.loc['Combined (IF + rules)', 'recall']) if att is not None else 'labels not present in this copy'} |
| Audit vs planted dirt | every planted issue found at row level except 3 salary typos (missed originally, now caught) |
| Tests | `pytest` (data, features, leakage, split, metrics, bands, official-helper parity, notebooks, script outputs) |

## Limitations and what I would do next
1. Only 383 of 2,383 rows have attendance/payroll history; get it for the 2,000 historical rows.
2. 12% of leavers had already resigned before the snapshot; add a notice flag available before the snapshot.
3. Test set has {int(yte.sum())} leavers: differences of a few points are noise; test ROC-AUC sits at about 0.69-0.70.
4. The v2 attendance detector row is post-hoc (written after seeing the labels) and optimistic.
5. Not run in the build environment: Docker, XGBoost/LightGBM/CatBoost, optuna, real JupyterLab (notebooks were executed with `scripts/execute_notebooks.py`); tested on Python 3.12, not 3.11.
6. Scores support human review only and must never decide an employment action (`docs/model_cards/attrition_baseline.md`).
"""
    (ROOT / "README.md").write_text(readme)

if __name__ == "__main__":
    main()
