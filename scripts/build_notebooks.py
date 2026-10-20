"""Writes the four learning notebooks (plain .ipynb JSON, no nbformat needed). Run execute_notebooks.py afterwards."""
import json, pathlib
OUT = pathlib.Path(__file__).resolve().parents[1] / "05_Starter" / "notebooks"; OUT.mkdir(parents=True, exist_ok=True)
SETUP = """import sys, pathlib
ROOT = pathlib.Path.cwd().resolve()
while not (ROOT / 'src').exists(): ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 50); pd.set_option('display.max_colwidth', 120)
from cortexhr import config as C, data_loading as D, audit as A, features as F, modeling as M, evaluation as E, reporting as R, runs"""

def nb(cells):
    out = []
    for kind, src in cells:
        c = {"cell_type": kind, "metadata": {}, "source": src.strip("\n").splitlines(True)}
        if kind == "code": c.update(execution_count=None, outputs=[])
        out.append(c)
    return {"cells": out, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}

md, code = (lambda s: ("markdown", s)), (lambda s: ("code", s))

N1 = [md("# 01 - Data audit (Day 1)\nAudit the attrition data **before** any model exists. Run top to bottom from a clean kernel.\n\n**Rule:** finish and commit this audit first; only then open the files in `04_Validation_Vault` (see `reports/vault_comparison_cx01.md`)."),
 code(SETUP),
 md("## 1. Load `attrition_dataset`: shape, columns, types, first 5 rows"),
 code("raw = D.load_attrition()\nprint('shape:', raw.shape)\nprint(list(raw.columns))\nraw.dtypes.to_frame('dtype')"), code("raw.head()"),
 md("**One row** = one employee at one observation date (a snapshot of their features). **Target `left_within_12m`** = 1 if that employee left within 12 months after `observation_date`, else 0."),
 md("## 2. Schema and data-quality inventory"), code("A.schema_inventory(raw)"),
 md("## 3. Missing values"), code("A.missingness(raw)"),
 md("## 4. Duplicates"), code("A.duplicates(raw)"),
 md("## 5. Impossible values (negative tenure, percentages above 100, ...)\nThere is no `age` column in this table, so 'age under 18' cannot be tested here."), code("A.impossible_values(raw)"),
 md("## 6. Outliers: IQR rule and Z-score\nOutliers are **investigation candidates, not automatic errors**: a very high salary can be a real senior person."),
 code("iq = A.iqr_outliers(raw); zs = A.zscore_outliers(raw)\niq.sort_values('iqr_outliers', ascending=False).head(10)"), code("zs.sort_values('z_outliers_abs_gt_3', ascending=False).head(10)"),
 code("A.pay_consistency_outliers(raw)  # salary typos: pay / comp_ratio should be ~constant inside a role"),
 md("## 7. Class imbalance and the majority-class trap"),
 code("cb = A.class_balance(raw); print(cb)\nprint(f\"Always predicting 'stay' is {cb['majority_accuracy_pct']}% accurate but finds 0 of {cb['left']} leavers.\")\nA.plot_class_balance(cb);"),
 md("## 8. Leakage hunt: prove each suspect"),
 code("post = [c for c in C.LEAKAGE_CONFIRMED if c != 'voluntary_left_within_12m']\nA.missingness_indicator_power(raw, post)"),
 code("lk = A.leakage_review(raw); lk[['column', 'verdict', 'evidence']]"),
 code("single = A.single_column_power(raw, C.NUMERIC_CORE + C.CATEGORICAL_CORE); single.head(10)  # best legitimate single column"),
 code("A.plot_leakage(A.missingness_indicator_power(raw, post), single);"),
 md("## 9. Audit table (problem, where, how many rows, evidence, severity, treatment, reason)"),
 code("audit_t, _, _ = A.build_audit_table(raw, D.load_sheet(C.SHEET_ATT), D.load_sheet(C.SHEET_PAY))\nprint(len(audit_t), 'distinct findings')\naudit_t[['problem', 'affected_column_or_location', 'rows_affected', 'severity', 'proposed_treatment']]"),
 code("audit_t[['problem', 'evidence', 'reason']]"),
 md("## 10. Decision per column (feeds Day 2)"), code("roles = A.column_roles(raw); roles.decision.value_counts()"), code("roles[roles.decision.isin(['DROP', 'TARGET'])]"),
 code("R.show_figure('audit_table', 14);"),
 md("Full written report: `reports/data_audit_report.md`. Vault comparison (after commit): `reports/vault_comparison_cx01.md`.")]

N2 = [md("# 02 - Reusable pipeline and honest baselines (Days 2-3)"), code(SETUP),
 md("## Day 2: safe columns and cutoff-safe attendance/payroll features"),
 code("S = runs.load_model_frame(include_monthly=True)\nS['join_report']"),
 code("print('Inputs:', len(S['X'].columns), 'columns; target in X?', C.TARGET in S['X'].columns)\nprint('Dropped for leakage/ids/dates/metadata:', C.FORBIDDEN_AS_INPUT)"),
 md("### `build_pipeline()` and the 3-row smoke test\nThe pipeline cleans impossible values, adds derived features (department-relative pay etc.), imputes, scales and one-hot encodes. Everything it learns comes from the rows passed to `fit`."),
 code("pipe = F.build_pipeline(True).fit(S['Xtr'], S['ytr'])\nraw3 = F.representative_rows(S['df'])\nZ = pipe.transform(raw3[F.input_columns(True)])\nprint('3 raw rows ->', Z.shape, '| all finite:', bool(np.isfinite(Z.to_numpy()).all()))\nraw3[['row_id','tenure_months','work_location','last_engagement_score']]"),
 code("Z.iloc[:, :8]"),
 code("t = raw3[F.input_columns(True)].copy(); t.iloc[0, t.columns.get_loc('role_id')] = 'R999'; t.iloc[1, t.columns.get_loc('department_id')] = np.nan\nprint('unseen category + missing department handled:', bool(np.isfinite(pipe.transform(t).to_numpy()).all()))"),
 code("pd.DataFrame(F.FEATURE_DOCS, columns=['feature', 'source', 'definition', 'temporal cutoff'])"),
 md("**Why fit on training data only (3 sentences).** 1) Medians, means, categories and department pay medians describe the data; if they are computed on test rows, test information leaks into training and scores look better than reality. 2) A new employee scored tomorrow is, by definition, not in the data the pipeline learned from, so the honest test is data the pipeline never saw. 3) `Pipeline.fit()` inside every cross-validation fold guarantees that."),
 md("## Day 3: stratified split, shared folds, baselines"),
 code("print('train', S['Xtr'].shape, 'test', S['Xte'].shape, '| leaver rate train/test:', round(S['ytr'].mean(), 4), round(S['yte'].mean(), 4))\nprint('fold sizes (train, valid):', [(len(a), len(b)) for a, b in S['folds']])"),
 code("o = runs.run_baselines(write=False)\no['comparison'].round(3)"),
 code("o['fold_metrics'][['model', 'fold', 'roc_auc', 'pr_auc', 'precision', 'recall', 'f1', 'accuracy', 'precision_defined']].round(3).head(20)"),
 md("Undefined precision (no positive predictions, as for the Dummy) is reported as 0 via `zero_division=0` and flagged by `precision_defined`."),
 code("print(o['tree']); o['imbalance'].round(3)"),
 code("o['comparison_core'][['model','roc_auc','benchmark_roc_auc','delta_roc_auc']].round(3)"),
 md("### Five-sentence interpretation"), code("print(*[f'{i}. {s}' for i, s in enumerate(o['interpretation'], 1)], sep='\\n')"),
 code("o['summary'][['model', 'roc_auc_mean', 'leakage_check']]"),
 md("Warnings captured during the run (none are hidden):"), code("o['warnings'] or 'none'")]

N3 = [md("# 03 - Honest evaluation and risk bands (Day 4)\nEverything below is reproducible from this notebook."), code(SETUP),
 code("ev = runs.run_evaluation(write=False)\nev['metrics'].round(3).T"),
 md("**ROC-AUC**: chance a random leaver outranks a random stayer. **PR-AUC**: purity of the flagged list when few people leave (compare with the base rate). **Top-10%**: review only the highest-scored 10%; ties broken by row position."),
 code("S = ev['state']; E.plot_roc_pr({'OOF': (S['ytr'], ev['oof']), 'Test': (S['yte'], ev['p_test'])});"),
 md("## Calibration (fitted on training OOF only, never on test labels)"),
 code("E.plot_calibration(ev['calibration']);"), code("print(ev['brier'])"),
 md("## Risk bands: High = top 10%, Medium = next 20%, Low = rest"),
 code("for k, t in ev['band_summary'].items():\n    print(k); print(t.round(2).to_string())"),
 code("E.plot_bands(ev['band_summary']['Held-out test']);"),
 md("## Lift / gains vs random selection"), code("ev['gains']['Held-out test'].round(2)"), code("E.plot_gains_lift(ev['gains']);"),
 md("Manager note: `reports/manager_summary.md`. Full report: `reports/model_evaluation_report.md`."),
 code("print(open(C.project_root() / 'reports' / 'manager_summary.md').read())")]

N4 = [md("# 04 - Attendance anomalies without labels (Day 5)\nLabels are used **only** to measure, after the flags are saved."), code(SETUP),
 code("from cortexhr import attendance as T\natt = D.load_sheet(C.SHEET_ATT); att.describe().T[['min', 'mean', 'max']]"),
 md("## Model: IsolationForest (3% contamination) + two simple rules"),
 code("flags = T.detect(att, contamination=0.03, seed=C.SEED)\nflags[['iforest_flag', 'iforest_raw_only_flag', 'rule_zero_attendance', 'rule_implausible_hours', 'rule_flag', 'flagged']].sum()"),
 code("T.assert_no_labels(T.RAW_FEATURES + T.HISTORY_FEATURES); print('features:', T.RAW_FEATURES + T.HISTORY_FEATURES)"),
 md("## Ranking flags for a limited review budget"), code("T.review_queue(flags, 15)[['attendance_feature_id', 'employee_id', 'month', 'flag_reason', 'priority_score', 'review_rank']]"),
 md("## Measurement against hidden labels\nThe flags are saved first (`runs.run_attendance`), then labels are read. If the labels file is not in `04_Validation_Vault/` the measurement is reported as NOT MEASURED."),
 code("res = runs.run_attendance(write=False)\nprint('labels file:', res['labels_path'] or 'NOT SUPPLIED -> monthly precision/recall not measured')\nres['measured']['overall'].round(3) if res['measured'] else res['benchmark']"),
 code("res['measured']['by_type'] if res['measured'] and res['measured']['by_type'] is not None else 'anomaly-type breakdown needs the labels file'"),
 md("## Supplementary: labelled DAILY sample (different table, measurement only)"), code("res['daily_proxy'].round(3)"),
 md("Report: `reports/attendance_anomaly_report.md`.")]

for name, cells in {"01_data_audit": N1, "02_baselines": N2, "03_model_evaluation": N3, "04_attendance_anomalies": N4}.items():
    (OUT / f"{name}.ipynb").write_text(json.dumps(nb(cells), indent=1)); print("wrote", name)
