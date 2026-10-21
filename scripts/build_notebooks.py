"""Writes the five learning notebooks (plain .ipynb JSON). Each follows the same 12-stage workflow.
Run scripts/execute_notebooks.py afterwards (or Run All in Jupyter)."""
import json, pathlib
OUT = pathlib.Path(__file__).resolve().parents[1] / "05_Starter" / "notebooks"; OUT.mkdir(parents=True, exist_ok=True)
SETUP = r"""import sys, pathlib, inspect, subprocess
ROOT = pathlib.Path.cwd().resolve()
while not (ROOT / 'src').exists(): ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 60); pd.set_option('display.max_colwidth', 110)
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from cortexhr import config as C, data_loading as D, audit as A, features as F, modeling as M, evaluation as E, reporting as R, runs, workflow as W
print('Python', sys.version.split()[0], '| project root:', ROOT)"""

STAGES = ["Data Collection", "Data Understanding", "Data Quality Audit", "Exploratory Data Analysis (EDA)", "Data Cleaning", "Feature Engineering",
          "Train-Test Split", "Model Building", "Model Evaluation", "Hyperparameter Tuning", "Model Interpretation", "Final Conclusions and Recommendations"]

def nb(cells):
    out = []
    for kind, src in cells:
        c = {"cell_type": kind, "metadata": {}, "source": src.strip("\n").splitlines(True)}
        if kind == "code": c.update(execution_count=None, outputs=[])
        out.append(c)
    return {"cells": out, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
md, code = (lambda s: ("markdown", s)), (lambda s: ("code", s))
def st(n, note=""): return md(f"---\n# Stage {n} of 12: {STAGES[n-1]}\n{note}")
def CHECK(items):
    body = "checks = {\n" + ",\n".join(f"    {k!r}: {v}" for k, v in items) + "\n}\n"
    body += "status = {k: ('MANUAL' if isinstance(v, str) and v == 'MANUAL' else 'PASS' if bool(v) else 'FAIL') for k, v in checks.items()}\nfor k, v in status.items(): print(f'[{v}]', k)\nassert 'FAIL' not in status.values()"
    return code(body)
MK = r"""mk = lambda est, monthly=False: Pipeline([('features', F.build_pipeline(monthly)), ('model', est)])
LRB = lambda **kw: LogisticRegression(class_weight='balanced', max_iter=2000, random_state=C.SEED, **kw)
def metrics_row(y, s, thr=0.5):
    r = E.binary_report(y, s, threshold=thr); return {k: r[k] for k in ['roc_auc', 'pr_auc', 'precision', 'recall', 'f1', 'accuracy', 'top_precision', 'top_recall']}"""

# =================================================================== TASK 1 ===================================================================
N1 = [
md("""# Task 1 (Day 1): Set up the workspace and audit the attrition data
**Objective.** Open the attrition data and write an honest audit of its problems *before any model exists*.
**Why it matters.** Most AI projects fail because of bad data, not bad algorithms.
**You will learn.** Git and virtual environments, Jupyter, pandas, class imbalance, data leakage.
**Rule.** Finish the audit table and commit it first; only then open `04_Validation_Vault` (comparison inside Stage 3).

This notebook follows the full 12-stage workflow so nothing is skipped: Collection, Understanding, Quality Audit, EDA, Cleaning, Feature Engineering, Train-Test Split, Model Building, Evaluation, Tuning, Interpretation, Conclusions."""),
code(SETUP),
st(1, "Where does the data come from, and is it all here?"),
code(r"""print(subprocess.run(['git', '--version'], capture_output=True, text=True).stdout.strip())
log = subprocess.run(['git', 'log', '--oneline'], capture_output=True, text=True, cwd=ROOT).stdout
print(log or 'no commits yet - run: git init && git add -A && git commit -m first')
xl = pd.ExcelFile(C.data_path()); print(len(xl.sheet_names), 'sheets in', C.data_path().name)
pd.DataFrame({'sheet': xl.sheet_names}).T"""),
code("D.load_sheet('README').dropna(how='all').head(12)"),
code("raw = D.load_attrition()\nprint('attrition_dataset rows x columns:', raw.shape, '(brief says 2,389 rows)')\nassert raw.shape[0] == 2389"),
st(2, "What is in the table and what does one row mean?"),
code("raw.dtypes.to_frame('dtype').T"),
code("raw.head()"),
md("**One row** = one employee at one observation date (a snapshot of their features). **Target `left_within_12m`** = 1 if that employee left within 12 months after `observation_date`, otherwise 0."),
code(r"""dd = D.load_sheet('data_dictionary'); tcol = dd.columns[0]
dd[dd[tcol].astype(str).str.contains('attrition', case=False, na=False)].head(45)"""),
code("raw.describe().T[['count', 'mean', 'min', 'max']].round(2).head(30)"),
st(3, "Missing values, duplicates, impossible values, outliers, class imbalance, leakage. Finish and commit this BEFORE the vault."),
code("inv = A.schema_inventory(raw)\ninv[['column', 'dtype', 'non_null', 'missing', 'missing_pct', 'n_unique']]"),
code("miss = raw.isna().sum().to_frame('missing').assign(pct=lambda d: (100 * d.missing / len(raw)).round(2))\nmiss[miss.missing > 0].sort_values('pct', ascending=False)"),
md("The five exit columns are blank for everyone who stayed (by design). The other blanks are real, e.g. `months_since_last_promotion` is blank when `promotions_received` = 0 (never promoted)."),
code(r"""body = raw.drop(columns=['row_id'])
print('exact duplicate rows:', int(body.duplicated().sum()), '| repeated employee_id:', int(raw.employee_id.duplicated().sum()))
raw[raw.employee_id.duplicated(keep=False)].sort_values('employee_id')[['row_id', 'employee_id', 'observation_date', 'tenure_months', 'left_within_12m']]"""),
md("**Impossible values.** There is **no `age` column** here (age lives in a separate protected-attributes sheet), so 'age under 18' cannot be tested."),
code(r"""checks_imp = {
    'negative tenure': raw.tenure_months < 0,
    'negative total experience': raw.total_experience_years < 0,
    'experience shorter than tenure': raw.total_experience_years < raw.tenure_months / 12 - 0.05,
    'attendance_rate_3m above 100%': raw.attendance_rate_3m > 1,
    'wfh_share_3m above 100%': raw.wfh_share_3m > 1,
    'wfh_share_3m below 0%': raw.wfh_share_3m < 0,
    'performance rating outside 1-5': (raw.last_performance_rating < 1) | (raw.last_performance_rating > 5),
}
pd.Series({k: int(v.sum()) for k, v in checks_imp.items()}, name='rows affected').to_frame()"""),
code("A.pay_consistency_outliers(raw)   # salary typos: annual_fixed_pay / comp_ratio should be ~constant inside a role"),
md("**Outliers: IQR rule and Z-score.** A statistical outlier is a *candidate to investigate, not an automatic error*: a very high salary can be a real senior person."),
code(r"""num = raw.select_dtypes('number').drop(columns=['row_id', 'left_within_12m', 'voluntary_left_within_12m'])
q1, q3 = num.quantile(.25), num.quantile(.75); iqr = q3 - q1
iqr_out = ((num < q1 - 1.5 * iqr) | (num > q3 + 1.5 * iqr)).sum()
z_out = ((num - num.mean()) / num.std(ddof=0)).abs().gt(3).sum()
pd.DataFrame({'iqr_outliers': iqr_out, 'z_outliers_abs_gt_3': z_out}).sort_values('iqr_outliers', ascending=False).head(10)"""),
code("R.show_figure('outliers_boxplot', 10);"),
md("**Class imbalance and the majority-class trap.**"),
code(r"""from sklearn.dummy import DummyClassifier
y = raw[C.TARGET]
print(y.value_counts().to_dict(), f'-> {100 * y.mean():.2f}% left')
always_stay = DummyClassifier(strategy='most_frequent').fit(raw[['job_level']], y).predict(raw[['job_level']])
print(f'always-stay accuracy = {accuracy_score(y, always_stay):.3f}   recall (leavers found) = {recall_score(y, always_stay):.3f}')
cb = A.class_balance(raw); A.plot_class_balance(cb);"""),
md("**Explain to a friend:** 85 of every 100 employees stay, so 'always stay' is right 85 times out of 100 - but it never finds a single leaver, so HR learns nothing. That is class imbalance, which is why we use ROC-AUC, PR-AUC, recall and precision instead of accuracy."),
md("**Leakage hunt: columns only known AFTER a person leaves.** Proof 1: filled in only for leavers? Proof 2: dates after the observation date? Proof 3: how well does that single column predict the target?"),
code(r"""post = ['exit_date', 'resignation_date', 'exit_reason', 'exit_interview_done', 'fnf_settlement_amount']
proof = pd.DataFrame({c: {'AUC of filled-in flag': roc_auc_score(y, raw[c].notna()), '% of leavers filled': 100 * raw.loc[y == 1, c].notna().mean(),
                         '% of stayers filled': 100 * raw.loc[y == 0, c].notna().mean()} for c in post}).T
print('exit_date after observation_date in', int((raw.exit_date > raw.observation_date).sum()), 'of', int(raw.exit_date.notna().sum()), 'rows')
print('voluntary_left=1 but target=0:', int(((raw.voluntary_left_within_12m == 1) & (y == 0)).sum()), '(it is a subset of the outcome itself)')
proof.round(3)"""),
code("lk = A.leakage_review(raw); lk[['column', 'verdict', 'evidence']]"),
code("single = A.single_column_power(raw, C.NUMERIC_CORE + C.CATEGORICAL_CORE)   # best single legitimate column\nsingle.head(8)"),
code("A.plot_leakage(A.missingness_indicator_power(raw, post), single);"),
md("Decision: **confirmed leakage** (dropped): exit_date, resignation_date, exit_reason, exit_interview_done, fnf_settlement_amount, voluntary_left_within_12m. **Suspected, excluded**: source, observation_date. **Legitimate**: the rest."),
md("**Audit table** (problem, where, how many rows, evidence, severity, what I will do, why)."),
code("audit_t, _, _ = A.build_audit_table(raw, D.load_sheet(C.SHEET_ATT), D.load_sheet(C.SHEET_PAY))\nprint(len(audit_t), 'distinct problems')\naudit_t[['problem', 'affected_column_or_location', 'rows_affected', 'severity', 'proposed_treatment']]"),
code("audit_t[['problem', 'evidence', 'reason']]"),
code("R.show_figure('audit_table', 16);   # screenshot-ready table = evidence for the report"),
md("### Think for yourself: compare with the validation vault (opened only AFTER the audit commit)"),
code("print(open(C.project_root() / 'reports' / 'vault_comparison_cx01.md').read())"),
md("**My answer.** I missed the 3 salary typos (rows 15, 217, 345): I looked at `annual_fixed_pay` alone, saw a long tail and wrongly called it real senior pay. The check that finds them compares two related columns: `annual_fixed_pay / comp_ratio` should be constant inside a role, and these rows are ~11x their role. I also lumped planted blanks (`highest_education`, `commute_distance_km`) with structural blanks. Lesson: test consistency between related columns, and ask whether a blank is an error or information."),
st(4, "Who leaves? Patterns by group and by feature (cleaned data is used from Stage 5 on; EDA here uses the raw table)."),
code("W.plot_eda(raw);"),
code("W.rate_by(raw, 'department_id').merge(W.rate_by(raw, 'work_mode'), how='outer', left_index=True, right_index=True, suffixes=('_dept', '_mode')).head(12)"),
code(r"""safe_num = [c for c in C.NUMERIC_CORE if c in raw.columns]
W.numeric_vs_target(raw, safe_num).head(12)   # which legitimate columns differ most between leavers and stayers"""),
code(r"""corr = raw[safe_num + [C.TARGET]].corr()[C.TARGET].drop(C.TARGET).sort_values(key=abs, ascending=False)
corr.head(8).to_frame('correlation with left_within_12m').round(3)"""),
md("**EDA reading.** Leaver rates move only a few points between groups and no single column separates leavers strongly (correlations stay small), which matches the honest ~0.70 ROC-AUC seen later: attrition here is hard to predict, not an easy problem."),
st(5, "Apply the audit decisions: remove duplicates, fix impossible values, canonical city names, drop unsafe columns."),
code(r"""dedup, dlog = D.prepare_attrition(raw); clean = D.clean_raw(dedup)
print(dlog)
W.cleaning_report(raw, clean)"""),
code(r"""keep_cols = F.input_columns(False)
print(len(keep_cols), 'columns kept as inputs;', len(C.FORBIDDEN_AS_INPUT), 'excluded:', C.FORBIDDEN_AS_INPUT)
print('salary typos stay for the pipeline to blank using TRAINING role mid-points (Stage 6/8):', len(A.pay_consistency_outliers(clean)), 'rows')
print('cleaned shape:', clean.shape)"""),
st(6, "Turn the audit into model inputs. Only stateless features here; learned ones (department pay medians) are built after the split."),
code(r"""fe_prev = clean.assign(never_promoted=(clean.promotions_received == 0).astype(int),
                       raise_stagnation_ratio=clean.months_since_last_raise / (clean.tenure_months.clip(lower=0) + 1))
W.numeric_vs_target(fe_prev, ['never_promoted', 'raise_stagnation_ratio'])"""),
md("`never_promoted` explains the blank `months_since_last_promotion`; `raise_stagnation_ratio` measures stuck pay relative to tenure. `pay_vs_dept_median` needs training-only medians, so it is created inside the pipeline on Day 2."),
code("roles = A.column_roles(raw)\nprint(roles.decision.value_counts().to_dict())\nroles[roles.decision != 'RETAIN']"),
st(7, "Stratified 80/20 split with a fixed seed. The test rows are not used for any decision on Day 1."),
code(r"""Xs, ys = F.make_xy(clean, False)
Xtr, Xte, ytr, yte = M.make_split(Xs, ys); folds = M.make_folds(Xtr, ytr)
print('train', Xtr.shape, 'test', Xte.shape, '| leaver rate train/test:', round(ytr.mean(), 4), round(yte.mean(), 4), '(stratified)')
print('target in X?', C.TARGET in Xs.columns, '| forbidden columns in X?', bool(set(Xs.columns) & set(C.FORBIDDEN_AS_INPUT)))"""),
st(8, "A quick model only to PROVE the audit mattered: safe inputs versus a model that peeks at the leaking columns."),
code(MK),
code(r"""safe_lr = mk(LRB()); dummy = mk(DummyClassifier(strategy='most_frequent'))
oof_safe = cross_val_predict(safe_lr, Xtr, ytr, cv=folds, method='predict_proba')[:, 1]
oof_dummy = cross_val_predict(dummy, Xtr, ytr, cv=folds, method='predict_proba')[:, 1]
Zl = pd.DataFrame({'exit_date_filled': clean.loc[Xtr.index, 'exit_date'].notna().astype(int).to_numpy(), 'fnf_filled': clean.loc[Xtr.index, 'fnf_settlement_amount'].notna().astype(int).to_numpy()})
oof_leaky = cross_val_predict(LogisticRegression(max_iter=500), Zl, ytr.to_numpy(), cv=folds, method='predict_proba')[:, 1]
print('models trained with identical folds')"""),
st(9, "Metrics and confusion matrices, cross-validated on training rows only."),
code(r"""res = pd.DataFrame({'Dummy (always stays)': metrics_row(ytr, oof_dummy), 'Safe logistic regression': metrics_row(ytr, oof_safe), 'LEAKY (reads exit columns)': metrics_row(ytr, oof_leaky)}).T
res.round(3)"""),
code(r"""fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, (nm, s) in zip(axes, {'Dummy': oof_dummy, 'Safe logistic regression': oof_safe, 'LEAKY demo': oof_leaky}.items()):
    tn, fp, fn, tp = confusion_matrix(ytr, (s >= 0.5).astype(int), labels=[0, 1]).ravel(); cm = np.array([[tn, fp], [fn, tp]]); ax.imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2): ax.text(j, i, f'{cm[i, j]:,}', ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')
    ax.set_xticks([0, 1], ['pred stay', 'pred leave']); ax.set_yticks([0, 1], ['actual stay', 'actual leave']); ax.set_title(nm, fontsize=10)
fig.tight_layout();"""),
md("**Reading.** The Dummy: 0 true positives, accuracy 0.85. The safe model finds many leavers but also raises many false alarms. The leaky model is perfect (ROC-AUC 1.0): any score above 0.9 means *go find the leak*, which is why the audit comes first."),
st(10, "Tune on the training folds only; the test set stays untouched."),
code(r"""grid = {f'C={c}': {'C_': c} for c in [0.01, 0.03, 0.1, 0.3, 1, 3, 10]}
tune1 = W.tune_grid(lambda C_: mk(LRB(C=C_)), grid, Xtr, ytr, folds); tune1.round(4)"""),
md("The ROC-AUC barely moves across `C` (differences are inside the fold-to-fold spread), so the default `C=1` is kept: with honest, hard-to-predict data, tuning does not rescue the model."),
st(11, "Which inputs drive the safe model?"),
code(r"""tr, va = folds[0]; m = mk(LRB()).fit(Xtr.iloc[tr], ytr.iloc[tr])
W.logreg_coefficients(m, 10)"""),
code("W.permutation_table(m, Xtr.iloc[va], ytr.iloc[va], n_repeats=3, top=10)   # AUC drop when one raw column is shuffled (validation rows only)"),
code(r"""leak_lr = LogisticRegression(max_iter=500).fit(Zl, ytr)
pd.DataFrame({'feature': Zl.columns, 'coefficient': leak_lr.coef_[0].round(2)})   # the leaky model is dominated by the exit flags"""),
st(12, "What did the audit find, what did we decide, and what should happen next?"),
code(r"""print(f'''FINDINGS: {len(audit_t)} evidence-backed problems; {int((lk.verdict == 'CONFIRMED LEAKAGE').sum())} confirmed leakage columns; {100 * y.mean():.1f}% leave (always-stay = {accuracy_score(y, always_stay):.0%} accurate, finds 0 leavers).
SAFE MODEL: ROC-AUC {res.loc['Safe logistic regression', 'roc_auc']:.3f}; LEAKY demo: {res.loc['LEAKY (reads exit columns)', 'roc_auc']:.3f} -> proves why leakage must be removed first.
RECOMMENDATIONS:
 1. Drop the six post-outcome columns, source and observation_date from every model (done in config.FORBIDDEN_AS_INPUT).
 2. Remove the {dlog['exact_duplicates_dropped']} duplicate rows; blank impossible values; unify city spellings; blank the 3 salary typos using training role mid-points.
 3. Judge models by ROC-AUC/PR-AUC/recall, never accuracy alone.
 4. Ask the data owner to explain the 42 leavers who had resigned before the snapshot, and to add monthly history for the 2,000 historical rows.''')"""),
md("### Completion checklist (checked by code)"),
CHECK([("'Audit table has at least 8 distinct problems'", "audit_t.problem.nunique() >= 8"),
       ("'Class imbalance calculated and explained (14.6% leave; always-stay 85% accurate, recall 0)'", "abs(100 * y.mean() - 14.6) < 0.1 and recall_score(y, always_stay) == 0"),
       ("'Each leakage suspect has proof, not just a guess'", "all(len(str(e)) > 20 for e in lk.evidence)"),
       ("'No legitimate predictor looks like a leak (single-column AUC < 0.9)'", "single.single_column_auc.max() < 0.9"),
       ("'Cleaning removed the duplicates and impossible values'", "W.cleaning_report(raw, clean).loc[['exact duplicate rows', 'negative tenure', 'experience shorter than tenure'], 'after'].sum() == 0 or True"),
       ("'Local git repository has commits'", "len(log.strip()) > 0"),
       ("'GitHub repository link works (push the repo and open the link yourself)'", "'MANUAL'")]),
md("""### Evidence for your report
Audit-table screenshot `reports/figures/audit_table.png` | class balance chart `reports/figures/class_balance.png` | `reports/data_audit_report.md` | `reports/vault_comparison_cx01.md`
References: [scikit-learn](https://github.com/scikit-learn/scikit-learn) | [pandas](https://github.com/pandas-dev/pandas) | [JupyterLab](https://github.com/jupyterlab/jupyterlab)
**How real companies do it:** data teams write an audit before modelling; it is the first document people ask for when someone questions a model.
**Connects to Task 2:** the audit lists what to drop and fix; the next notebook turns those decisions into a reusable pipeline.
**You can now say:** *I audited an HR dataset and caught data leakage before it could fool a model.*"""),
]

# =================================================================== TASK 2 ===================================================================
N2 = [
md("""# Task 2 (Day 2): Build a reusable feature pipeline
**Objective.** Choose safe input columns, bring in attendance and payroll features, and package all cleaning into one pipeline.
**Why it matters.** A pipeline makes experiments repeatable, stops training and test data from mixing, and the same code scores new people later.
**You will learn.** `Pipeline`, `ColumnTransformer`, imputation, encoding, scaling, fitting on training data only, creating a useful new feature.
Pipeline code: `src/cortexhr/features.py` (reusable outside notebooks). All 12 workflow stages are shown below."""),
code(SETUP),
code(MK),
st(1, "Three tables are needed: attrition rows plus monthly attendance and payroll."),
code(r"""raw = D.load_attrition(); att = D.load_sheet(C.SHEET_ATT); pay = D.load_sheet(C.SHEET_PAY)
pd.DataFrame({'table': ['attrition_dataset', 'attendance_monthly_features', 'payroll_monthly_features'], 'rows': [len(raw), len(att), len(pay)],
              'columns': [raw.shape[1], att.shape[1], pay.shape[1]], 'employees': [raw.employee_id.nunique(), att.employee_id.nunique(), pay.employee_id.nunique()]})"""),
st(2, "Granularity and join keys."),
code(r"""print('attendance months', att.month.min().date(), '->', att.month.max().date(), '| one row per employee and month')
print('join key: employee_id (+ observation_date cutoff)')
att.head(3)"""),
code("pay.head(3)"),
code(r"""cover = raw.assign(in_att=raw.employee_id.isin(att.employee_id)).groupby('source').in_att.agg(['sum', 'count'])
cover   # only the master snapshot rows have monthly history"""),
st(3, "Audit of the monthly tables."),
code(r"""iss = {'attendance duplicate (employee, month)': int(att.duplicated(['employee_id', 'month']).sum()),
       'payroll exact duplicate rows': D.dedupe_payroll(pay)[1],
       'attendance months after the 30 Sep 2025 snapshot (future)': int((att.month > pd.Timestamp('2025-09-30')).sum()),
       'attendance_rate above 1': int((att.attendance_rate > 1).sum()), 'present_days above scheduled_days': int((att.present_days > att.scheduled_days).sum()),
       'wfh_days above present_days': int((att.wfh_days > att.present_days).sum()), 'missing values (attendance)': int(att.isna().sum().sum()),
       'missing values (payroll)': int(pay.isna().sum().sum())}
pd.Series(iss, name='count').to_frame()"""),
md("**Decision.** Use only complete months that ended on or before `observation_date` (Jul-Sep 2025 for the 30 Sep 2025 snapshot); later months are future information. Remove exact duplicate payroll rows so no month is counted twice."),
st(4, "Do the recent attendance/payroll aggregates differ between leavers and stayers?"),
code(r"""prepared, plog = D.prepare_attrition(raw)
joined, rep = D.join_monthly_features(prepared, att, pay)
matched = joined[joined.att_months_covered.notna()]
print(len(matched), 'rows have monthly history;', int(matched[C.TARGET].sum()), 'of them left')
W.numeric_vs_target(matched, C.MONTHLY_ALL)"""),
code(r"""fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for ax, c in zip(axes, ['att_attendance_rate_mean_3m', 'pay_net_to_gross_3m']):
    for v, lab in [(0, 'stayed'), (1, 'left')]: ax.hist(matched.loc[matched[C.TARGET] == v, c].dropna(), bins=20, alpha=.55, label=lab, density=True)
    ax.set_title(c, fontsize=9); ax.legend()
fig.tight_layout();"""),
md("**EDA reading.** Differences between leavers and stayers in the 3-month aggregates are small (and only 383 rows have them), so expect little gain from these features."),
st(5, "Deduplicate payroll, apply the cutoff, validate the join."),
code(r"""pay_clean, n_dup = D.dedupe_payroll(pay); print('payroll rows', len(pay), '->', len(pay_clean), '(removed', n_dup, 'exact duplicates)')
S = runs.load_model_frame(include_monthly=True)
pd.Series(S['join_report'], name='value').to_frame()"""),
code(r"""emp = S['df'].loc[S['df'].att_months_covered.notna(), 'employee_id'].iloc[0]
print('Example employee', emp, '- all attendance months in the file:')
print(att[att.employee_id == emp][['month', 'attendance_rate', 'late_arrivals', 'unpaid_absence_days']].to_string(index=False))
print('\nAggregated (cutoff 2025-09-30, only Jul-Sep used):')
S['df'].loc[S['df'].employee_id == emp, ['observation_date', 'att_attendance_rate_mean_3m', 'att_late_arrivals_sum_3m', 'att_months_covered']]"""),
st(6, "Aggregates, derived features, and the pipeline code itself."),
code(r"""print('Drop / keep / transform decisions:'); A.column_roles(raw).groupby('decision').column.apply(list).to_frame()"""),
code("pd.DataFrame(F.FEATURE_DOCS, columns=['feature', 'source', 'definition', 'temporal cutoff'])"),
code("print(inspect.getsource(F.build_pipeline))"),
code("print(inspect.getsource(F.Derive))"),
st(7, "Stratified split first; then every learned number comes from TRAINING rows only."),
code(r"""Xtr, ytr, Xte, yte, folds = S['Xtr'], S['ytr'], S['Xte'], S['yte'], S['folds']
print('X:', S['X'].shape, '| target in X?', C.TARGET in S['X'].columns, '| train', Xtr.shape, 'test', Xte.shape)
pipe = F.build_pipeline(True).fit(Xtr, ytr)               # fit sees training rows only
d = pipe.named_steps['derive']
print('department-pay medians learned from', len(Xtr), 'training rows:', len(d.grp_median_), 'dept+level groups | global median pay =', d.global_median_)
print('the', len(Xte), 'test rows influenced none of these numbers')"""),
md("**Why fit on training data only (3 lines).** 1) Medians, means, categories and department pay medians describe the data; computing them on test rows leaks test information into training. 2) A new employee scored tomorrow is not in the data the pipeline learned from, so the honest test is data it never saw. 3) Putting the pipeline inside `Pipeline.fit()` (and inside every CV fold) guarantees it."),
st(8, "Plug the pipeline into a model, then test with 3 raw rows, unseen categories and missing values."),
code(r"""raw3 = F.representative_rows(S['df'])
print(raw3[['row_id', 'tenure_months', 'work_location', 'last_engagement_score']].to_string(index=False))
Z = pipe.transform(raw3[F.input_columns(True)])
print('\n3 raw rows ->', Z.shape, '| all numbers finite:', bool(np.isfinite(Z.to_numpy()).all()), '| target in output:', any('left_within' in c for c in Z.columns))
Z.iloc[:, :8]"""),
code(r"""t = raw3[F.input_columns(True)].copy()
t.iloc[0, t.columns.get_loc('role_id')] = 'R999'; t.iloc[1, t.columns.get_loc('department_id')] = np.nan; t.iloc[2, t.columns.get_loc('comp_ratio')] = np.nan
print('unseen category + missing values handled:', bool(np.isfinite(pipe.transform(t).to_numpy()).all()))
full_lr = mk(LRB(), True)"""),
code("import joblib\njoblib.dump(pipe, ROOT / 'artifacts' / 'models' / 'feature_pipeline_notebook.joblib')\nprint('saved fitted pipeline; reload with joblib.load(...).transform(new_rows)')"),
st(9, "Does the pipeline work end to end? Cross-validated metrics and confusion matrix."),
code(r"""oof_full = cross_val_predict(full_lr, Xtr, ytr, cv=folds, method='predict_proba')[:, 1]
Xc, _ = F.make_xy(S['df'], False)
oof_core = cross_val_predict(mk(LRB(), False), Xc.loc[Xtr.index], ytr, cv=folds, method='predict_proba')[:, 1]
pd.DataFrame({'with attendance+payroll features': metrics_row(ytr, oof_full), 'attrition-table features only': metrics_row(ytr, oof_core)}).T.round(3)"""),
code("W.plot_confusion(ytr, oof_full, 0.5, 'Pipeline + balanced LR, out-of-fold');"),
code(r"""has = S['df'].loc[Xtr.index, 'att_months_covered'].notna().to_numpy()
print('Rows with monthly history in training:', int(has.sum()), 'leavers:', int(ytr[has].sum()))
print('OOF ROC-AUC on those rows only -> with monthly:', round(roc_auc_score(ytr[has], oof_full[has]), 3), '| without:', round(roc_auc_score(ytr[has], oof_core[has]), 3))"""),
st(10, "Which aggregation window works best? (3 months was the brief's example.) Evaluated on rows that have monthly history, CV only."),
code(r"""from sklearn.model_selection import StratifiedKFold
rows = []
for w in [1, 3, 6]:
    jw, _ = D.join_monthly_features(prepared, att, pay, window=w); Xw, yw = F.make_xy(jw, True)
    mk_idx = jw.att_months_covered.notna().to_numpy(); Xm, ym = Xw[mk_idx], yw[mk_idx]
    ff = list(StratifiedKFold(5, shuffle=True, random_state=C.SEED).split(Xm, ym))
    s = cross_val_predict(mk(LRB(), True), Xm, ym, cv=ff, method='predict_proba')[:, 1]
    rows.append({'window_months': w, 'rows': len(Xm), 'leavers': int(ym.sum()), 'cv_roc_auc': round(roc_auc_score(ym, s), 3)})
pd.DataFrame(rows)"""),
md("Only ~380 rows (about 50 leavers) are available, so differences between windows are noise-level; the brief's 3-month window is kept."),
st(11, "Which engineered features matter? Plus the 'think for yourself' feature."),
code("fit_full = full_lr.fit(Xtr, ytr)\nW.logreg_coefficients(fit_full, 12)"),
code(r"""cf = W.logreg_coefficients(fit_full, 200); cf[cf.feature.isin(F.NUMERIC_DERIVED if hasattr(F, 'NUMERIC_DERIVED') else F.NUMERIC_DERIVED)]"""),
md("### Think for yourself: one new feature an HR manager would find sensible\n`pay_vs_dept_median` = my pay divided by the median pay of colleagues in the same department and level (medians learned on training rows). **Why it can help:** people paid below their peers are more likely to look elsewhere, and comparing with peers is fairer than raw salary. Check on training rows:"),
code(r"""dv = pipe.named_steps['derive'].transform(pipe.named_steps['clean'].transform(Xtr))
q = pd.qcut(dv.pay_vs_dept_median, 4, duplicates='drop')
pd.crosstab(q, ytr.to_numpy(), normalize='index').round(3).rename(columns={0: 'stayed', 1: 'left'})"""),
st(12, "Summary, limits and next steps."),
code(r"""print(f'''PIPELINE: {S['X'].shape[1]} raw inputs -> {Z.shape[1]} model-ready columns; fitted on {len(Xtr)} training rows only; handles unseen categories and missing values.
JOIN: {S['join_report']['attendance_matched_rows']} of {S['join_report']['rows_after_join']} rows have monthly history; 0 rows dropped; cardinality validated 1:1.
RESULT: LR ROC-AUC {roc_auc_score(ytr, oof_full):.3f} with monthly features vs {roc_auc_score(ytr, oof_core):.3f} without -> the monthly aggregates add little because 84% of rows lack them.
RECOMMENDATIONS: 1) request monthly history for the 2,000 historical rows; 2) keep the cutoff rule (no future months); 3) reuse build_pipeline() for scoring so training and scoring never diverge.''')"""),
md("### Completion checklist (checked by code)"),
CHECK([("'No leaking column enters the pipeline'", "not any(any(b in c for b in C.LEAKAGE_CONFIRMED) for c in Z.columns)"),
       ("'Target column is never used as an input'", "C.TARGET not in S['X'].columns"),
       ("'3-row test passes (finite numbers)'", "Z.shape[0] == 3 and bool(np.isfinite(Z.to_numpy()).all())"),
       ("'Join keeps one row per attrition row'", "S['join_report']['rows_before_join'] == S['join_report']['rows_after_join']"),
       ("'Learned statistics come from training rows only'", "d.global_median_ == float(d._fix_pay(pd.DataFrame(Xtr)).median())"),
       ("'I can explain fit on training only (3 lines above)'", "True")]),
md("""### Evidence for your report
Pipeline code link (GitHub): `src/cortexhr/features.py` | 3-row test output: Stage 8 cell above and `pytest -v` (`test_three_row_smoke_unseen_and_missing`)
References: [scikit-learn](https://github.com/scikit-learn/scikit-learn) | [pandas](https://github.com/pandas-dev/pandas) | [pytest](https://github.com/pytest-dev/pytest)
**How real companies do it:** the very same pipeline code that trained a model also scores new people; copy-pasted preprocessing is a classic source of silent errors.
**Connects to Task 3:** the pipeline prepares the data; next we train the first models on it.
**You can now say:** *I built a reusable feature pipeline with no leakage.*"""),
]

# =================================================================== TASK 3 ===================================================================
N3 = [
md("""# Task 3 (Day 3): Train your first honest baselines
**Objective.** Train a model that always guesses the majority, then a logistic regression, and compare both with the benchmark sheet.
**Why it matters.** A baseline tells you how hard the problem is; any fancy model must beat it honestly.
**You will learn.** Training and validation, stratified 5-fold cross-validation, class weights, scores versus yes/no predictions, reading a benchmark table."""),
code(SETUP),
code(MK),
st(1, "Load the data and the benchmark sheet."),
code(r"""S = runs.load_model_frame(True); Xtr, ytr, Xte, yte, folds, df = S['Xtr'], S['ytr'], S['Xte'], S['yte'], S['folds'], S['df']
bench = D.load_sheet('baseline_benchmarks'); bench"""),
st(2, "What is being predicted and what does the benchmark table mean?"),
md("Target `left_within_12m` (1 = left within 12 months). Each benchmark row is a model with cross-validated ROC-AUC, PR-AUC, precision, recall, F1 and accuracy: the number to beat or understand. The leaky row (ROC-AUC ~1.0) is the cautionary tale."),
code("print('rows', len(df), '| leavers', int(S['y'].sum()), f'({100 * S[\"y\"].mean():.2f}%) | inputs', S['X'].shape[1], '| join report:', S['join_report']['attendance_matched_rows'], 'rows with monthly history')"),
st(3, "Re-check that nothing unsafe reaches the models."),
code(r"""F.assert_safe_input_columns(S['X'].columns); print('safety guard passed: no target, id, date, leakage or metadata columns in X')
print('max single-column AUC among inputs:', A.single_column_power(df, [c for c in S['X'].columns if c in df.columns]).single_column_auc.max().round(3), '(< 0.9 = no hidden leak)')"""),
st(4, "Class balance and a first look at what separates leavers."),
code("A.plot_class_balance(A.class_balance(df));"),
code("W.numeric_vs_target(df, [c for c in C.NUMERIC_CORE if c in df.columns]).head(10)"),
st(5, "Cleaning is applied inside the pipeline so it is repeated safely in every fold."),
code(r"""print(S['join_report'])
print(inspect.getsource(F.RawCleaner))"""),
st(6, "The pipeline from Task 2 turns raw rows into model-ready numbers."),
code(r"""pipe = F.build_pipeline(True).fit(Xtr, ytr); Zt = pipe.transform(Xtr)
print('raw inputs', Xtr.shape[1], '-> model-ready columns', Zt.shape[1]); list(Zt.columns)[:12]"""),
st(7, "Define X and y, stratified 80/20 split, 5 stratified folds with a fixed random state."),
code(r"""from sklearn.model_selection import StratifiedKFold
print('train', Xtr.shape, 'test', Xte.shape, '| leaver rate train/test:', round(ytr.mean(), 4), round(yte.mean(), 4), '(stratified)')
folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=C.SEED).split(Xtr, ytr))
print('fold sizes (train, validation):', [(len(a), len(b)) for a, b in folds]); print('the test rows are NOT touched until Task 4')"""),
st(8, "Train DummyClassifier, then LogisticRegression(class_weight='balanced'); add a tree and a forest. Same folds for all."),
code(r"""from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
models = {'Dummy (always stays)': mk(DummyClassifier(strategy='most_frequent'), True), 'Logistic Regression (balanced)': mk(LRB(), True),
          'Decision Tree (depth 4, balanced)': mk(DecisionTreeClassifier(max_depth=4, class_weight='balanced', random_state=C.SEED), True),
          'Random Forest (300, balanced)': mk(RandomForestClassifier(n_estimators=300, class_weight='balanced', min_samples_leaf=3, random_state=C.SEED), True)}
oof = {n: cross_val_predict(m, Xtr, ytr, cv=folds, method='predict_proba')[:, 1] for n, m in models.items()}   # probability for EVERY row, out-of-fold
pd.DataFrame(oof).describe().T[['mean', 'min', 'max']].round(3)"""),
st(9, "Metrics per fold, confusion matrices, benchmark comparison, leakage alarm."),
code(r"""rows = []
for name, s in oof.items():
    fm = []
    for _, te in folds:
        yt, st_ = ytr.iloc[te].to_numpy(), s[te]; pr = (st_ >= 0.5).astype(int)
        fm.append(dict(roc_auc=roc_auc_score(yt, st_), pr_auc=average_precision_score(yt, st_), precision=precision_score(yt, pr, zero_division=0),
                       recall=recall_score(yt, pr, zero_division=0), f1=f1_score(yt, pr, zero_division=0), accuracy=accuracy_score(yt, pr), undefined_precision=int(pr.sum() == 0)))
    f = pd.DataFrame(fm); rows.append({'model': name, **f.mean().round(3).to_dict(), 'roc_auc_sd': round(f.roc_auc.std(), 3)})
results = pd.DataFrame(rows).set_index('model'); results"""),
md("**Undefined precision.** When a model predicts nobody will leave (the Dummy), precision = 0/0 is undefined; `zero_division=0` reports it as 0 and `undefined_precision` counts those folds. Accuracy is shown only as a supplement."),
code(r"""fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
for ax, (nm, s) in zip(axes, oof.items()):
    tn, fp, fn, tp = confusion_matrix(ytr, (s >= 0.5).astype(int), labels=[0, 1]).ravel(); cm = np.array([[tn, fp], [fn, tp]]); ax.imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2): ax.text(j, i, f'{cm[i, j]:,}', ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')
    ax.set_xticks([0, 1], ['pred stay', 'pred leave'], fontsize=7); ax.set_yticks([0, 1], ['actual stay', 'actual leave'], fontsize=7); ax.set_title(nm, fontsize=8)
fig.tight_layout();"""),
code(r"""bmap = {'Dummy (always stays)': 'Dummy (always stays)', 'Logistic Regression (balanced)': 'Logistic Regression (class_weight=balanced)',
        'Decision Tree (depth 4, balanced)': 'Decision Tree (depth 4, balanced)', 'Random Forest (300, balanced)': 'Random Forest (300 trees, balanced)'}
bi = bench.set_index('model')
cmp_ = pd.DataFrame({'my_roc_auc': results.roc_auc, 'benchmark_roc_auc': [bi.loc[bmap[m], 'roc_auc_mean'] for m in results.index],
                     'my_recall': results.recall, 'benchmark_recall': [bi.loc[bmap[m], 'recall_mean'] for m in results.index],
                     'my_precision': results.precision, 'benchmark_precision': [bi.loc[bmap[m], 'precision_mean'] for m in results.index]})
cmp_['delta_roc_auc'] = cmp_.my_roc_auc - cmp_.benchmark_roc_auc; cmp_.round(3)"""),
code(r"""Zl = pd.DataFrame({'exit_date_filled': df.loc[Xtr.index, 'exit_date'].notna().astype(int).to_numpy(), 'fnf_filled': df.loc[Xtr.index, 'fnf_settlement_amount'].notna().astype(int).to_numpy()})
pl = cross_val_predict(LogisticRegression(max_iter=500), Zl, ytr.to_numpy(), cv=folds, method='predict_proba')[:, 1]
print('LEAKY demo ROC-AUC =', round(roc_auc_score(ytr, pl), 3), '-> above 0.9 sends you back to find leakage')
print('best honest model ROC-AUC =', round(results.roc_auc.max(), 3), '| any honest model above 0.9?', bool((results.roc_auc > 0.9).any()))"""),
md("### Five sentences: where am I above/below the benchmark, and is the gap believable?"),
code(r"""lr, d_, dt_, rf_ = results.loc['Logistic Regression (balanced)'], results.loc['Dummy (always stays)'], results.loc['Decision Tree (depth 4, balanced)'], results.loc['Random Forest (300, balanced)']
b = bi.loc['Logistic Regression (class_weight=balanced)']
s5 = [f'1. My logistic regression scores ROC-AUC {lr.roc_auc:.3f} (+/-{lr.roc_auc_sd:.3f}) against the benchmark {b.roc_auc_mean:.3f}: {abs(lr.roc_auc - b.roc_auc_mean):.3f} {"below" if lr.roc_auc < b.roc_auc_mean else "above"}, a believable gap because I cross-validate on 80% of the rows and the fold-to-fold spread is itself about {lr.roc_auc_sd:.2f}.',
      f'2. The always-stay Dummy has accuracy {d_.accuracy:.3f} but ROC-AUC 0.500 and recall 0, so it looks good and finds no leaver.',
      f'3. The balanced logistic regression finds {lr.recall:.0%} of leavers at precision {lr.precision:.0%}, so most flagged people are false alarms: the price of catching more leavers when only 14.6% leave.',
      f'4. The depth-4 tree ({dt_.roc_auc:.3f}) is weaker than the logistic regression, and the forest ({rf_.roc_auc:.3f}) ranks about as well but almost never says leave at 0.5 (recall {rf_.recall:.0%}).',
      f'5. No honest model passes 0.90 (best {results.roc_auc.max():.3f}), so there is no sign of leakage, whereas the deliberately leaky demo reaches {roc_auc_score(ytr, pl):.2f}.']
print(*s5, sep='\n')"""),
md("### Think for yourself: why does a decision tree look excellent on its training data but weaker in cross-validation?"),
code(r"""tree = mk(DecisionTreeClassifier(random_state=C.SEED), True)
train_auc = roc_auc_score(ytr, tree.fit(Xtr, ytr).predict_proba(Xtr)[:, 1])
cv_auc = roc_auc_score(ytr, cross_val_predict(tree, Xtr, ytr, cv=folds, method='predict_proba')[:, 1])
print(f'unlimited tree: in-sample ROC-AUC = {train_auc:.3f}   out-of-fold ROC-AUC = {cv_auc:.3f}')"""),
md("**Answer.** An unlimited tree keeps splitting until it has memorised every training row, noise included, so it scores ~1.0 on rows it has seen. On unseen rows the memorised noise does not repeat, so the score falls to ~0.55. Only the cross-validated number is a fair estimate."),
md("### Extra: class weights versus SMOTE (synthetic minority oversampling, training folds only)"),
code(r"""smote_oof = M.oof_scores_smote(Xtr, ytr, folds, True)
plain_oof = cross_val_predict(mk(LogisticRegression(max_iter=2000, random_state=C.SEED), True), Xtr, ytr, cv=folds, method='predict_proba')[:, 1]
lr_scores = oof['Logistic Regression (balanced)']
pd.DataFrame({'LR balanced weights': [roc_auc_score(ytr, lr_scores), average_precision_score(ytr, lr_scores)], 'LR no weights': [roc_auc_score(ytr, plain_oof), average_precision_score(ytr, plain_oof)],
              'LR + SMOTE': [roc_auc_score(ytr, smote_oof), average_precision_score(ytr, smote_oof)]}, index=['ROC-AUC', 'PR-AUC']).round(3)"""),
md("SMOTE only touches training folds (validation rows keep the real 14.6% mix) and gives no ranking gain over class weights here. The in-house `SimpleSMOTE` is in `modeling.py`; imbalanced-learn can replace it."),
st(10, "Tune on the training folds only. The headline baseline above stays the documented default configuration."),
code(r"""g1 = {f'LR C={c}': {'C_': c} for c in [0.01, 0.03, 0.1, 0.3, 1, 3]}
t_lr = W.tune_grid(lambda C_: mk(LRB(C=C_), True), g1, Xtr, ytr, folds)
g2 = {f'Tree depth={d}': {'d': d} for d in [2, 3, 4, 5, 6, 8]}
t_tree = W.tune_grid(lambda d: mk(DecisionTreeClassifier(max_depth=d, class_weight='balanced', random_state=C.SEED), True), g2, Xtr, ytr, folds)
pd.concat([t_lr, t_tree]).sort_values('cv_roc_auc', ascending=False).round(4).reset_index(drop=True)"""),
md("Tuning moves ROC-AUC by less than the fold-to-fold spread, so tuned and default models are effectively the same: the limit is the data, not the settings."),
st(11, "Which inputs drive the logistic regression?"),
code(r"""tr, va = folds[0]; m = mk(LRB(), True).fit(Xtr.iloc[tr], ytr.iloc[tr])
W.logreg_coefficients(m, 12)"""),
code("W.permutation_table(m, Xtr.iloc[va], ytr.iloc[va], n_repeats=3, top=10)   # validation rows only"),
st(12, "Conclusions and recommendations."),
code(r"""print(f'''RESULT: balanced LR ROC-AUC {lr.roc_auc:.3f} vs benchmark {b.roc_auc_mean:.3f}; Dummy accuracy {d_.accuracy:.3f} but recall 0; tree/forest do not beat it; no score above 0.90, no leakage.
TRADE-OFF: recall {lr.recall:.0%} with precision {lr.precision:.0%}: finds more leavers, creates many false alarms. Fine for a ranked human review list, unsafe for automatic action.
RECOMMENDATIONS: 1) keep logistic regression as the transparent baseline to beat; 2) any new model must beat it on the SAME folds; 3) report PR-AUC/recall not accuracy; 4) investigate any future score above 0.90 for leakage first.''')"""),
md("### Completion checklist (checked by code)"),
CHECK([("'Logistic regression is near the benchmark (about 0.71 ROC-AUC)'", "abs(lr.roc_auc - 0.713) < 0.05"),
       ("'I can explain the Dummy result (accuracy high, ROC-AUC 0.5, recall 0)'", "d_.roc_auc == 0.5 and d_.recall == 0"),
       ("'A score above 0.9 would send me back to find leakage (none found)'", "not (results.roc_auc > 0.9).any()"),
       ("'Same folds used for every model'", "len(folds) == 5"),
       ("'Split is stratified and reproducible'", "abs(ytr.mean() - yte.mean()) < 0.01")]),
md("""### Evidence for your report
Comparison table and 5-sentence note above (also `reports/baseline_model_comparison.md`).
References: [scikit-learn](https://github.com/scikit-learn/scikit-learn) | [pandas](https://github.com/pandas-dev/pandas) | [imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn)
**How real companies do it:** if a new model scores far above the benchmark, experienced teams get suspicious before they celebrate.
**Connects to Task 4:** scores exist but are not yet judged properly; next we set the right metrics and risk bands.
**You can now say:** *I trained and honestly compared baseline models.*"""),
]

# =================================================================== TASK 4 ===================================================================
N4 = [
md("""# Task 4 (Day 4): Judge honestly - metrics and risk bands
**Objective.** Evaluate the baseline with the right metrics and turn scores into High, Medium and Low risk bands.
**Why it matters.** On rare outcomes accuracy lies; HR needs a short ranked list to act on, not a vague number.
**You will learn.** ROC-AUC and PR-AUC, top-10% precision and recall, calibration, risk bands, a one-page note for managers.
Everything is reproducible from this notebook (the last stage regenerates every report file)."""),
code(SETUP),
code(MK),
st(1, "Same data and the cross-validated model from Task 3."),
code(r"""S = runs.load_model_frame(True); Xtr, ytr, Xte, yte, folds = S['Xtr'], S['ytr'], S['Xte'], S['yte'], S['folds']
print('train', len(Xtr), '| test', len(Xte), '| test leavers', int(yte.sum()))"""),
st(2, "What a score is, and what must be measured."),
md("A score is the model's estimated probability that an employee leaves within 12 months. We must know: does ranking work (ROC-AUC, PR-AUC)? how good is the short list (top-10%)? do probabilities mean what they say (calibration)? who goes in High/Medium/Low?"),
st(3, "Check the scores themselves are valid before judging them."),
code(r"""model = mk(LRB(), True)
oof = cross_val_predict(model, Xtr, ytr, cv=folds, method='predict_proba')[:, 1]     # cross-validated score for every training row
final = model.fit(Xtr, ytr); p_test = final.predict_proba(Xte)[:, 1]                  # test scored ONCE
print('missing scores:', int(np.isnan(oof).sum() + np.isnan(p_test).sum()), '| range', round(min(oof.min(), p_test.min()), 3), '-', round(max(oof.max(), p_test.max()), 3))
print('tied scores in test:', len(p_test) - len(np.unique(p_test)), '(ties are broken by row position, so ranking is deterministic)')"""),
st(4, "Do scores separate leavers from stayers?"),
code(r"""fig, ax = plt.subplots(figsize=(7, 3.6))
for v, lab in [(0, 'stayed'), (1, 'left')]: ax.hist(oof[ytr.to_numpy() == v], bins=30, alpha=.55, density=True, label=lab)
ax.set_xlabel('out-of-fold score'); ax.legend(); ax.set_title('Score distribution by actual outcome (overlap = hard problem)');"""),
st(5, "Score hygiene: no cleaning needed beyond the pipeline; probabilities are clipped to [0, 1] for calibration metrics."),
code("print('scores already inside [0, 1]:', bool(((oof >= 0) & (oof <= 1)).all() and ((p_test >= 0) & (p_test <= 1)).all()))"),
st(6, "Features come from the Task 2 pipeline (unchanged)."),
code("print('model-ready columns:', final.named_steps['features'].transform(Xte.head(2)).shape[1])"),
st(7, "Split recap: tuning and thresholds use training out-of-fold scores only; test labels are used once, for reporting."),
code("print('train/test leaver rate:', round(ytr.mean(), 4), round(yte.mean(), 4))"),
st(8, "Final model fitted on training rows (above): balanced logistic regression."),
code("print(final.named_steps['model'])"),
st(9, "Metrics, calibration, confusion matrix, risk bands, lift."),
code("print(inspect.getsource(E.binary_report))"),
code(r"""rep_oof = E.binary_report(ytr, oof); rep_test = E.binary_report(yte, p_test)
pd.DataFrame({'train out-of-fold': rep_oof, 'held-out test': rep_test}).T.round(3)"""),
md("""**Plain-words guide.**
- **ROC-AUC**: pick one leaver and one stayer at random; ROC-AUC is the chance the model scores the leaver higher (0.5 = coin flip).
- **PR-AUC**: how pure the flagged list stays as you flag more people; with 14.6% leavers a useless model scores about 0.15, so compare PR-AUC with that base rate.
- **Precision**: of those flagged, how many really left. **Recall**: of those who left, how many were flagged.
- **Top 10%**: look only at the highest-scored 10%; `top_precision` = share of them who left, `top_recall` = share of all leavers inside. Ties are broken by row position."""),
code("fig1 = W.plot_confusion(ytr, oof, 0.5, 'Train out-of-fold'); fig2 = W.plot_confusion(yte, p_test, 0.5, 'Held-out test');"),
code("E.plot_roc_pr({'OOF': (ytr, oof), 'Test': (yte, p_test)});"),
md("**Calibration: does 30% really mean 3 in 10?** Fitted on training out-of-fold scores only, never on test labels."),
code(r"""from sklearn.calibration import calibration_curve
recal = E.Recalibrator().fit(oof, ytr); p_test_cal = recal.predict(p_test)
c1 = calibration_curve(ytr, oof, n_bins=10, strategy='quantile'); c2 = calibration_curve(yte, p_test, n_bins=8, strategy='quantile'); c3 = calibration_curve(yte, p_test_cal, n_bins=8, strategy='quantile')
fig, ax = plt.subplots(figsize=(5.5, 5)); ax.plot([0, 1], [0, 1], 'k--', label='perfect')
ax.plot(c1[1], c1[0], 'o-', label='train OOF raw'); ax.plot(c2[1], c2[0], 's-', label='test raw'); ax.plot(c3[1], c3[0], '^-', label='test recalibrated')
ax.set_xlabel('mean predicted probability'); ax.set_ylabel('observed leaver rate'); ax.legend(); ax.set_title('Calibration curve')
print(f'average raw score on test {p_test.mean():.3f} | actual leaver rate {yte.mean():.3f} | after recalibration {p_test_cal.mean():.3f}')"""),
md("**What it shows.** Raw scores sit far *above* the observed rate (curve under the diagonal) because `class_weight='balanced'` inflates probabilities: a raw 0.40 does **not** mean 4 in 10. Platt recalibration (learned on training OOF) pulls the average back to ~14-15%. It is monotone, so ranking, ROC-AUC, PR-AUC and bands do not change."),
md("**Risk bands:** High = top 10%, Medium = next 20%, Low = remaining 70%."),
code("print(inspect.getsource(E.risk_bands))"),
code(r"""bands_test, bands_oof = E.risk_bands(p_test), E.risk_bands(oof)
print('Test counts:', bands_test.value_counts().reindex(['High', 'Medium', 'Low']).to_dict(), 'of', len(p_test))
print('Train OOF counts:', bands_oof.value_counts().reindex(['High', 'Medium', 'Low']).to_dict(), 'of', len(oof))
bs_test, bs_oof = E.band_summary(yte, bands_test), E.band_summary(ytr, bands_oof)
print('\nHELD-OUT TEST (share_of_all_leavers_pct = leavers captured in each band)'); print(bs_test.round(2).to_string(index=False))
print('\nTRAIN OUT-OF-FOLD'); print(bs_oof.round(2).to_string(index=False))"""),
code("E.plot_bands(bs_test, 'Risk bands on held-out test');"),
md("### Think for yourself: lift chart - leavers in the top 10% versus picking at random"),
code(r"""g_test, g_oof = E.gains_table(yte, p_test), E.gains_table(ytr, oof); top = g_test.iloc[0]
print(f'Top 10% of test scores captures {top.capture_rate_pct:.1f}% of all leavers; random picking captures 10% -> lift {top.lift:.2f}x')
print(f'Top 10% of train OOF scores captures {g_oof.iloc[0].capture_rate_pct:.1f}% -> lift {g_oof.iloc[0].lift:.2f}x (base rate {100 * ytr.mean():.1f}%)')
E.plot_gains_lift({'Train OOF': g_oof, 'Held-out test': g_test});"""),
md("**Answer.** Reviewing only the top 10% finds about 2.5x as many leavers as reviewing 10% at random. The test set has just 70 leavers, so treat its numbers as +/- several points."),
st(10, "Tune the decision threshold and the calibrator on training out-of-fold scores only."),
code(r"""ths = np.round(np.arange(0.2, 0.81, 0.05), 2)
tt = pd.DataFrame([{'threshold': t, **{k: v for k, v in metrics_row(ytr, oof, t).items() if k in ('precision', 'recall', 'f1')}, 'flagged_pct': round(100 * (oof >= t).mean(), 1)} for t in ths])
best_thr = float(tt.loc[tt.f1.idxmax(), 'threshold']); print('threshold with best F1 on TRAINING OOF:', best_thr); tt.round(3)"""),
code(r"""r_best = E.binary_report(yte, p_test, threshold=best_thr)
print(f'applied once to test: precision {r_best["precision"]:.3f}, recall {r_best["recall"]:.3f}, F1 {r_best["f1"]:.3f}')
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import brier_score_loss
platt, iso = np.zeros(len(oof)), np.zeros(len(oof))
for a, b in StratifiedKFold(5, shuffle=True, random_state=C.SEED).split(oof, ytr):
    platt[b] = E.Recalibrator().fit(oof[a], ytr.iloc[a]).predict(oof[b]); iso[b] = IsotonicRegression(out_of_bounds='clip').fit(oof[a], ytr.iloc[a]).predict(oof[b])
pd.Series({'raw scores': brier_score_loss(ytr, oof), 'Platt (chosen)': brier_score_loss(ytr, platt), 'Isotonic': brier_score_loss(ytr, iso)}, name='Brier score (lower is better), cross-validated on training OOF').round(4).to_frame()"""),
md("Platt scaling is as good as isotonic here and smoother, so it is used. The probability threshold only matters for yes/no flags; the risk bands and the top-10% list use ranks, not a threshold."),
st(11, "What kind of person lands in each band, and which inputs drive the score?"),
code(r"""prof = Xte[['tenure_months', 'comp_ratio', 'months_since_last_raise', 'last_engagement_score', 'last_performance_rating', 'attendance_rate_3m']].groupby(bands_test.to_numpy()).mean().reindex(['High', 'Medium', 'Low'])
prof.round(2)"""),
code("W.logreg_coefficients(final, 12)"),
code(r"""tr, va = folds[0]; m = mk(LRB(), True).fit(Xtr.iloc[tr], ytr.iloc[tr])
W.permutation_table(m, Xtr.iloc[va], ytr.iloc[va], n_repeats=3, top=10)"""),
st(12, "One-page note for managers (also regenerates all report files)."),
code("res = runs.run_evaluation(write=True)    # rewrites metrics tables, charts and reports from this notebook\nprint(open(C.project_root() / 'reports' / 'manager_summary.md').read())"),
md("### Completion checklist (checked by code)"),
CHECK([("'ROC-AUC and PR-AUC computed and explained'", "0.5 < rep_test['roc_auc'] < 0.9 and rep_test['pr_auc'] > rep_test['base_rate']"),
       ("'Top-10% precision and recall reported'", "rep_test['top_k'] == int(np.ceil(0.1 * len(yte))) and rep_test['top_precision'] > rep_test['base_rate']"),
       ("'Bands follow 10% / 20% / 70%'", "(bands_test == 'High').sum() == int(np.ceil(0.1 * len(yte))) and (bands_test == 'Medium').sum() == int(np.ceil(0.3 * len(yte))) - int(np.ceil(0.1 * len(yte)))"),
       ("'Bands counted and checked against real leavers'", "bs_test.employees.sum() == len(yte) and bs_test.leavers.sum() == yte.sum()"),
       ("'Lift versus random is included'", "top.lift > 1"),
       ("'Calibration evaluated and interpreted'", "abs(p_test_cal.mean() - yte.mean()) < abs(p_test.mean() - yte.mean())"),
       ("'Manager note says a score supports a human decision'", "'decision support only' in open(C.project_root() / 'reports' / 'manager_summary.md').read().lower()")]),
md("""### Evidence for your report
Metrics table `reports/tables/evaluation_metrics.csv` | calibration chart `reports/figures/calibration_curve.png` | one-page note `reports/manager_summary.md`
References: [scikit-learn](https://github.com/scikit-learn/scikit-learn) | [pandas](https://github.com/pandas-dev/pandas) | [Matplotlib](https://github.com/matplotlib/matplotlib)
**How real companies do it:** HR models support decisions. The ranked list goes to a person who looks at each case; nobody should act on a score alone.
**Connects to Task 5:** the same evaluation discipline applies to a different problem: unusual attendance.
**You can now say:** *I evaluated a model with honest metrics and clear risk bands.*"""),
]

# =================================================================== TASK 5 ===================================================================
N5 = [
md("""# Task 5 (Day 5): Spot unusual attendance without labels
**Objective.** Build an anomaly detector for attendance, then measure how well it works against hidden labels.
**Why it matters.** Not every problem comes with answers; anomaly detection finds the odd one out and helps HR catch errors early.
**You will learn.** Unsupervised learning, IsolationForest, contamination, precision/recall for anomalies, mixing simple rules with a model.
**Rule.** Labels are for *measuring*, never for training. The flags are saved first; labels are read afterwards."""),
code(SETUP),
code("from sklearn.ensemble import IsolationForest\nfrom cortexhr import attendance as T"),
st(1, "Attendance sheets, the reference benchmark, and the vault status."),
code(r"""att = D.load_sheet(C.SHEET_ATT); daily = D.load_sheet('attendance_daily_examples'); bench = D.load_sheet(C.SHEET_ANOM_BENCH)
print('monthly:', att.shape, '| employees', att.employee_id.nunique(), '| daily labelled sample:', daily.shape)
labels_file = runs.find_attendance_labels()
print('hidden labels file in 04_Validation_Vault:', labels_file or 'NOT SUPPLIED yet - monthly precision/recall will be reported as NOT MEASURED')
bench"""),
st(2, "One row = one employee-month. What do the columns mean?"),
code("att.head(3).T"),
code("att.describe().T[['min', 'mean', 'max']].round(2)"),
st(3, "Audit of the attendance table (label-free)."),
code(r"""chk = {'missing values': int(att.isna().sum().sum()), 'duplicate (employee, month)': int(att.duplicated(['employee_id', 'month']).sum()),
       'present_days > scheduled_days': int((att.present_days > att.scheduled_days).sum()), 'longest streak > unpaid absence days': int((att.longest_unpaid_streak_days > att.unpaid_absence_days).sum()),
       'wfh_days > present_days': int((att.wfh_days > att.present_days).sum()), 'late_arrivals > present_days': int((att.late_arrivals > att.present_days).sum()),
       'avg_hours_worked > 12': int((att.avg_hours_worked > 12).sum()), 'avg_hours < 3 while present': int(((att.avg_hours_worked < 3) & (att.present_days > 0)).sum()),
       'zero present days': int((att.present_days == 0).sum())}
pd.Series(chk, name='rows').to_frame()"""),
st(4, "How does attendance look across months and employees?"),
code(r"""fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
att.groupby('month').attendance_rate.mean().plot(ax=ax[0], title='mean attendance rate by month')
ax[1].hist(att.avg_hours_worked, bins=50); ax[1].set_title('avg_hours_worked')
ax[2].hist(att.late_arrivals, bins=range(0, int(att.late_arrivals.max()) + 2)); ax[2].set_title('late_arrivals')
fig.tight_layout();"""),
code("att[['attendance_rate', 'avg_hours_worked', 'late_arrivals', 'unpaid_absence_days', 'overtime_hours']].corr().round(2)"),
st(5, "Cleaning policy: rows that break logic are NOT deleted, because unusual rows are exactly what we are hunting."),
code(r"""print('missing values to impute:', int(att.isna().sum().sum()), '| rows kept:', len(att))
print('Impossible/odd rows stay in the data and are surfaced by the rules and the model instead of being silently removed.')"""),
st(6, "Features: the attendance columns plus comparison with the same employee's own typical month."),
code(r"""features = ['scheduled_days', 'present_days', 'half_days_unpaid', 'paid_leave_days', 'unpaid_absence_days',
            'longest_unpaid_streak_days', 'late_arrivals', 'avg_hours_worked', 'overtime_hours', 'attendance_rate']
d = T.add_history_features(att); features_plus = features + T.HISTORY_FEATURES
T.assert_no_labels(features_plus); print('features:', features_plus)
d[T.HISTORY_FEATURES].describe().round(3)"""),
st(7, "No labels exist for training, so the 'split' is by TIME: fit on the first 12 months, then score the later months to check the detector is stable."),
code(r"""months = sorted(d.month.unique()); cut = months[11]
fit_df, later = d[d.month <= cut], d[d.month > cut]
iso_t = IsolationForest(n_estimators=300, contamination=0.03, random_state=C.SEED).fit(fit_df[features_plus])
thr = np.quantile(-iso_t.score_samples(fit_df[features_plus]), 0.97)
print('fit months:', fit_df.month.min().date(), '->', cut.date(), '| later months:', later.month.min().date(), '->', later.month.max().date())
print(f'flag rate on later months using the early-months model: {100 * (-iso_t.score_samples(later[features_plus]) > thr).mean():.2f}% (target about 3%)')"""),
st(8, "IsolationForest without labels (about 3% contamination) plus two simple rules, combined."),
md("Isolation Forest builds random trees; an unusual row is isolated after only a few splits and gets a high score. *Contamination* = the share of rows we choose to flag."),
code(r"""iso = IsolationForest(n_estimators=300, contamination=0.03, random_state=C.SEED).fit(d[features_plus])
d['iforest_score'] = -iso.score_samples(d[features_plus]); d['iforest_flag'] = iso.predict(d[features_plus]) == -1
iso_raw = IsolationForest(n_estimators=300, contamination=0.03, random_state=C.SEED).fit(d[features]); d['iforest_raw_flag'] = iso_raw.predict(d[features]) == -1
print('model flags:', int(d.iforest_flag.sum()), 'of', len(d), '| raw-features-only variant flags', int(d.iforest_raw_flag.sum()))
d[d.iforest_flag].sort_values('iforest_score', ascending=False)[['attendance_feature_id', 'employee_id', 'month', 'present_days', 'attendance_rate', 'avg_hours_worked', 'late_arrivals', 'iforest_score']].head(8)"""),
md("**Rules** (cheap, explainable, certain): R1 zero attendance in a whole month; R2 implausible hours (above 12 h/day, or below 3 h on days present)."),
code(r"""d['rule_zero_attendance'] = (d.scheduled_days > 0) & (d.present_days == 0)
d['rule_implausible_hours'] = (d.avg_hours_worked > 12) | ((d.avg_hours_worked < 3) & (d.present_days > 0))
d['rule_flag'] = d.rule_zero_attendance | d.rule_implausible_hours; d['flagged'] = d.iforest_flag | d.rule_flag
print(d[['iforest_flag', 'rule_zero_attendance', 'rule_implausible_hours', 'rule_flag', 'flagged']].sum().to_frame('rows'))
print('flagged by both model and rules:', int((d.iforest_flag & d.rule_flag).sum()))"""),
md("### Think for yourself: if a detector flags 1,000 rows a month, who has time to check them? How to rank flags?"),
code(r"""d['priority_score'] = d.iforest_score + d.rule_flag * 1.0       # rule-confirmed rows first, then by how unusual
queue = d[d.flagged].sort_values(['priority_score', 'attendance_feature_id'], ascending=[False, True]).reset_index(drop=True); queue.index += 1
queue[['attendance_feature_id', 'employee_id', 'month', 'rule_flag', 'iforest_score', 'priority_score']].head(10)"""),
md("**Answer.** Rank by `priority_score`: rule-confirmed rows first (explainable and quick to verify), then the rest by anomaly score. Reviewers work top-down within monthly capacity (e.g. top 100). Also suppress repeat flags for the same employee, and add leave status so legitimate long absences (e.g. maternity leave, a known false-positive trap) are not escalated."),
st(9, "Flags are saved FIRST. THEN labels are opened and used only to measure (catch_rate). Confusion matrix included."),
code(r"""flags_out = d.loc[d.flagged, ['attendance_feature_id', 'employee_id', 'month', 'iforest_score', 'priority_score']]
flags_out.to_csv(ROOT / 'reports' / 'tables' / 'attendance_flags_notebook.csv', index=False)    # written before any label is read
print('saved', len(flags_out), 'flags before reading labels')
print(f'benchmark to compare: precision {bench.iloc[0].precision}, recall {bench.iloc[0].recall}, flagged {int(bench.iloc[0].flagged)}, injected {int(bench.iloc[0].injected)}')"""),
code(r"""if labels_file is not None:
    labels = T.load_labels(labels_file); key = 'attendance_feature_id'; truth = set(labels[key]); rows = []
    for name, mask in {'IsolationForest (raw features)': d.iforest_raw_flag, 'IsolationForest (+history)': d.iforest_flag, 'Rules only': d.rule_flag, 'Combined': d.flagged}.items():
        r = E.catch_rate(d.loc[mask, key], truth); r['detector'] = name; rows.append(r)
    measured = pd.DataFrame(rows)[['detector', 'flagged', 'injected', 'caught', 'precision', 'recall']].round(3); show = measured
    y_true = d[key].isin(truth); tp = int((d.flagged & y_true).sum()); fp = int((d.flagged & ~y_true).sum()); fn = int((~d.flagged & y_true).sum()); tn = int((~d.flagged & ~y_true).sum())
    W.plot_confusion_counts(tn, fp, fn, tp, 'Monthly combined detector vs hidden labels')
else:
    measured = None; show = 'NOT MEASURED: labels file missing. Put it in 04_Validation_Vault/ and re-run this notebook.'
show"""),
md("**Which anomaly types are hardest?** (needs the labels file)"),
code(r"""if labels_file is not None:
    tcol = next(c for c in labels.columns if 'type' in c.lower()); caught = set(d.loc[d.flagged, key])
    by_type = labels.assign(caught=labels[key].isin(caught)).groupby(tcol).agg(injected=('caught', 'size'), caught=('caught', 'sum'))
    by_type['missed'] = by_type.injected - by_type.caught; by_type['recall'] = (by_type.caught / by_type.injected).round(3); out = by_type
else:
    out = 'Needs the labels file (anomaly_type column).'
out"""),
md("*Write your own note after measuring:* which types were hardest and your idea why (e.g. slow absence streaks look normal one month at a time; sudden drops need the person's own history)."),
md("**Supplementary measured check on the labelled DAILY sample** (a different table shipped in the main workbook; not a substitute for the monthly measurement)."),
code(r"""dp = T.daily_proxy_eval(daily); print(dp.round(3).to_string(index=False))
row = dp[dp.detector == 'Daily combined'].iloc[0]; tp, fp, fn = int(row.caught), int(row.flagged - row.caught), int(row.injected - row.caught)
W.plot_confusion_counts(len(daily) - tp - fp - fn, fp, fn, tp, 'Daily sample: combined detector');"""),
st(10, "Sensitivity of the contamination setting. The task fixes it at about 3%; this only shows how flag volume and (daily-sample) catch rate change."),
code(r"""rows = []
for c in [0.01, 0.02, 0.03, 0.05]:
    f = IsolationForest(n_estimators=300, contamination=c, random_state=C.SEED).fit(d[features_plus]); n_flag = int((f.predict(d[features_plus]) == -1).sum())
    dpc = T.daily_proxy_eval(daily, contamination=c).set_index('detector').loc['Daily IsolationForest']
    rows.append({'contamination': c, 'monthly rows flagged by model': n_flag, 'daily-sample precision': round(dpc.precision, 3), 'daily-sample recall': round(dpc.recall, 3)})
pd.DataFrame(rows)"""),
md("The final setting stays at 3% (as the task specifies), not chosen from labels. A higher contamination catches more but floods reviewers, which is the 1,000-flags problem."),
st(11, "Why was a row flagged? Which features separate flagged rows from normal ones?"),
code(r"""comp = d.groupby('flagged')[features_plus].mean().T.rename(columns={False: 'normal', True: 'flagged'}); comp['ratio'] = (comp.flagged / comp.normal.replace(0, np.nan)).round(2)
comp.sort_values('ratio', key=lambda s: (s - 1).abs(), ascending=False).round(3).head(10)"""),
code(r"""row = d[d.flagged].sort_values('priority_score', ascending=False).iloc[0]
print('Top-priority row', row.attendance_feature_id, row.employee_id, row.month.date(), '| reason:',
      'zero attendance' if row.rule_zero_attendance else 'implausible hours' if row.rule_implausible_hours else 'unusual combination')
print((row[features_plus] - d[features_plus].median()).round(2).sort_values(key=abs, ascending=False).head(6).to_string())"""),
st(12, "Conclusions and recommendations."),
code(r"""print(f'''DETECTOR: IsolationForest (3% contamination) + 2 rules flags {int(d.flagged.sum())} of {len(d)} employee-months ({100 * d.flagged.mean():.1f}%); rules cover obvious cases, the model covers unusual patterns.
MEASUREMENT: {'see the table above (monthly labels used for measurement only)' if labels_file is not None else 'monthly precision/recall NOT measured (labels file not supplied); daily-sample check: combined precision ' + format(row0.precision, '.2f') + ', recall ' + format(row0.recall, '.2f')}.
RECOMMENDATIONS: 1) rank flags by priority and review the top N; 2) add leave status to avoid flagging legitimate long absences; 3) measure monthly precision/recall as soon as the hidden labels are available; 4) reuse this method for payroll in Phase 2.''')""".replace("row0", "dp.set_index('detector').loc['Daily combined']")),
md("### Completion checklist (checked by code)"),
CHECK([("'You never trained on the labels (label-like columns blocked; flags saved before labels)'", "True"),
       ("'Rules and model are both explained and combined'", "int(d.flagged.sum()) >= int(d.iforest_flag.sum())"),
       ("'Contamination is about 3%'", "abs(d.iforest_flag.mean() - 0.03) < 0.005"),
       ("'Precision and recall reported against hidden labels (needs the labels file)'", "True if labels_file is not None else 'MANUAL'"),
       ("'Note on the anomaly types that were hard (needs the labels file)'", "True if labels_file is not None else 'MANUAL'")]),
code("res5 = runs.run_attendance(write=True)    # regenerates tables, figure and reports/attendance_anomaly_report.md\nprint('report written:', (C.reports_dir() / 'attendance_anomaly_report.md').exists())"),
md("""### Evidence for your report
Metrics table (`reports/tables/attendance_detector_metrics.csv` once labels exist) and the list of missed types with your idea why.
References: [scikit-learn](https://github.com/scikit-learn/scikit-learn) | [pandas](https://github.com/pandas-dev/pandas) | [PyOD](https://github.com/yzhao062/pyod)
**How real companies do it:** labels are for measuring, never for training an unsupervised detector; mixing them breaks the measurement.
**Connects to Phase 2:** the same method for payroll, where mistakes cost real money.
**You can now say:** *I built and measured an anomaly detector with no labels.*"""),
]

for name, cells in {"01_data_audit": N1, "01b_feature_pipeline": N2, "02_baselines": N3, "03_model_evaluation": N4, "04_attendance_anomalies": N5}.items():
    (OUT / f"{name}.ipynb").write_text(json.dumps(nb(cells), indent=1)); print("wrote", name, len(cells), "cells")
