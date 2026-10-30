# ============================ PHASE 1: BUSINESS AND PROBLEM UNDERSTANDING ============================
CELLS += [phase(1, "BUSINESS AND PROBLEM UNDERSTANDING"),
cov(1, "Business Understanding", [
 ("Understand the business problem and requirements; identify the problem to solve", "Cell 1.1: HR loses people it would like to keep; wants an early, ranked list of people at risk of leaving in 12 months"),
 ("Why a machine-learning solution", "Cell 1.2 compares a simple rule with ML on the same data"),
 ("Stakeholders, end users, business objectives", "Cell 1.1 table"), ("Current process and its limitations", "Cell 1.1 table"),
 ("Expected business outcome; measurable success criteria", "Cell 1.1 table + Cell 1.3 (thresholds checked at the end, Section 37)"),
 ("Cost of incorrect predictions", "Cell 1.3: explicit, adjustable cost assumptions and expected-cost table"),
 ("Operational, legal, ethical, privacy constraints", "Cell 1.1 table (human review only, protected attributes kept out of the model, DPDP-style privacy)")]),
md("""**1.1 Business brief**

| Item | Answer |
|---|---|
| Problem | Valued employees leave unexpectedly; replacing them is slow and costly |
| Why ML | Many weak signals (pay position, raise age, engagement, workload) combine better in a model than in a manager's memory; a ranked list scales to thousands of people |
| Stakeholders / users | HR business partners and line managers (users); CHRO (sponsor); employees (affected); data/legal/privacy teams |
| Current process | Exit interviews after the fact; ad-hoc manager judgement; no ranked early-warning list |
| Expected outcome | A monthly ranked list; HR reviews the top group and acts on fixable causes (pay gap, stalled raise, workload) |
| Success criteria | Top-10% group captures at least 2x the leavers random selection would (lift >= 2); ROC-AUC >= 0.70; nothing leaks from the future; no automatic employment decisions |
| Constraints | Scores support a human decision only; protected attributes (age, gender, ...) are not model inputs and are used only for fairness checks; employee data stays internal; scores are not shown to the employee |"""),
code(r"""raw_csv = H.load('attrition_dataset')                                   # official loader (CSV export of the workbook)
raw = D.load_attrition()                                                # same table with typed dates (xlsx)
assert raw_csv.shape == raw.shape == (2389, 42); print('rows x columns:', raw.shape)
y_all = raw[C.TARGET]
print(f'attrition rate {100 * y_all.mean():.2f}%  ->  a 14.6% base rate means a random list of 10 people holds ~1.5 leavers')"""),
code(r"""# 1.2 Is ML better than a simple rule? Compare a hand-written 'stale raise' rule score with the model (cross-validated) later in 3.3.
rule_score = raw.months_since_last_raise.fillna(raw.months_since_last_raise.median())
print('single hand rule (months since last raise) ROC-AUC:', round(roc_auc_score(y_all, rule_score), 3))"""),
code(r"""# 1.3 Cost of wrong predictions: explicit ASSUMPTIONS (change them to your company's numbers)
cost_fn, cost_fp = 5.0, 0.5       # cost units: missing a leaver (replacement, lost knowledge) vs a pointless review conversation
n, pos = len(raw), int(y_all.sum())
def expected_cost(flagged, caught): return (pos - caught) * cost_fn + (flagged - caught) * cost_fp
pd.DataFrame({'strategy': ['review nobody', 'review everybody', 'review a random 10%'],
              'flagged': [0, n, int(.1 * n)], 'leavers caught': [0, pos, int(.1 * pos)]}).assign(
    expected_cost=lambda d: [expected_cost(f, c) for f, c in zip(d.flagged, d['leavers caught'])])"""),
cov(2, "Problem Definition", [
 ("Convert the business problem into an ML problem; confirm classification", "Cell 2.1: predict a yes/no outcome per employee"),
 ("Unit of prediction; prediction time; horizon", "Cell 2.1: one employee at one `observation_date`; predict at that date; horizon 12 months"),
 ("Target variable and meaning; positive and negative classes", "Cell 2.1: `left_within_12m`, 1 = left, 0 = stayed"),
 ("Binary / multiclass / multilabel / ordinal", "Cell 2.2 verifies exactly two classes -> binary classification"),
 ("Baseline performance and expected improvement", "Cell 2.3: majority baseline and supplied benchmark sheet"),
 ("Primary evaluation metric from business requirements", "Cell 2.3: PR-AUC and top-10% recall/lift (business list), ROC-AUC as ranking check")]),
code(r"""print('unit of prediction : one employee at one observation_date')
print('prediction time      :', raw.observation_date.min().date(), 'to', raw.observation_date.max().date(), '(snapshot date of each row)')
print('prediction horizon   : 12 months after observation_date')
print('target               :', C.TARGET, '| classes:', sorted(y_all.unique()), '| positive class = left (1)')"""),
code(r"""print('number of classes:', y_all.nunique(), '-> BINARY classification (not multiclass, multilabel or ordinal)')
print('label mapping kept:', {0: 'stayed', 1: 'left'})"""),
code(r"""bench = H.load('baseline_benchmarks'); print(bench[['model', 'roc_auc_mean', 'pr_auc_mean', 'recall_mean', 'precision_mean', 'accuracy_mean']].round(3).to_string(index=False))
print(f'\nmajority baseline accuracy {100 * (1 - y_all.mean()):.1f}% with recall 0')
print('PRIMARY metrics: PR-AUC and top-10% recall/lift (the list HR can act on); SECONDARY: ROC-AUC, recall, precision. Accuracy is NOT used to choose models.')"""),
cov(3, "Problem Feasibility", [
 ("Sufficient data available", "Cell 3.1: rows, positives and positives-per-feature"),
 ("Reliable target labels exist", "Cell 3.2: label consistency with exit dates"),
 ("Data quality and representativeness", "Cell 3.2: source mix, time span, coverage of monthly tables"),
 ("Project cost, time and compute", "Cell 3.3: measured training times"),
 ("Is the expected prediction achievable", "Cell 3.3: cross-validated AUC well above 0.5 but modest"),
 ("ML better than a simple rule-based solution", "Cell 3.3: rule vs logistic regression, same folds")]),
code(r"""print(f'{len(raw):,} rows, {int(y_all.sum())} positives ({100 * y_all.mean():.1f}%), {len(C.NUMERIC_CORE) + len(C.CATEGORICAL_CORE)} candidate features')
print('events per candidate feature:', round(y_all.sum() / (len(C.NUMERIC_CORE) + len(C.CATEGORICAL_CORE)), 1), '(>10 is comfortable for regularised models)')"""),
code(r"""left = raw[raw[C.TARGET] == 1]
days = (left.exit_date - left.observation_date).dt.days
print('label reliability: all leavers have an exit_date within 12 months of the snapshot ->', bool(days.between(0, 366).all()), '| range', int(days.min()), 'to', int(days.max()), 'days')
print('stayers with an exit_date:', int(raw.loc[raw[C.TARGET] == 0, 'exit_date'].notna().sum()))
print(raw.groupby('source')[C.TARGET].agg(['count', 'mean']).round(3))
print('representativeness caveat: 2,000 historical rows (2021-2025) have no attendance/payroll history; only the 30 Sep 2025 snapshot rows do')"""),
code(r"""S = runs.load_model_frame(True); Xtr, ytr, Xte, yte, folds, dfj = S['Xtr'], S['ytr'], S['Xte'], S['yte'], S['folds'], S['df']
mk = lambda est, monthly=True: Pipeline([('features', F.build_pipeline(monthly)), ('model', est)])
LRB = lambda **kw: LogisticRegression(class_weight='balanced', max_iter=2000, random_state=SEED, **kw)
t0 = time.time(); oof_lr = cross_val_predict(mk(LRB()), Xtr, ytr, cv=folds, method='predict_proba')[:, 1]; train_s = time.time() - t0
rule_oof = Xtr.months_since_last_raise.fillna(Xtr.months_since_last_raise.median()).to_numpy()
print(f'ROC-AUC  rule: {roc_auc_score(ytr, rule_oof):.3f}   logistic regression (5-fold OOF): {roc_auc_score(ytr, oof_lr):.3f}   -> ML beats the single rule, but only moderately')
print(f'compute: 5-fold fit + predict took {train_s:.1f}s on a laptop CPU: cost and time are small')
print('feasible: yes - signal exists (AUC about 0.7) but the problem is hard; plan for decision SUPPORT, not decision making')"""),

# ============================ PHASE 2 ============================
phase(2, "DATA COLLECTION AND UNDERSTANDING"),
cov(4, "Data Collection", [
 ("Collect from CSV/Excel/SQL/APIs/warehouses", "Cell 4.1: Excel workbook (26 sheets) and its CSV exports; other source types noted"),
 ("Data owners and sources; permissions and privacy", "Cell 4.2: README sheet (fictional company, who owns what); protected attributes excluded"),
 ("Collection methods, sampling", "Cell 4.2: snapshot rows + historical extension rows"),
 ("Freshness, historical coverage, update frequency", "Cell 4.3: date ranges per table"),
 ("Verify schema and data dictionary; meaning and units of columns", "Cell 4.4: columns vs `data_dictionary` sheet"),
 ("Represents intended population; target defined consistently", "Cell 4.5")]),
code(r"""xl = pd.ExcelFile(C.data_path()); print(len(xl.sheet_names), 'sheets:', xl.sheet_names)
print('CSV exports in 03_Data/csv:', len(list((ROOT / '03_Data' / 'csv').glob('*.csv'))), '| other sources (SQL, APIs, warehouses) are not used in this phase')"""),
code(r"""readme = H.load('README'); readme.dropna(how='all').head(14)"""),
code(r"""att, pay = D.load_sheet(C.SHEET_ATT), D.load_sheet(C.SHEET_PAY)
pd.DataFrame({'table': ['attrition (observation_date)', 'attendance (month)', 'payroll (month)'],
              'from': [raw.observation_date.min().date(), att.month.min().date(), pay.month.min().date()],
              'to': [raw.observation_date.max().date(), att.month.max().date(), pay.month.max().date()],
              'rows': [len(raw), len(att), len(pay)], 'update frequency': ['snapshots', 'monthly', 'monthly']})"""),
code(r"""dd = H.load('data_dictionary'); tcol, ccol = dd.columns[0], dd.columns[1]
d_attr = dd[dd[tcol].astype(str).str.contains('attrition_dataset', na=False)]
missing_in_dict = sorted(set(raw.columns) - set(d_attr[ccol].astype(str)))
print('columns documented in data_dictionary:', len(d_attr), '| undocumented columns:', missing_in_dict)
d_attr.head(8)"""),
code(r"""print('target defined consistently: left_within_12m == (exit within 365 days of observation_date):',
      bool(((raw.exit_date - raw.observation_date).dt.days.between(0, 366) == (raw[C.TARGET] == 1)).all()))
print('units: money in INR per year; tenure in months; rates are fractions 0-1; hours per day')"""),
cov(5, "Data Loading and Initial Inspection", [
 ("Load dataset; first and last records; rows and columns", "Cells 5.1-5.2"), ("Column names and data types; numeric/categorical/datetime/text/id columns", "Cell 5.3"),
 ("Descriptive statistics; unique values per feature", "Cell 5.4"), ("Constant and near-constant columns", "Cell 5.5"),
 ("Dataset not empty / correctly loaded; expected schema and integrity", "Cell 5.6")]),
code(r"""print(raw.shape); raw.head(3)"""), code(r"""raw.tail(3)"""),
code(r"""kinds = {c: ('identifier' if c in C.ID_COLS else 'datetime' if 'date' in c else 'text' if c == 'exit_reason' else 'categorical' if raw[c].dtype == object or str(raw[c].dtype).startswith('str') else 'numeric') for c in raw.columns}
print(pd.Series(kinds).value_counts().to_dict()); pd.DataFrame({'dtype': raw.dtypes.astype(str), 'kind': pd.Series(kinds)}).T"""),
code(r"""raw.describe(include='all').T[['count', 'unique', 'mean', 'std', 'min', 'max']].head(40)"""),
code(r"""nun = raw.nunique().sort_values(); top = raw.apply(lambda s: s.value_counts(normalize=True, dropna=False).iloc[0])
print('constant columns:', list(nun[nun <= 1].index), '| near-constant (>=98% one value):', list(top[top >= .98].index))
pd.DataFrame({'n_unique': nun, 'top_value_share': top.round(3)}).head(12)"""),
code(r"""assert len(raw) > 0 and raw.shape == (2389, 42) and not raw.columns.duplicated().any()
print('integrity: non-empty, 2,389 x 42, unique column names, row_id unique ->', raw.row_id.is_unique)"""),
]
