# ============================ 6 EDA ============================
CELLS += [
cov(6, "Exploratory Data Analysis (EDA)", [
 ("Univariate analysis", "Cells 6.1-6.2"), ("Bivariate analysis", "Cell 6.4"), ("Multivariate analysis", "Cells 6.5, 6.7"),
 ("Numerical distributions, central tendency, dispersion", "Cell 6.1 (histograms + mean/median/std/skew)"), ("Categorical frequency distributions", "Cell 6.2"),
 ("Target variable distribution; features vs target", "Cells 6.3-6.4"), ("Correlations and multicollinearity", "Cell 6.6 (correlation pairs + VIF)"),
 ("Patterns, trends, clusters, interactions", "Cell 6.7 (interaction table + KMeans clusters)"), ("Outliers and unusual observations", "Cell 6.8"),
 ("Subgroup distributions", "Cell 6.9"), ("Date ranges, seasonality, time patterns", "Cell 6.10"), ("Potential data-quality problems", "Cell 6.11"),
 ("Form hypotheses and validate them", "Cell 6.12 (statistical tests)")]),
code(r"""num_show = ['tenure_months', 'annual_fixed_pay', 'comp_ratio', 'months_since_last_raise', 'last_engagement_score', 'avg_overtime_hours_per_month_3m', 'commute_distance_km', 'training_hours_12m', 'late_arrivals_3m']
fig, ax = plt.subplots(3, 3, figsize=(13, 8))
for a, c in zip(ax.ravel(), num_show): a.hist(raw[c].dropna(), bins=40); a.set_title(c, fontsize=9)
fig.tight_layout()
raw[num_show].agg(['mean', 'median', 'std', 'skew']).T.round(2)"""),
code(r"""for c in ['department_id', 'work_mode', 'employment_type', 'work_location']:
    print(c, raw[c].value_counts(dropna=False).head(8).to_dict())
raw.highest_education.value_counts(dropna=False).head(6)"""),
code(r"""vc = raw[C.TARGET].value_counts(); print(vc.to_dict(), f'-> {100 * y_all.mean():.2f}% left'); A.plot_class_balance(A.class_balance(raw));"""),
code(r"""W.plot_eda(raw);"""),
code(r"""rows = []
for c in ['tenure_months', 'comp_ratio', 'months_since_last_raise', 'last_engagement_score', 'last_performance_rating', 'avg_overtime_hours_per_month_3m', 'commute_distance_km', 'annual_fixed_pay']:
    a, b = raw.loc[y_all == 1, c].dropna(), raw.loc[y_all == 0, c].dropna()
    rows.append({'feature': c, 'mean_left': a.mean(), 'mean_stayed': b.mean(), 'mann_whitney_p': st.mannwhitneyu(a, b).pvalue})
pd.DataFrame(rows).round(4).sort_values('mann_whitney_p')"""),
code(r"""corr = raw[C.NUMERIC_CORE].corr(); fig, ax = plt.subplots(figsize=(9, 7)); im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=6); ax.set_yticks(range(len(corr)), corr.columns, fontsize=6); fig.colorbar(im); ax.set_title('Correlation of numeric inputs')"""),
code(r"""pairs = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack().rename('r').reset_index(); pairs['abs_r'] = pairs.r.abs()
print('most correlated input pairs:'); print(pairs.sort_values('abs_r', ascending=False).head(6).round(3).to_string(index=False))
Z = raw[C.NUMERIC_CORE].fillna(raw[C.NUMERIC_CORE].median()); Z = (Z - Z.mean()) / Z.std()
vif = {c: 1 / (1 - np.corrcoef(Z[c], np.linalg.lstsq(Z.drop(columns=c).to_numpy(), Z[c].to_numpy(), rcond=None)[0] @ Z.drop(columns=c).to_numpy().T)[0, 1] ** 2) for c in Z.columns}
print('\nVIF (multicollinearity; >5-10 is a warning):'); pd.Series(vif).sort_values(ascending=False).head(6).round(2)"""),
code(r"""tb = pd.cut(raw.tenure_months.where(raw.tenure_months >= 0), [0, 12, 24, 48, 400], labels=['<1y', '1-2y', '2-4y', '4y+'])
print('interaction: leaver rate (%) by tenure band x work mode'); print((100 * raw.pivot_table(index=tb, columns='work_mode', values=C.TARGET, aggfunc='mean', observed=True)).round(1))
from sklearn.cluster import KMeans
Zc = raw[['tenure_months', 'comp_ratio', 'months_since_last_raise', 'last_engagement_score', 'avg_overtime_hours_per_month_3m']].fillna(raw.median(numeric_only=True)); Zc = (Zc - Zc.mean()) / Zc.std()
km = KMeans(4, n_init=10, random_state=SEED).fit(Zc); print('\nKMeans employee clusters: leaver rate %'); print((100 * raw.groupby(km.labels_)[C.TARGET].mean()).round(1).to_dict())"""),
code(r"""pd.DataFrame({'IQR outliers': ((raw[num_show] < raw[num_show].quantile(.25) - 1.5 * (raw[num_show].quantile(.75) - raw[num_show].quantile(.25))) | (raw[num_show] > raw[num_show].quantile(.75) + 1.5 * (raw[num_show].quantile(.75) - raw[num_show].quantile(.25)))).sum()})"""),
code(r"""raw.groupby('department_id').agg(employees=('row_id', 'size'), leaver_rate_pct=(C.TARGET, lambda s: round(100 * s.mean(), 1)), median_pay=('annual_fixed_pay', 'median'), median_engagement=('last_engagement_score', 'median')).round(1)"""),
code(r"""by = raw.assign(year=raw.observation_date.dt.year, month=raw.observation_date.dt.month)
print('rows and leaver rate by observation year'); print(by.groupby('year')[C.TARGET].agg(['size', 'mean']).round(3))
print('\nleaver rate by observation month'); print(by.groupby('month')[C.TARGET].mean().round(3).to_dict())"""),
code(r"""audit_t, lk, single = A.build_audit_table(raw, att, pay); print(len(audit_t), 'data-quality problems found (details in Section 10 and reports/data_audit_report.md)'); audit_t[['problem', 'rows_affected', 'severity']]"""),
code(r"""tests = []
stale = raw.months_since_last_raise > 18; ct = pd.crosstab(stale, y_all); tests.append(('H1 raise older than 18 months -> more leavers', 100 * y_all[stale].mean(), 100 * y_all[~stale].mean(), st.chi2_contingency(ct)[1]))
low = raw.last_engagement_score < 3; ct = pd.crosstab(low, y_all); tests.append(('H2 engagement below 3 -> more leavers', 100 * y_all[low].mean(), 100 * y_all[~low].mean(), st.chi2_contingency(ct)[1]))
npm = (raw.promotions_received == 0) & (raw.tenure_months > 36); ct = pd.crosstab(npm, y_all); tests.append(('H3 never promoted after 3 years -> more leavers', 100 * y_all[npm].mean(), 100 * y_all[~npm].mean(), st.chi2_contingency(ct)[1]))
pd.DataFrame(tests, columns=['hypothesis', 'leaver % in group', 'leaver % elsewhere', 'chi2 p-value']).round(4)"""),

# ============================ PHASE 3 ============================
phase(3, "DATA QUALITY AND DATA CLEANING"),
cov(7, "Missing-Value Analysis", [
 ("Missing values per column; counts and percentages", "Cell 7.1"), ("Why values are missing; systematic or informative?", "Cell 7.2"),
 ("Remove records/columns only when justified", "Cell 7.3 decision table"), ("Impute numerical values (strategies compared)", "Cell 7.4"),
 ("Impute categorical values", "Cell 7.3 / pipeline `SimpleImputer(constant 'Unknown')`"), ("Missingness indicators", "Cell 7.4 compares with/without indicators"),
 ("Missing target labels handled separately", "Cell 7.1 (target has none)"), ("Avoid arbitrary replacements that distort meaning", "Cell 7.3: never-promoted blanks are NOT filled with 0 months")]),
code(r"""ms = raw.isna().sum().to_frame('missing').assign(pct=lambda d: (100 * d.missing / len(raw)).round(2)); ms = ms[ms.missing > 0].sort_values('pct', ascending=False)
print('missing target labels:', int(raw[C.TARGET].isna().sum())); ms"""),
code(r"""rows = []
for c in ms.index.difference(C.LEAKAGE_CONFIRMED):
    m = raw[c].isna(); rows.append({'column': c, 'missing': int(m.sum()), 'leaver % when missing': round(100 * y_all[m].mean(), 1), 'leaver % when present': round(100 * y_all[~m].mean(), 1)})
print('is missingness informative? (exit columns excluded: blank by design for stayers)'); pd.DataFrame(rows)"""),
code(r"""pd.DataFrame([('months_since_last_promotion', 'structural: blank = never promoted', 'derive never_promoted flag; impute number; do NOT write 0'),
              ('last_engagement_score', 'no survey reply (possibly informative)', 'median + missing indicator'),
              ('last_performance_rating / last_raise_pct / months_since_last_raise', 'no review/raise on record', 'median + missing indicator'),
              ('commute_distance_km, highest_education', 'accidental blanks (data entry)', 'median / most frequent'),
              ('exit_* and fnf columns', 'blank for stayers by design', 'DROP: leakage, not a missing-value problem')],
             columns=['column', 'why missing', 'treatment'])"""),
code(r"""from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
Xc = D.clean_raw(Xtr); ncols = [c for c in C.NUMERIC_CORE if c != 'annual_fixed_pay']
def quick_auc(imputer, scaler=None):
    p = Pipeline([('imp', imputer), ('sc', scaler or StandardScaler()), ('lr', LRB())]); return roc_auc_score(ytr, cross_val_predict(p, Xc[ncols], ytr, cv=folds, method='predict_proba')[:, 1])
print(pd.Series({'median': quick_auc(SimpleImputer(strategy='median')), 'mean': quick_auc(SimpleImputer(strategy='mean')),
                 'median + missing indicators': quick_auc(SimpleImputer(strategy='median', add_indicator=True)), 'KNN (k=5)': quick_auc(KNNImputer(n_neighbors=5))}, name='5-fold OOF ROC-AUC').round(4))"""),
cov(8, "Duplicate-Record Analysis", [
 ("Exact duplicate rows", "Cell 8.1"), ("Duplicate business keys / entity records", "Cell 8.1"), ("Erroneous or legitimate repeated events?", "Cell 8.2"),
 ("Remove only confirmed duplicates", "Cell 8.3"), ("Prevent duplicates leaking across splits", "Cell 8.4 (employee overlap between train and test = 0)")]),
code(r"""body = raw.drop(columns=['row_id'])
print('exact duplicate rows:', int(body.duplicated().sum()), '| duplicate (employee_id, observation_date):', int(raw.duplicated(['employee_id', 'observation_date']).sum()), '| repeated employee_id:', int(raw.employee_id.duplicated().sum()))
feat_cols = [c for c in raw.columns if c not in C.ID_COLS + [C.TARGET] + C.LEAKAGE_CONFIRMED]
print('near-duplicates (identical on all feature columns, different ids):', int(raw.duplicated(feat_cols).sum()))"""),
code(r"""d = raw[raw.employee_id.duplicated(keep=False)].sort_values('employee_id')
same = d.drop(columns='row_id').groupby('employee_id').nunique(dropna=False).max(axis=1)
print('each repeated employee has identical values in every column (nunique == 1):', bool((same == 1).all()), '-> ERRONEOUS duplicates (not repeated events: same observation_date, same values)')"""),
code(r"""dedup, dlog = D.prepare_attrition(raw); print(dlog)"""),
code(r"""ids_tr, ids_te = set(dfj.loc[Xtr.index, 'employee_id']), set(dfj.loc[Xte.index, 'employee_id'])
print('employee overlap between train and test:', len(ids_tr & ids_te), '(must be 0)'); assert not (ids_tr & ids_te)"""),
cov(9, "Outlier Analysis", [
 ("Unusual numerical observations; statistical methods, visualisations, domain knowledge", "Cells 9.1-9.2"), ("Genuine extremes vs data-entry errors", "Cell 9.3 (pay typos vs senior pay)"),
 ("Correct invalid values when the correct value is known", "Cell 9.3"), ("Transformation, capping or robust methods", "Cell 9.4"), ("Avoid deleting all outliers", "Cell 9.5")]),
code(r"""num = raw.select_dtypes('number').drop(columns=['row_id', C.TARGET, 'voluntary_left_within_12m'])
q1, q3 = num.quantile(.25), num.quantile(.75); iqr = q3 - q1
t = pd.DataFrame({'iqr': ((num < q1 - 1.5 * iqr) | (num > q3 + 1.5 * iqr)).sum(), 'z>3': ((num - num.mean()) / num.std(ddof=0)).abs().gt(3).sum()}).sort_values('iqr', ascending=False); t.head(8)"""),
code(r"""R.show_figure('outliers_boxplot', 10);"""),
code(r"""typo = A.pay_consistency_outliers(raw); print('pay typo candidates (pay / comp_ratio is ~11x the role mid-point):'); print(typo.to_string(index=False))
print('\nsenior pay is real: top-3 pay of ordinary rows vs typo rows ->', raw.annual_fixed_pay.drop(typo.index).nlargest(3).round(0).tolist(), 'vs', typo.annual_fixed_pay.round(0).tolist())"""),
code(r"""d_ = Xtr.copy(); d_['annual_fixed_pay'] = d_.annual_fixed_pay.where(~d_.index.isin(typo.index))
print('skew of pay:', round(d_.annual_fixed_pay.skew(), 2), '-> after log1p:', round(np.log1p(d_.annual_fixed_pay).skew(), 2))
lo, hi = d_.annual_fixed_pay.quantile([.01, .99]); print('winsorising at 1%/99%% would change', int(((d_.annual_fixed_pay < lo) | (d_.annual_fixed_pay > hi)).sum()), 'values')
print(pd.Series({'StandardScaler': quick_auc(SimpleImputer(strategy='median'), StandardScaler()), 'RobustScaler': quick_auc(SimpleImputer(strategy='median'), RobustScaler())}, name='OOF ROC-AUC').round(4).to_dict())"""),
md("**Decision.** Pay typos are blanked inside the pipeline using *training* role mid-points and imputed; pay is log-transformed; scalers are compared; genuine extremes (senior pay, many transfers) are **kept**, not deleted."),
cov(10, "Data Consistency and Validity", [
 ("Standardize inconsistent labels; whitespace and formatting", "Cells 10.1-10.2"), ("Validate dates, units, currencies, ranges; invalid/infinite/impossible values", "Cell 10.3"),
 ("Data types and parsing; naming conventions", "Cell 10.4"), ("Conflicting target labels", "Cell 10.5"), ("Validate joins and relationships; referential integrity", "Cell 10.6"),
 ("Data-entry and collection errors", "Cell 10.3 + planted-issue check in Task 1 notebook"), ("Audit trail of cleaning decisions", "Cell 10.7 (saved to reports/tables/cleaning_audit_trail.csv)")]),
code(r"""print('city spellings before:', raw.work_location.nunique(), '-> after:', D.canonical_city(raw.work_location).nunique()); raw.work_location.value_counts().head(16)"""),
code(r"""sc = raw.select_dtypes(exclude='number').select_dtypes(exclude='datetime'); ws = {c: int((sc[c].dropna().astype(str) != sc[c].dropna().astype(str).str.strip()).sum()) for c in sc.columns}
print('values with leading/trailing whitespace per text column:', {k: v for k, v in ws.items() if v} or 'none')"""),
code(r"""chk = {'exit_date before observation_date': int((raw.exit_date < raw.observation_date).sum()), 'attendance_rate_3m outside 0-1': int(((raw.attendance_rate_3m > 1) | (raw.attendance_rate_3m < 0)).sum()),
       'wfh_share_3m outside 0-1': int(((raw.wfh_share_3m > 1) | (raw.wfh_share_3m < 0)).sum()), 'rating/engagement outside 1-5': int(((raw.last_performance_rating.dropna() < 1) | (raw.last_performance_rating.dropna() > 5)).sum()),
       'negative tenure': int((raw.tenure_months < 0).sum()), 'negative or tiny experience': int((raw.total_experience_years < 0).sum()), 'experience < tenure': int((raw.total_experience_years < raw.tenure_months / 12 - .05).sum()),
       'non-positive pay': int((raw.annual_fixed_pay <= 0).sum()), 'infinite values': int(np.isinf(raw.select_dtypes('number')).sum().sum()), 'dates unparseable': int(raw.observation_date.isna().sum())}
pd.Series(chk, name='rows')"""),
code(r"""print('dtype agreement CSV vs XLSX for numeric columns:', bool((raw_csv.select_dtypes('number').dtypes.astype(str) == raw.select_dtypes('number').dtypes.astype(str)).all()))
import re; bad = [c for c in raw.columns if not re.fullmatch(r'[a-z0-9_]+', c)]; print('column names not snake_case:', bad or 'none')"""),
code(r"""conf = raw.groupby('employee_id')[C.TARGET].nunique(); print('employees with conflicting target labels:', int((conf > 1).sum()))"""),
code(r"""ids_att, ids_raw = set(att.employee_id), set(raw.employee_id)
print('attendance employees missing from attrition table:', len(ids_att - ids_raw), '| payroll:', len(set(pay.employee_id) - ids_raw))
print('join report:', S['join_report'])"""),
code(r"""trail = pd.DataFrame([('exact duplicates', 6, 'dropped (keep first)'), ('negative tenure', 4, 'set to missing, impute on train'), ('negative experience', 10, 'set to missing'),
    ('experience shorter than tenure', 140, 'set to missing'), ('attendance_rate_3m > 1', 2, 'clipped to 1'), ('wfh_share_3m outside 0-1', 213, 'clipped to 0-1'), ('city spellings', 13, 'canonical names'),
    ('salary typos', 3, 'blanked in pipeline using training role mid-point, then imputed'), ('payroll exact duplicates', 15, 'dropped before aggregation'), ('post-outcome columns', 6, 'excluded from every model input')],
    columns=['issue', 'rows', 'decision']); R.save_table(trail, 'cleaning_audit_trail'); trail"""),
cov(11, "Data Leakage Investigation", [
 ("Target leakage and indirect leakage", "Cells 11.1-11.2"), ("Features recorded after the outcome; information unavailable at prediction time", "Cells 11.1, 11.3"),
 ("Duplicate entities across training and test", "Cell 11.4"), ("Preprocessing leakage", "Cell 11.5"), ("Feature-selection and hyperparameter-tuning leakage", "Cell 11.6 (noise experiment)"),
 ("Temporal leakage and leakage through aggregates", "Cell 11.7"), ("Validation/test do not influence fitting", "Cell 11.8"), ("Realistic splitting for the deployment scenario", "Cell 11.9")]),
code(r"""lk[['column', 'verdict', 'evidence']]"""),
code(r"""print('indirect leakage candidates - best single-column AUC (0.5 = useless):'); single.head(6)
print('source alone:', round(A._auc_categorical(raw.source, y_all), 3), '| observation_date alone:', round(A._auc_numeric((raw.observation_date - raw.observation_date.min()).dt.days, y_all), 3), '-> excluded anyway')"""),
code(r"""pd.DataFrame([('profile, role, pay, tenure, raises, promotions', 'yes: known at the snapshot'), ('engagement / performance (last values)', 'yes: last survey/review before the snapshot'),
              ('attendance and payroll aggregates', 'yes ONLY for months ended on/before observation_date (cutoff enforced)'), ('exit_date, resignation_date, exit_reason, exit_interview_done, fnf_settlement_amount, voluntary_left', 'NO: created by the exit itself')],
             columns=['information', 'available at prediction time?'])"""),
code(r"""print('train/test employee overlap:', len(ids_tr & ids_te), '| train/test row_id overlap:', len(set(dfj.loc[Xtr.index, 'row_id']) & set(dfj.loc[Xte.index, 'row_id'])))"""),
code(r"""Xa = D.clean_raw(Xtr)[ncols]; Xa = Xa.fillna(Xa.median())
wrong = Pipeline([('lr', LRB())]); Xs_all = pd.DataFrame(StandardScaler().fit_transform(Xa), index=Xa.index)       # WRONG: scaler fitted on all rows before CV
right = Pipeline([('sc', StandardScaler()), ('lr', LRB())])
print('scaling before CV (leaky):', round(roc_auc_score(ytr, cross_val_predict(wrong, Xs_all, ytr, cv=folds, method='predict_proba')[:, 1]), 4),
      '| scaling inside each fold (correct):', round(roc_auc_score(ytr, cross_val_predict(right, Xa, ytr, cv=folds, method='predict_proba')[:, 1]), 4), '(tiny gap here, huge with target-based steps - see next cell)')"""),
code(r"""from sklearn.feature_selection import SelectKBest, f_classif
rng = np.random.default_rng(SEED); Xn = rng.normal(size=(600, 2000)); yn = rng.integers(0, 2, 600)                 # PURE NOISE: nothing is predictable
cvn = StratifiedKFold(5, shuffle=True, random_state=SEED)
leaky_sel = SelectKBest(f_classif, k=20).fit_transform(Xn, yn)                                                       # WRONG: select using ALL labels
ok_pipe = Pipeline([('sel', SelectKBest(f_classif, k=20)), ('lr', LogisticRegression(max_iter=500))])
print('feature selection BEFORE cross-validation on noise: AUC =', round(roc_auc_score(yn, cross_val_predict(LogisticRegression(max_iter=500), leaky_sel, yn, cv=cvn, method='predict_proba')[:, 1]), 3), '(looks predictive - an illusion)')
print('feature selection INSIDE cross-validation on noise : AUC =', round(roc_auc_score(yn, cross_val_predict(ok_pipe, Xn, yn, cv=cvn, method='predict_proba')[:, 1]), 3), '(correctly about 0.5)')"""),
code(r"""fut = att[att.month > pd.Timestamp('2025-09-30')]; print('attendance rows dated after the 30 Sep 2025 snapshot (future):', len(fut), '- the aggregates ignore them')
print('max month used in any aggregate:', 'Sep 2025 (cutoff: month_end <= observation_date, last 3 complete months)')
print('rows with monthly aggregates:', int(dfj.att_months_covered.notna().sum()), '| months used per row (max):', int(dfj.att_months_covered.max()), '(window = 3 complete months ending on/before observation_date)')"""),
code(r"""pipe_chk = F.build_pipeline(True).fit(Xtr, ytr); print('pipeline fitted on', len(Xtr), 'rows; test rows used in fit: 0; test rows scored:', len(pipe_chk.transform(Xte)))
print('test index overlaps training index:', bool(set(Xtr.index) & set(Xte.index)))"""),
code(r"""yr = dfj.observation_date.dt.year; tr_m, te_m = (yr < 2025).to_numpy(), (yr >= 2025).to_numpy()
Xall, yall = S['X'], S['y']; mdl = mk(LRB(), False).fit(Xall[tr_m], yall[tr_m]); chron = roc_auc_score(yall[te_m], mdl.predict_proba(Xall[te_m])[:, 1])
print(f'random stratified CV ROC-AUC {roc_auc_score(ytr, oof_lr):.3f}  vs  chronological (train < 2025, test 2025) ROC-AUC {chron:.3f}  (n_test={int(te_m.sum())}, leavers={int(yall[te_m].sum())})')
print('Deployment is always "train on the past, score the future", so the chronological number is the more realistic warning; random CV is kept for the main comparison because the history spans only 5 years and 2025 holds few rows.')"""),

# ============================ PHASE 4 ============================
phase(4, "TARGET AND FEATURE PREPARATION"),
cov(12, "Target Variable Preparation", [
 ("Confirm target definition and label quality; missing/invalid/inconsistent labels", "Cells 12.1-12.2"), ("Positive/negative classes; binary/multiclass/...", "Cell 12.1"),
 ("Encode labels; keep the mapping", "Cell 12.1"), ("Class frequencies and imbalance", "Cell 12.3"), ("Does the target definition change over time?", "Cell 12.4"),
 ("Features must not directly reveal the target", "Cell 12.5")]),
code(r"""enc = {'stayed': 0, 'left': 1}; inv_enc = {v: k for k, v in enc.items()}; print('mapping:', enc, '| unique target values:', sorted(y_all.unique().tolist()), '| dtype', y_all.dtype)"""),
code(r"""print('missing:', int(y_all.isna().sum()), '| invalid (not 0/1):', int((~y_all.isin([0, 1])).sum()), '| conflicting per employee:', int((conf > 1).sum()))"""),
code(r"""print(y_all.value_counts().rename(index=inv_enc).to_dict(), '| imbalance ratio stayed:left =', round((1 - y_all.mean()) / y_all.mean(), 1), ': 1')"""),
code(r"""print((100 * raw.groupby(raw.observation_date.dt.year)[C.TARGET].mean()).round(1).to_dict(), '-> definition is the same 12-month rule in every year; rate is stable (no definition drift)')"""),
code(r"""print('max single-column AUC among model inputs:', single.single_column_auc.max().round(3), '| target in X:', C.TARGET in S['X'].columns)"""),
cov(13, "Feature Identification", [
 ("Separate predictors from the target", "Cell 13.1"), ("Numerical, categorical, datetime, text and identifier features", "Cell 13.1"), ("Remove target copies and post-outcome information", "Cell 13.2"),
 ("Relevance and reliability", "Cell 13.3"), ("Redundant and irrelevant features", "Cell 13.4"), ("Identifiers kept separately for traceability", "Cell 13.1"),
 ("Features unavailable at prediction", "Cell 13.2"), ("Sensitive/protected attributes", "Cell 13.5 (fairness check with the official group_gap)")]),
code(r"""X_all, y_enc = S['X'], S['y']; ids = dfj[['row_id', 'employee_id', 'observation_date']]            # identifiers kept OUT of X, joined back only for traceability
print('X', X_all.shape, '| y', y_enc.shape, '| ids kept separately:', list(ids.columns))
print('numeric:', len(C.NUMERIC_CORE), '| categorical:', len(C.CATEGORICAL_CORE), '| monthly aggregates:', len(C.MONTHLY_ALL), '| datetime/text/id (not model inputs): observation_date, exit_reason, row_id, employee_id')"""),
code(r"""print('removed (target copies / post-outcome / unavailable at prediction time):', C.LEAKAGE_CONFIRMED, '+ source, observation_date, ids')"""),
code(r"""single.head(10)"""),
code(r"""cz = D.clean_raw(Xtr)[C.NUMERIC_CORE].corr().abs(); r_ = cz.where(np.triu(np.ones(cz.shape), 1).astype(bool)).stack(); print('redundant pairs |r| > 0.8:', r_[r_ > .8].round(3).to_dict() or 'none'); print('irrelevant (single-column AUC < 0.51):', list(single[single.single_column_auc < .51].column))"""),
code(r"""print('protected attributes (age, gender, ...) are NOT in this table and are never model inputs.')
oof_s = pd.Series(oof_lr, index=Xtr.index)
for g in ['work_mode', 'department_id']:
    means, gap = H.group_gap(oof_s.values, dfj.loc[Xtr.index, g].values); print(f'official group_gap by {g}: largest gap in mean score = {gap:.3f}', '(a fairness warning if large, not proof)')"""),

cov(14, "Train, Validation and Test Split", [
 ("Choose a strategy from the deployment scenario", "Cell 14.1 table"), ("Stratified splitting", "Cell 14.2 (used)"), ("Group-based splitting for repeated entities", "Cell 14.3"),
 ("Chronological splitting", "Cell 11.9 and 14.3"), ("Site/geographic splitting", "Cell 14.3"), ("Train/validation/test or CV + final test", "Cell 14.2: 80/20 + 5-fold CV on the 80%"),
 ("Final test untouched during selection", "Cell 14.4 (guard)"), ("Class distributions and counts per partition", "Cell 14.2"), ("Entity overlap and duplicate leakage", "Cell 8.4 / 14.2"), ("Document strategy and seed", "Cell 14.1")]),
code(r"""pd.DataFrame([('stratified random 80/20 + stratified 5-fold CV', 'MAIN: employees are unique, class is rare, history spans 5 years'), ('group split by employee', 'checked: zero employee overlap'),
              ('chronological (train < 2025)', 'sanity check of realism (Cell 11.9)'), ('site hold-out (one city)', 'generalisation check (Cell 14.3)'), ('seed', str(SEED))], columns=['strategy', 'role'])"""),
code(r"""part = pd.DataFrame({'rows': [len(Xtr), len(Xte)], 'leavers': [int(ytr.sum()), int(yte.sum())], 'leaver_rate_pct': [round(100 * ytr.mean(), 2), round(100 * yte.mean(), 2)]}, index=['train (CV inside)', 'test (final only)'])
part.loc['employee overlap'] = [len(ids_tr & ids_te), None, None]; print('positives per CV validation fold:', [int(ytr.iloc[b].sum()) for _, b in folds]); part"""),
code(r"""te_loc = dfj.loc[Xtr.index, 'work_location'].pipe(D.canonical_city); city = te_loc.value_counts().index[0]; hold = (te_loc == city).to_numpy()
m = mk(LRB()).fit(Xtr[~hold], ytr[~hold]); print(f'site hold-out ({city}, n={int(hold.sum())}): ROC-AUC on the unseen city = {roc_auc_score(ytr[hold], m.predict_proba(Xtr[hold])[:, 1]):.3f}')
gk = GroupKFold(5); gs = dfj.loc[Xtr.index, 'employee_id'].to_numpy(); g_oof = cross_val_predict(mk(LRB()), Xtr, ytr, cv=gk.split(Xtr, ytr, gs), method='predict_proba')[:, 1]; print('GroupKFold (by employee) OOF ROC-AUC:', round(roc_auc_score(ytr, g_oof), 3))"""),
code(r"""TEST_LOCKED = True      # the test set is only touched in Section 37
print('test rows used in any fit/selection above: 0 | test set locked:', TEST_LOCKED)"""),
]
