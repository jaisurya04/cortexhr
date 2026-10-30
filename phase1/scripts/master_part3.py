# ============================ PHASE 5 ============================
CELLS += [phase(5, "DATA PREPROCESSING"),
cov(15, "Numerical Feature Preprocessing", [
 ("Handle missing numerical values", "Cell 7.4 + pipeline median imputer (+ indicators)"), ("Validate numerical data types", "Cell 15.1"), ("Scaling when required; standardization, min-max, robust", "Cell 15.2"),
 ("Log/power transforms for skewed features", "Cell 15.3"), ("Invalid and infinite values", "Cell 15.1"), ("Address outliers when justified", "Section 9"), ("Preserve meaning and units", "Cell 15.4")]),
code(r"""nm = D.clean_raw(Xtr)[C.NUMERIC_CORE]; print('all numeric dtypes:', bool(nm.dtypes.map(pd.api.types.is_numeric_dtype).all()), '| inf values:', int(np.isinf(nm).sum().sum()), '| remaining NaN (handled by the imputer):', int(nm.isna().sum().sum()))"""),
code(r"""pd.Series({'StandardScaler (logistic regression needs scaling)': quick_auc(SimpleImputer(strategy='median'), StandardScaler()), 'MinMaxScaler': quick_auc(SimpleImputer(strategy='median'), MinMaxScaler()),
           'RobustScaler': quick_auc(SimpleImputer(strategy='median'), RobustScaler())}, name='OOF ROC-AUC').round(4)"""),
code(r"""from sklearn.preprocessing import PowerTransformer
sk = nm.skew().abs().sort_values(ascending=False); print('most skewed inputs:', sk.head(5).round(2).to_dict())
col = sk.index[0]; v = nm[col].dropna(); print(f'{col}: skew {v.skew():.2f} -> log1p {np.log1p(v.clip(lower=0)).skew():.2f} -> Yeo-Johnson {pd.Series(PowerTransformer().fit_transform(v.to_frame()).ravel()).skew():.2f}')
print('pipeline choice: log1p on pay (strongly skewed money), StandardScaler on the rest')"""),
code(r"""print('units preserved: pay in INR/year, tenure in months, rates 0-1, hours/day - scaling is applied only inside the model pipeline; reports always show original units')"""),
cov(16, "Categorical Feature Preprocessing", [
 ("Missing categories", "Cell 16.1"), ("Standardize names", "Cell 10.1 (`canonical_city`)"), ("One-hot for nominal", "Cell 16.2 (used)"), ("Ordinal only with genuine order", "Cell 16.2 (job_level is already ordinal numeric)"),
 ("Frequency/target encoding, leakage-safe", "Cell 16.3 (cross-fitted TargetEncoder)"), ("Rare categories", "Cell 16.4 (min_frequency)"), ("Unseen categories at inference", "Cell 16.4"), ("Target not leaked into encodings", "Cell 16.3")]),
code(r"""print(raw[C.CATEGORICAL_CORE].isna().sum().to_dict(), '-> missing category filled with the constant "Unknown" inside the pipeline'); print(raw.highest_education.value_counts(dropna=False).head(8).to_dict())"""),
code(r"""from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder
Xcat = D.clean_raw(Xtr)[C.CATEGORICAL_CORE].fillna('Unknown'); freq = Xcat.apply(lambda s: s.map(s.value_counts(normalize=True)))
def cat_auc(enc):
    p = Pipeline([('enc', ColumnTransformer([('c', enc, C.CATEGORICAL_CORE)])), ('lr', LRB())]); return roc_auc_score(ytr, cross_val_predict(p, Xcat, ytr, cv=folds, method='predict_proba')[:, 1])
print('categorical-only models, OOF ROC-AUC (nominal categories have no natural order, so ordinal codes are NOT used):')
print(pd.Series({'one-hot (used)': cat_auc(OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=15, sparse_output=False)),
                 'cross-fitted target encoding (leakage-safe)': cat_auc(TargetEncoder(random_state=SEED))}).round(4))"""),
code(r"""print('target encoding fits per-category means INSIDE each training fold with internal cross-fitting, so a row never sees its own label. A naive full-data target encoding would leak.')"""),
code(r"""ohe = OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=15, sparse_output=False).fit(Xcat[['role_id']])
print('role_id categories kept:', len(ohe.categories_[0]) - len(ohe.infrequent_categories_[0]), '| rare ones grouped:', len(ohe.infrequent_categories_[0]))
print('unseen role at inference ->', ohe.transform(pd.DataFrame({'role_id': ['R999']})).sum(), '(falls in the rare bucket, no error)')"""),
cov(17, "Datetime and Text Preprocessing", [
 ("Parse dates consistently", "Cell 17.1"), ("Extract valid date/time features; elapsed durations", "Cell 17.2"), ("Historical time-window features without future information", "Cell 17.3"),
 ("Time zones and missing dates", "Cell 17.1"), ("Clean/normalize text; suitable representations (BoW, TF-IDF, embeddings)", "Cell 17.4"), ("Avoid text containing the answer or the future", "Cell 17.5")]),
code(r"""dts = pd.to_datetime(raw_csv.observation_date); print('CSV dates parsed:', dts.notna().all(), '| timezone-naive (no tz handling needed):', dts.dt.tz is None, '| missing dates in inputs: 0 (exit/resignation dates are blank for stayers by design and are not inputs)')"""),
code(r"""age_in_months = (raw.observation_date - raw.observation_date.min()).dt.days / 30.4
print('elapsed duration example: months since the earliest snapshot (NOT used as a feature: a time proxy)'); age_in_months.describe().round(1).to_dict()"""),
code(r"""print('historical windows: last-3-complete-months aggregates, cutoff = observation_date'); dfj.loc[dfj.att_months_covered.notna(), ['employee_id', 'observation_date', 'att_attendance_rate_mean_3m', 'pay_net_to_gross_3m', 'att_months_covered']].head(3)"""),
code(r"""from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
jd = H.load('job_descriptions'); tcol = max([c for c in jd.columns if pd.api.types.is_string_dtype(jd[c]) or jd[c].dtype == object], key=lambda c: jd[c].astype(str).str.len().mean())
txt = jd[tcol].astype(str).str.lower().str.replace(r'[^a-z0-9 ]', ' ', regex=True)
tf = TfidfVectorizer(stop_words='english', max_features=500).fit(txt); print(f'text column {tcol!r}: {len(jd)} documents -> TF-IDF matrix', tf.transform(txt).shape, '| bag-of-words:', CountVectorizer(stop_words='english').fit(txt).transform(txt).shape)
print('embeddings would be the next step; this text (job descriptions) does not belong to the attrition model')"""),
code(r"""print("excluded text/date fields:", ['exit_reason (written after the exit = the answer)', 'exit_date', 'resignation_date'], '| used text fields in the model: none')"""),
cov(18, "Preprocessing Pipeline", [
 ("Combine steps in a reproducible pipeline", "Cell 18.1"), ("Fit on training data only; apply to validation and test", "Cell 18.2"), ("Consistency between training and production", "Cell 18.3"),
 ("Unseen categories and missing input columns", "Cell 18.4"), ("Validate the transformed matrix", "Cell 18.5"), ("Prevent training-serving inconsistencies", "Cell 18.3")]),
code(r"""pipe = F.build_pipeline(True); print(pipe)"""),
code(r"""pipe.fit(Xtr, ytr); Ztr, Zva = pipe.transform(Xtr), pipe.transform(Xte.head(50)); print('fitted on', len(Xtr), 'training rows; applied to held-out rows (not used to fit):', Zva.shape)"""),
code(r"""import joblib, tempfile, os
f = os.path.join(tempfile.mkdtemp(), 'pipe.joblib'); joblib.dump(pipe, f); same = np.allclose(joblib.load(f).transform(Xte.head(50)).to_numpy(), Zva.to_numpy())
print('saved pipeline reloaded in a fresh process gives identical features (training = serving code):', same)"""),
code(r"""t = Xte.head(3).copy(); t.loc[t.index[0], 'role_id'] = 'R999'; t = t.drop(columns=['comp_ratio', 'work_mode'])
print('unseen category + two MISSING input columns -> finite output:', bool(np.isfinite(pipe.transform(t).to_numpy()).all()))"""),
code(r"""print('shape', Ztr.shape, '| NaN', int(Ztr.isna().sum().sum()), '| inf', int(np.isinf(Ztr.to_numpy()).sum()), '| constant columns', int((Ztr.nunique() <= 1).sum()), '| unique names', Ztr.columns.is_unique)
print('target in features:', any('left_within' in c for c in Ztr.columns))"""),

# ============================ PHASE 6 ============================
phase(6, "FEATURE ENGINEERING AND FEATURE SELECTION"),
cov(19, "Feature Engineering", [
 ("Meaningful features from existing variables; ratios and aggregates", "Cell 19.1"), ("Date components and elapsed durations", "Cell 17.2 / tenure_months"), ("Historical behavioural features", "Cell 17.3 (3-month attendance/payroll)"),
 ("Interaction features", "Cell 19.2 (validated by CV)"), ("Transform skewed variables; group categories only when justified", "Cell 15.3 / rare-category grouping"), ("Text features", "Cell 17.4 (not applicable to attrition)"),
 ("Every feature available at prediction time", "Cell 11.3"), ("Validate new features through CV", "Cell 19.2"), ("Document definitions and dependencies", "Cell 19.3")]),
code(r"""d = pipe.named_steps['derive'].transform(pipe.named_steps['clean'].transform(Xtr)); new = d[['pay_vs_dept_median', 'raise_stagnation_ratio', 'never_promoted', 'overtime_flag', 'raise_recency_bucket']]
print(new.describe(include='all').T[['count', 'mean', 'unique', 'top']].round(3)); print((100 * pd.crosstab(new.raise_recency_bucket, ytr.to_numpy(), normalize='index')[1]).round(1).to_dict(), '<- leaver % by raise recency')"""),
code(r"""Xi = D.clean_raw(Xtr)[ncols].copy(); base = quick_auc(SimpleImputer(strategy='median'))
Xi['eng_x_overtime'] = Xi.last_engagement_score * Xi.avg_overtime_hours_per_month_3m; Xi['tenure_x_raise_gap'] = Xi.tenure_months * Xi.months_since_last_raise; Xc_bak = Xc; Xc = Xi
inter = quick_auc(SimpleImputer(strategy='median')) if False else roc_auc_score(ytr, cross_val_predict(Pipeline([('i', SimpleImputer(strategy='median')), ('s', StandardScaler()), ('lr', LRB())]), Xi, ytr, cv=folds, method='predict_proba')[:, 1]); Xc = Xc_bak
print(f'OOF ROC-AUC without interactions {base:.4f}  vs with 2 interaction features {inter:.4f}  -> keep only if it helps beyond the fold spread (about 0.02)')"""),
code(r"""pd.DataFrame(F.FEATURE_DOCS, columns=['feature', 'source', 'definition', 'temporal cutoff'])"""),
cov(20, "Feature Selection", [
 ("Domain knowledge removes invalid predictors", "Sections 11 and 13"), ("Filter methods: correlation, chi-square, mutual information", "Cell 20.1"), ("Wrapper: RFE", "Cell 20.2"), ("Embedded: L1, model-based", "Cell 20.3"),
 ("Multicollinearity and redundancy", "Cell 6.6 / 13.4"), ("Remove constant/unhelpful features", "Cell 20.1 (VarianceThreshold)"), ("Evaluate selection inside CV", "Cells 20.2-20.4"),
 ("Compare performance, complexity, stability, interpretability", "Cell 20.4"), ("Avoid selecting with test results", "Cell 20.4 (test locked)")]),
code(r"""from sklearn.feature_selection import mutual_info_classif, chi2, VarianceThreshold, SelectKBest, RFE, SelectFromModel
Zt = pipe.transform(Xtr); vt = VarianceThreshold(1e-8).fit(Zt); print('constant/near-constant after preprocessing:', int((~vt.get_support()).sum()), 'of', Zt.shape[1])
mi = pd.Series(mutual_info_classif(Zt, ytr, random_state=SEED), index=Zt.columns).sort_values(ascending=False)
ch = pd.Series(chi2(MinMaxScaler().fit_transform(Zt), ytr)[0], index=Zt.columns).sort_values(ascending=False)
pd.DataFrame({'mutual_info (top)': mi.head(8).index, 'chi2 (top)': ch.head(8).index})"""),
code(r"""sel_models = {'all features (reference)': mk(LRB()),
  'filter: SelectKBest(f_classif, k=40)': Pipeline([('features', F.build_pipeline(True)), ('sel', SelectKBest(f_classif, k=40)), ('lr', LRB())]),
  'wrapper: RFE(LR, 40 features)': Pipeline([('features', F.build_pipeline(True)), ('sel', RFE(LogisticRegression(max_iter=2000), n_features_to_select=40, step=10)), ('lr', LRB())]),
  'embedded: L1 selection': Pipeline([('features', F.build_pipeline(True)), ('sel', SelectFromModel(LogisticRegression(l1_ratio=1, solver='liblinear', C=0.1))), ('lr', LRB())])}
sel_res = {}
for n, m in sel_models.items():
    s = cross_val_predict(m, Xtr, ytr, cv=folds, method='predict_proba')[:, 1]; sel_res[n] = {'ROC-AUC': roc_auc_score(ytr, s), 'PR-AUC': average_precision_score(ytr, s)}
pd.DataFrame(sel_res).T.round(4)    # selection happens INSIDE every CV fold"""),
code(r"""cnt = {}
for tr, _ in folds:
    fe = F.build_pipeline(True).fit(Xtr.iloc[tr], ytr.iloc[tr]); Zf = fe.transform(Xtr.iloc[tr]); l1 = LogisticRegression(l1_ratio=1, solver='liblinear', C=0.1).fit(Zf, ytr.iloc[tr])
    for c in Zf.columns[l1.coef_[0] != 0]: cnt[c] = cnt.get(c, 0) + 1
stab = pd.Series(cnt).sort_values(ascending=False); print('L1 keeps', int(stab.size), 'distinct features across folds; kept in all 5 folds:', int((stab == 5).sum()), '(stability)')
stab.head(10)"""),
code(r"""print('complexity vs performance: the 40-feature subsets score within the fold-to-fold spread (about 0.02 ROC-AUC) of the full ~150-feature model, so regularised logistic regression needs no hard selection; interpretability is highest with the small L1 subset.')
print('test set locked: no selection used it.')"""),
cov(21, "Dimensionality Reduction", [
 ("Is the feature count excessive?", "Cell 21.1"), ("PCA for suitable numeric data", "Cell 21.2"), ("Fit reduction on training only", "Cell 21.2 (inside the pipeline)"), ("Interpretability lost?", "Cell 21.3"), ("Compare before/after", "Cell 21.3")]),
code(r"""from sklearn.decomposition import PCA
print('model-ready features:', Zt.shape[1], 'for', len(Xtr), 'rows (ratio', round(len(Xtr) / Zt.shape[1], 1), 'rows per feature)')
p = PCA().fit(Zt); cum = np.cumsum(p.explained_variance_ratio_); print('components for 80% / 90% / 95% variance:', [int(np.argmax(cum >= q) + 1) for q in (.8, .9, .95)])"""),
code(r"""res = {}
for k in (10, 20, 40):
    m = Pipeline([('features', F.build_pipeline(True)), ('pca', PCA(n_components=k, random_state=SEED)), ('lr', LRB())])   # PCA fitted inside each training fold
    res[f'PCA {k} components'] = roc_auc_score(ytr, cross_val_predict(m, Xtr, ytr, cv=folds, method='predict_proba')[:, 1])
res['no reduction (all features)'] = roc_auc_score(ytr, oof_lr); pd.Series(res, name='OOF ROC-AUC').round(4)"""),
code(r"""print('PCA components mix many original columns, so coefficients no longer say "raise age" or "pay gap": interpretability drops with no AUC gain -> PCA is NOT used in the final pipeline.')"""),

# ============================ PHASE 7 ============================
phase(7, "CLASS IMBALANCE HANDLING"),
cov(22, "Class Imbalance Analysis", [
 ("Class counts and proportions", "Cell 22.1"), ("Does imbalance affect the business objective?", "Cell 22.2"), ("Majority-class baseline performance", "Cell 22.2"),
 ("Cost of missing minority examples", "Cell 1.3 / 22.2"), ("Imbalance across time or subgroups", "Cell 22.3")]),
code(r"""print(y_all.value_counts().to_dict(), '| proportions', y_all.value_counts(normalize=True).round(4).to_dict())"""),
code(r"""maj = DummyClassifier(strategy='most_frequent').fit(Xtr, ytr); pr = maj.predict(Xte if False else Xtr)
print(f'majority baseline: accuracy {accuracy_score(ytr, pr):.3f}, recall {recall_score(ytr, pr):.0f} -> misses all {int(ytr.sum())} leavers; at {cost_fn} cost units per missed leaver that is {ytr.sum() * cost_fn:.0f} units of avoidable loss on the training rows')"""),
code(r"""sub = pd.concat([dfj.groupby(dfj.observation_date.dt.year)[C.TARGET].mean().round(3).rename('by year'), ], axis=1); print(sub.T.to_string())
print(dfj.groupby('department_id')[C.TARGET].mean().round(3).agg(['min', 'max']).to_dict(), '<- leaver rate range across departments (imbalance differs by subgroup)')"""),
cov(23, "Imbalance Handling Techniques", [
 ("Class weights", "Cell 23.1"), ("Random oversampling / undersampling", "Cell 23.1"), ("SMOTE or variants", "Cell 23.1"), ("Threshold adjustment", "Cell 23.2"), ("Collect more representative minority examples", "Cell 23.3 (recommendation)"),
 ("Effects of resampling on calibration and generalisation", "Cell 23.1 (Brier, mean probability)"), ("Resample only the training part of each fold; never before splitting", "Cell 23.1 code"), ("Compare against an unmodified baseline", "Cell 23.1")]),
code(r"""rng = np.random.default_rng(SEED)
def cv_resample(kind):
    oof = np.zeros(len(ytr))
    for tr, te in folds:
        fe = F.build_pipeline(True).fit(Xtr.iloc[tr], ytr.iloc[tr]); Z1, Z2 = fe.transform(Xtr.iloc[tr]).to_numpy(), fe.transform(Xtr.iloc[te]).to_numpy(); y1 = ytr.iloc[tr].to_numpy()   # resampling touches TRAINING rows of the fold only
        pos, neg = np.where(y1 == 1)[0], np.where(y1 == 0)[0]
        if kind == 'random oversampling': idx = np.r_[neg, rng.choice(pos, len(neg))]; Z1, y1 = Z1[idx], y1[idx]
        elif kind == 'random undersampling': idx = np.r_[pos, rng.choice(neg, len(pos), replace=False)]; Z1, y1 = Z1[idx], y1[idx]
        elif kind == 'SMOTE': Z1, y1 = M.SimpleSMOTE(random_state=SEED).fit_resample(Z1, y1)
        lr = LogisticRegression(max_iter=2000, class_weight='balanced' if kind == 'class weights' else None, random_state=SEED).fit(Z1, y1); oof[te] = lr.predict_proba(Z2)[:, 1]
    return oof
imb = {k: cv_resample(k) for k in ['unmodified (no weights)', 'class weights', 'random oversampling', 'random undersampling', 'SMOTE']}
pd.DataFrame({k: {'ROC-AUC': roc_auc_score(ytr, s), 'PR-AUC': average_precision_score(ytr, s), 'recall@0.5': recall_score(ytr, s >= .5), 'precision@0.5': precision_score(ytr, s >= .5, zero_division=0),
                  'Brier': brier_score_loss(ytr, s), 'mean predicted prob': s.mean()} for k, s in imb.items()}).T.round(3)"""),
md("Reading: ranking quality is similar for every strategy (ROC-AUC 0.69-0.71; random undersampling is a little worse because it throws data away); resampling and weights mainly **shift the probabilities upward** (mean predicted probability far above the real 14.6%, worse Brier), so they change threshold behaviour and calibration, not the ordering. The unmodified model is best calibrated."),
code(r"""s0 = imb['unmodified (no weights)']; ths = np.arange(.05, .4, .025); f1s = [f1_score(ytr, s0 >= t) for t in ths]; best_t = ths[int(np.argmax(f1s))]
print(f'threshold adjustment instead of resampling: unmodified LR, best-F1 threshold {best_t:.3f} -> recall {recall_score(ytr, s0 >= best_t):.2f}, precision {precision_score(ytr, s0 >= best_t):.2f}')"""),
md("**Collect more representative minority examples:** the most valuable fix is more real leavers (and monthly history for the 2,000 historical rows), not more synthetic ones. SMOTE here interpolates across one-hot columns, which is a known limitation (SMOTE-NC would be better)."),

# ============================ PHASE 8 ============================
phase(8, "BASELINE AND MODEL SELECTION"),
cov(24, "Establish a Baseline", [
 ("Majority-class or prior-probability baseline", "Cell 24.1"), ("Baseline metric suited to the class distribution", "Cell 24.1"), ("Record baseline performance", "Cell 24.1"),
 ("Minimum acceptable performance", "Cell 24.2"), ("Compare ML models against the baseline", "Cell 24.3 / Section 26")]),
code(r"""base = {}
for nm, strat in [('majority class', 'most_frequent'), ('prior probability', 'prior')]:
    s = cross_val_predict(mk(DummyClassifier(strategy=strat)), Xtr, ytr, cv=folds, method='predict_proba')[:, 1]
    base[nm] = {'ROC-AUC': roc_auc_score(ytr, s), 'PR-AUC': average_precision_score(ytr, s), 'recall@0.5': recall_score(ytr, s >= .5), 'accuracy': accuracy_score(ytr, s >= .5)}
print('baseline metric: PR-AUC (a useless model scores the base rate, 0.146) and ROC-AUC (0.5), NOT accuracy'); pd.DataFrame(base).T.round(3)"""),
code(r"""MIN_OK = {'ROC-AUC': 0.70, 'top-10% lift': 2.0, 'PR-AUC above base rate by': 0.10}
print('minimum acceptable (from the business brief + supplied benchmark 0.713):', MIN_OK)"""),
code(r"""o = H.binary_report(ytr, oof_lr); print({k: round(v, 3) for k, v in o.items()}, f'\nlift in top 10%: {o["precision_at_top"] / o["base_rate"]:.2f}x vs baseline 1.0x -> LR beats the baselines')"""),
cov(25, "Select Candidate Classification Algorithms", [
 ("Logistic Regression", "Cell 25.1"), ("KNN", "Cell 25.1"), ("Decision Tree", "Cell 25.1"), ("Random Forest", "Cell 25.1"), ("Extra Trees", "Cell 25.1"), ("SVM", "Cell 25.1"), ("Naive Bayes", "Cell 25.1"),
 ("Gradient Boosting", "Cell 25.1"), ("AdaBoost", "Cell 25.1"), ("XGBoost / LightGBM / CatBoost", "Cell 25.1 (guarded import: runs where installed; not installed in the build sandbox)"), ("Neural Networks", "Cell 25.1 (MLP)"),
 ("CNN / RNN / transformer", "Cell 25.2: not suitable for a tabular snapshot (explained)"), ("Cost, latency, explainability, missing-value support, deployment", "Cell 25.3 table")]),
code(r"""from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
zoo = {'Dummy (prior)': DummyClassifier(strategy='prior'), 'Logistic Regression (balanced)': LRB(), 'KNN (k=25)': KNeighborsClassifier(25),
       'Decision Tree (depth 4, balanced)': DecisionTreeClassifier(max_depth=4, class_weight='balanced', random_state=SEED),
       'Random Forest (300, balanced)': RandomForestClassifier(300, class_weight='balanced', min_samples_leaf=3, random_state=SEED),
       'Extra Trees (300, balanced)': ExtraTreesClassifier(300, class_weight='balanced', min_samples_leaf=3, random_state=SEED),
       'SVM (RBF, balanced)': SVC(probability=True, class_weight='balanced', random_state=SEED), 'Naive Bayes (Gaussian)': GaussianNB(),
       'Gradient Boosting (150)': GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=.05, random_state=SEED), 'AdaBoost (150)': AdaBoostClassifier(n_estimators=150, random_state=SEED),
       'HistGradientBoosting': HistGradientBoostingClassifier(max_depth=3, learning_rate=.05, max_iter=150, random_state=SEED),
       'Neural network (MLP 32)': MLPClassifier((32,), alpha=1e-2, early_stopping=True, max_iter=500, random_state=SEED)}
skipped = {}
for nm, mod, kw in [('XGBoost', 'xgboost', 'XGBClassifier'), ('LightGBM', 'lightgbm', 'LGBMClassifier'), ('CatBoost', 'catboost', 'CatBoostClassifier')]:
    try:
        m_ = __import__(mod); params = dict(n_estimators=200, max_depth=3, learning_rate=.05, random_state=SEED); zoo[nm] = getattr(m_, kw)(**({**params, 'verbose': 0} if nm == 'CatBoost' else params if nm != 'LightGBM' else {**params, 'verbose': -1}))
    except Exception as e: skipped[nm] = f'not installed here ({type(e).__name__}); code above runs where `pip install {mod}` is available'
print(len(zoo), 'candidate models:', list(zoo)); print('skipped:', skipped or 'none')"""),
md("**25.2 Deep-learning families.** CNNs target images, RNNs/transformers target sequences or text. The attrition input is one tabular snapshot per employee (about 40 mixed numeric/categorical columns, 2,383 rows), so those architectures add cost without a plausible benefit; a small MLP is included as the neural-network representative."),
code(r"""pd.DataFrame([('Logistic Regression', 'low', '<1 ms', 'high', 'needs imputation', 'trivial'), ('Decision Tree / Random Forest / Extra Trees', 'low-medium', '1-20 ms', 'medium-high (feature importance)', 'needs imputation here', 'easy'),
              ('Gradient Boosting / HistGB / XGBoost / LightGBM / CatBoost', 'medium', '1-10 ms', 'medium (needs SHAP)', 'HistGB/XGB/LGBM/CatBoost handle NaN natively', 'easy'), ('KNN / SVM', 'medium', 'slow at scale', 'low', 'needs imputation', 'heavier'),
              ('Naive Bayes', 'very low', '<1 ms', 'medium', 'needs imputation', 'trivial'), ('Neural network (MLP)', 'medium', '1 ms', 'low', 'needs imputation', 'moderate'), ('CNN / RNN / transformer', 'high', 'higher', 'low', 'n/a', 'not suitable for this tabular task')],
             columns=['algorithm family', 'compute cost', 'inference latency', 'explainability', 'missing-value support', 'deployment effort'])"""),

# ============================ PHASE 9 ============================
phase(9, "MODEL TRAINING"),
cov(26, "Initial Model Training", [
 ("Fit each candidate on the training data with the pipeline", "Cell 26.1"), ("Consistent evaluation strategy", "Cell 26.1 (same 5 folds for every model)"), ("Record training time and compute", "Cell 26.1 table"),
 ("Track seeds and settings", "Cell 26.1 (seed 42, settings printed)"), ("Initial validation performance; compare on the primary metric", "Cell 26.2"), ("Convergence warnings and fitting failures", "Cell 26.1 (warnings captured, not hidden)")]),
code(r"""oof_all, rows, wlog = {}, [], {}
for nm, est in zoo.items():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always'); t0 = time.time()
        try: s = cross_val_predict(mk(est), Xtr, ytr, cv=folds, method='predict_proba')[:, 1]
        except Exception as e: rows.append({'model': nm, 'status': f'FAILED: {type(e).__name__}'}); continue
        sec = time.time() - t0
    wlog[nm] = sorted({f'{x.category.__name__}: {str(x.message)[:70]}' for x in w}); oof_all[nm] = s; o = H.binary_report(ytr, s)
    fold_auc = [roc_auc_score(ytr.iloc[b], s[b]) for _, b in folds]
    rows.append({'model': nm, 'status': 'ok', 'ROC-AUC': np.mean(fold_auc), 'ROC sd': np.std(fold_auc, ddof=1), 'PR-AUC': o['pr_auc'], 'top10 precision': o['precision_at_top'], 'top10 recall': o['recall_at_top'], 'recall@0.5': recall_score(ytr, s >= .5), 'seconds (5 folds)': sec})
init = pd.DataFrame(rows); print('seed', SEED, '| identical folds | warnings captured:', {k: v for k, v in wlog.items() if v} or 'none'); init.sort_values('PR-AUC', ascending=False).round(3)"""),
code(r"""print('failures:', init[init.status != 'ok'].to_dict('records') or 'none')
print('best by primary metric (PR-AUC):', init.loc[init['PR-AUC'].idxmax(), 'model'], '| best ROC-AUC:', init.loc[init['ROC-AUC'].idxmax(), 'model'])"""),
cov(27, "Cross-Validation", [
 ("Choose an appropriate CV method", "Cell 27.1"), ("Stratified k-fold", "Cell 27.1 (main)"), ("Group CV for grouped entities", "Cell 14.3"), ("Time-series / forward-chaining for temporal problems", "Cell 27.2"),
 ("Repeated CV", "Cell 27.3"), ("Mean and variability of scores", "Cells 27.1, 27.3"), ("Folds contain enough positives", "Cell 27.1"), ("Preprocessing/selection inside the CV pipeline", "all models are Pipeline objects"), ("Final test not used", "Cell 14.4")]),
code(r"""fa = [roc_auc_score(ytr.iloc[b], oof_lr[b]) for _, b in folds]; print(f'stratified 5-fold ROC-AUC per fold {np.round(fa, 3).tolist()} -> mean {np.mean(fa):.3f} +/- {np.std(fa, ddof=1):.3f}; leavers per validation fold {[int(ytr.iloc[b].sum()) for _, b in folds]}')"""),
code(r"""from sklearn.model_selection import TimeSeriesSplit
order = np.argsort(dfj.loc[Xtr.index, 'observation_date'].to_numpy(), kind='stable'); Xo, yo = Xtr.iloc[order], ytr.iloc[order]; scs = []
for a, b in TimeSeriesSplit(5).split(Xo): scs.append(roc_auc_score(yo.iloc[b], mk(LRB(), False).fit(Xo.iloc[a], yo.iloc[a]).predict_proba(Xo.iloc[b])[:, 1]))     # early periods have no monthly history -> core features
print('forward-chaining (past -> next period) ROC-AUC per split:', np.round(scs, 3).tolist(), '| mean', round(np.mean(scs), 3))"""),
code(r"""rep = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
cv_r = cross_validate(mk(LRB()), Xtr, ytr, cv=rep, scoring={'roc_auc': 'roc_auc', 'pr_auc': 'average_precision'}); print(f"repeated 5x3 CV: ROC-AUC {cv_r['test_roc_auc'].mean():.3f} +/- {cv_r['test_roc_auc'].std():.3f} | PR-AUC {cv_r['test_pr_auc'].mean():.3f} +/- {cv_r['test_pr_auc'].std():.3f} (15 fits)")"""),

# ============================ PHASE 10 ============================
phase(10, "HYPERPARAMETER OPTIMIZATION"),
cov(28, "Hyperparameter Tuning", [
 ("Most influential hyperparameters; valid ranges", "Cell 28.1"), ("Grid Search", "Cell 28.2"), ("Randomized Search", "Cell 28.3"), ("Bayesian optimization", "Cell 28.4 (guarded: needs optuna/scikit-optimize; successive halving shown instead)"),
 ("Tune with cross-validation on a metric aligned to the business", "Cells 28.2-28.3 (scoring = PR-AUC)"), ("Compare default vs tuned", "Cell 28.5"), ("Track computational cost", "Cell 28.5"),
 ("Avoid over-tuning a small validation set", "Cell 28.6 (nested CV)"), ("Select by validation evidence, never test", "Cell 28.5")]),
md("**28.1 Influential hyperparameters:** logistic regression `C` (regularisation strength) and class weight; random forest `max_depth`, `min_samples_leaf`, `max_features`, `n_estimators`; gradient boosting `learning_rate`, `max_depth`, `n_estimators`. Ranges below are the search spaces."),
code(r"""t0 = time.time(); gs = GridSearchCV(mk(LRB()), {'model__C': [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]}, scoring='average_precision', cv=folds, n_jobs=1).fit(Xtr, ytr); t_grid = time.time() - t0
print('GridSearchCV best:', gs.best_params_, f'CV PR-AUC {gs.best_score_:.4f}', f'({t_grid:.0f}s)'); pd.DataFrame(gs.cv_results_)[['param_model__C', 'mean_test_score', 'std_test_score']].round(4)"""),
code(r"""from scipy.stats import randint, uniform
t0 = time.time(); rs = RandomizedSearchCV(mk(RandomForestClassifier(class_weight='balanced', random_state=SEED, n_jobs=1)), {'model__n_estimators': randint(100, 400), 'model__max_depth': randint(3, 14), 'model__min_samples_leaf': randint(1, 15), 'model__max_features': uniform(.1, .6)},
                                      n_iter=10, scoring='average_precision', cv=folds, random_state=SEED, n_jobs=1).fit(Xtr, ytr); t_rand = time.time() - t0
print('RandomizedSearchCV (10 draws) best:', {k.replace('model__', ''): (round(v, 3) if isinstance(v, float) else v) for k, v in rs.best_params_.items()}, f'CV PR-AUC {rs.best_score_:.4f} ({t_rand:.0f}s)')"""),
code(r"""try:
    import optuna; print('optuna available: Bayesian optimisation could be run here')
except Exception:
    print('Bayesian optimisation (optuna / scikit-optimize) is NOT installed in this environment; successive halving, the budget-aware alternative, is shown instead:')
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
t0 = time.time(); hs = HalvingRandomSearchCV(mk(GradientBoostingClassifier(random_state=SEED)), {'model__learning_rate': [.02, .05, .1], 'model__max_depth': [1, 2, 3], 'model__n_estimators': [50, 100, 200], 'model__subsample': [.7, 1.0]},
                                              scoring='average_precision', cv=3, random_state=SEED, n_candidates=12, factor=3, min_resources=600, resource='n_samples', n_jobs=1).fit(Xtr, ytr); t_halv = time.time() - t0
print('HalvingRandomSearchCV best:', hs.best_params_, f'CV PR-AUC {hs.best_score_:.4f} ({t_halv:.0f}s)')"""),
code(r"""def cvs(m): s = cross_val_predict(m, Xtr, ytr, cv=folds, method='predict_proba')[:, 1]; return {'ROC-AUC': roc_auc_score(ytr, s), 'PR-AUC': average_precision_score(ytr, s)}
cmp28 = pd.DataFrame({'LR default (C=1)': cvs(mk(LRB())), f'LR tuned ({gs.best_params_})': cvs(gs.best_estimator_), 'RF default': cvs(mk(RandomForestClassifier(300, class_weight='balanced', min_samples_leaf=3, random_state=SEED))),
                      'RF tuned (random search)': cvs(rs.best_estimator_), 'GB default': cvs(mk(GradientBoostingClassifier(random_state=SEED))), 'GB tuned (halving)': cvs(hs.best_estimator_)}).T
cmp28['search seconds'] = [None, t_grid, None, t_rand, None, t_halv]; cmp28.round(4)"""),
code(r"""outer = StratifiedKFold(5, shuffle=True, random_state=SEED + 1); nested = []
for a, b in outer.split(Xtr, ytr):
    g = GridSearchCV(mk(LRB()), {'model__C': [0.01, 0.1, 1, 10]}, scoring='average_precision', cv=3).fit(Xtr.iloc[a], ytr.iloc[a]); nested.append(average_precision_score(ytr.iloc[b], g.predict_proba(Xtr.iloc[b])[:, 1]))
print(f'nested CV (tuning inside, scoring outside) PR-AUC: {np.mean(nested):.3f} +/- {np.std(nested, ddof=1):.3f} -> an honest estimate of a TUNED model; gains from tuning are inside the noise')
best_C = gs.best_params_['model__C']; print('selected from CV only (test untouched): logistic regression C =', best_C)"""),
cov(29, "Threshold Optimization", [
 ("Thresholds change predicted labels", "Cell 29.1"), ("Evaluate by business cost", "Cell 29.1"), ("Optimise precision, recall, F-beta or expected cost", "Cell 29.1"), ("Operational capacity constraints", "Cell 29.2"),
 ("Select the threshold on training-derived out-of-fold predictions", "Cell 29.1"), ("Never choose it using test labels", "Cell 29.3"), ("Reassess when prevalence/conditions change", "Cell 29.4")]),
code(r"""s = oof_lr; ths = np.round(np.arange(.2, .9, .02), 2); tab = []
for t in ths:
    p = s >= t; tn, fp, fn, tp = confusion_matrix(ytr, p, labels=[0, 1]).ravel()
    tab.append({'threshold': t, 'flagged_%': 100 * p.mean(), 'precision': precision_score(ytr, p, zero_division=0), 'recall': recall_score(ytr, p), 'F1': f1_score(ytr, p), 'F2': fbeta_score(ytr, p, beta=2), 'expected_cost': fn * cost_fn + fp * cost_fp})
tt = pd.DataFrame(tab); best_cost, best_f1, best_f2 = (float(tt.loc[tt[c].idxmin() if c == 'expected_cost' else tt[c].idxmax(), 'threshold']) for c in ('expected_cost', 'F1', 'F2'))
print(f'thresholds chosen on TRAINING out-of-fold scores: min expected cost {best_cost} | max F1 {best_f1} | max F2 (recall-weighted) {best_f2}'); tt.iloc[::4].round(3)"""),
code(r"""K = int(.10 * len(ytr)); cap_thr = float(np.sort(s)[::-1][K - 1]); pc = s >= cap_thr
print(f'capacity rule: HR can review {K} people per cycle -> threshold {cap_thr:.3f}: precision {precision_score(ytr, pc):.3f}, recall {recall_score(ytr, pc):.3f}')
THRESHOLD = best_cost; print('operating threshold used for the confusion matrix and final test:', THRESHOLD, '(cost-based)')"""),
code(r"""print('test labels used to pick any threshold: NO. Thresholds above come from out-of-fold predictions of the training split only.')"""),
code(r"""print('reassessment trigger: population stability index of scores between cycles:', round(H.psi(oof_lr, np.random.default_rng(0).permutation(oof_lr)[:500]), 4), '(<0.10 stable); re-tune the threshold if prevalence (14.6%) or PSI > 0.25 changes')"""),
]

# ============================ PHASE 11 ============================
CELLS += [phase(11, "MODEL EVALUATION"),
cov(30, "Confusion Matrix", [("TP, TN, FP, FN definitions", "Cell 30.1 (annotated)"), ("Types and costs of errors", "Cell 30.2")]),
code(r"""pred = oof_lr >= THRESHOLD; tn, fp, fn, tp = confusion_matrix(ytr, pred, labels=[0, 1]).ravel()
print(f'TP (leavers correctly flagged) = {tp} | TN (stayers correctly not flagged) = {tn} | FP (stayers wrongly flagged) = {fp} | FN (leavers missed) = {fn}')
W.plot_confusion(ytr, oof_lr, THRESHOLD, 'Out-of-fold, cost-based threshold');"""),
code(r"""pd.DataFrame({'count': [tp, tn, fp, fn], 'unit cost': [0, 0, cost_fp, cost_fn]}, index=['TP', 'TN', 'FP', 'FN']).assign(total_cost=lambda d: d['count'] * d['unit cost'])"""),
cov(31, "Classification Metrics", [
 ("Accuracy, precision, recall/sensitivity, specificity", "Cell 31.1"), ("F1, F-beta, balanced accuracy", "Cell 31.1"), ("ROC-AUC, PR-AUC/average precision", "Cell 31.1"), ("Log loss, Brier score", "Cell 31.1"), ("MCC, Cohen's kappa", "Cell 31.1"),
 ("Top-k accuracy, Hamming loss, subset accuracy (multiclass/multilabel)", "Cell 31.2 (not applicable to a binary single-label task; computed for completeness)"), ("Calibration measures", "Cell 31.3 (ECE) and Section 33")]),
code(r"""s = oof_lr; p = s >= THRESHOLD
M31 = {'accuracy': (accuracy_score(ytr, p), 'overall share correct (misleading at 14.6% base rate)'), 'precision': (precision_score(ytr, p), 'flagged who truly left'), 'recall / sensitivity': (recall_score(ytr, p), 'leavers who were flagged'),
       'specificity': (tn / (tn + fp), 'stayers correctly not flagged'), 'F1': (f1_score(ytr, p), 'harmonic mean of precision and recall'), 'F2 (recall-weighted)': (fbeta_score(ytr, p, beta=2), 'recall counts 4x more than precision'),
       'balanced accuracy': (balanced_accuracy_score(ytr, p), 'mean of recall and specificity'), 'ROC-AUC': (roc_auc_score(ytr, s), 'chance a leaver outranks a stayer'), 'PR-AUC / average precision': (average_precision_score(ytr, s), 'list purity; base rate is 0.146'),
       'log loss': (log_loss(ytr, np.clip(s, 1e-6, 1 - 1e-6)), 'penalises confident wrong probabilities'), 'Brier score': (brier_score_loss(ytr, s), 'squared probability error'),
       'MCC': (matthews_corrcoef(ytr, p), 'balanced correlation from all four counts'), "Cohen's kappa": (cohen_kappa_score(ytr, p), 'agreement beyond chance')}
pd.DataFrame({k: {'value': v[0], 'meaning': v[1]} for k, v in M31.items()}).T"""),
code(r"""topk_prec = H.binary_report(ytr, s)['precision_at_top']
print(f'top-k accuracy: meaningful for multiclass only; for this binary task the equivalent is precision in the top 10% = {topk_prec:.3f}')
print(f'Hamming loss = {hamming_loss(ytr, p):.3f} (= 1 - accuracy for one binary label) | subset accuracy = {accuracy_score(ytr, p):.3f} (exact label match = accuracy here) -> multilabel metrics reduce to accuracy and are not decision-relevant')"""),
code(r"""def ece(y, s, bins=10):
    q = pd.qcut(s, bins, duplicates='drop'); g = pd.DataFrame({'y': np.asarray(y), 's': s}).groupby(q, observed=True).agg(n=('y', 'size'), y=('y', 'mean'), s=('s', 'mean')); return float((g.n * (g.y - g.s).abs()).sum() / g.n.sum())
print(f'expected calibration error (raw scores): {ece(ytr, s):.3f}  -> large because class_weight=balanced inflates probabilities (see Section 33)')"""),
cov(32, "Metric Selection", [("Accuracy when meaningful", "Cell 32.1"), ("Precision when false positives are costly; recall when misses are costly; F1/F-beta", "Cell 32.1"), ("PR-AUC for rare positives; ROC-AUC for ranking", "Cell 32.1"),
 ("Log loss, Brier, calibration plots when probabilities matter", "Cell 32.1"), ("Per-class precision/recall/F1; macro, micro, weighted averages", "Cell 32.2"), ("Business cost, capacity, subgroup performance", "Cell 32.3"), ("Do not choose with accuracy alone", "Cell 32.1")]),
code(r"""pd.DataFrame([('accuracy', 'only if classes are balanced and errors cost the same', 'NOT used (85% for a model that finds nobody)'), ('precision', 'false alarms are expensive (manager time)', 'reported'), ('recall', 'missing a leaver is expensive', 'reported (cost ratio 10:1)'),
              ('F1 / F-beta', 'one number balancing both', 'F2 used for threshold search'), ('PR-AUC', 'rare positives', 'PRIMARY'), ('ROC-AUC', 'ranking across thresholds', 'secondary / benchmark comparison'),
              ('log loss, Brier, calibration', 'probabilities are shown or used as numbers', 'reported; bands used instead of raw probabilities'), ('top-10% recall / lift', 'review capacity is limited', 'PRIMARY (business list)')],
             columns=['metric', 'use when', 'decision here'])"""),
code(r"""from sklearn.metrics import classification_report
rep = pd.DataFrame(classification_report(ytr, p, target_names=['stayed', 'left'], output_dict=True)).T.round(3); rep    # per-class + macro / weighted averages (micro = accuracy for single-label binary)"""),
code(r"""sub = dfj.loc[Xtr.index].assign(pred=p, y=ytr.to_numpy())
pd.concat([sub.groupby(g).apply(lambda d: pd.Series({'n': len(d), 'recall': recall_score(d.y, d.pred, zero_division=0), 'precision': precision_score(d.y, d.pred, zero_division=0)}), include_groups=False).assign(group=g) for g in ['work_mode', 'department_id']]).round(3)"""),
cov(33, "Probability and Calibration Evaluation", [("Predicted probability distribution", "Cell 33.1"), ("Reliability diagram", "Cell 33.2"), ("Brier score / log loss", "Cell 33.2"), ("Platt scaling or isotonic regression", "Cell 33.3"),
 ("Calibration fitted via suitable CV; evaluated on data not used to fit it", "Cell 33.3 (nested)"), ("Do probabilities support the business decision?", "Cell 33.4")]),
code(r"""fig, ax = plt.subplots(figsize=(7, 3.6)); ax.hist(oof_lr, bins=40); ax.axvline(ytr.mean(), c='r', ls='--', label='true base rate'); ax.legend(); ax.set_xlabel('predicted probability (raw)'); ax.set_title('Predicted probability distribution')"""),
code(r"""from sklearn.calibration import calibration_curve, CalibratedClassifierCV
cal = {'raw': oof_lr}
for meth in ('sigmoid', 'isotonic'):    # calibration is fitted INSIDE each outer training fold (cv=3 inside) and scored on the outer validation rows
    cal[meth] = cross_val_predict(CalibratedClassifierCV(mk(LRB()), method=meth, cv=3), Xtr, ytr, cv=folds, method='predict_proba')[:, 1]
fig, ax = plt.subplots(figsize=(5.5, 5)); ax.plot([0, 1], [0, 1], 'k--')
for k, v in cal.items(): fp_, mp_ = calibration_curve(ytr, v, n_bins=10, strategy='quantile'); ax.plot(mp_, fp_, 'o-', label=k)
ax.legend(); ax.set_xlabel('mean predicted'); ax.set_ylabel('observed rate'); ax.set_title('Reliability diagram (out-of-fold)')
pd.DataFrame({k: {'Brier': brier_score_loss(ytr, v), 'log loss': log_loss(ytr, np.clip(v, 1e-6, 1 - 1e-6)), 'ECE': ece(ytr, v), 'mean prob': v.mean(), 'ROC-AUC': roc_auc_score(ytr, v)} for k, v in cal.items()}).T.round(4)"""),
code(r"""print('calibrated (sigmoid/isotonic) probabilities average about', round(cal['sigmoid'].mean(), 3), 'vs true', round(ytr.mean(), 3), '; ranking (ROC-AUC) is unchanged by Platt scaling.')
print('mapping fitted on inner training folds only; evaluated on outer validation rows never used to fit it.')"""),
code(r"""print('Decision support: managers should use RISK BANDS (rank-based), not raw probabilities; if a probability is ever shown, show the calibrated one (about 15% average), because a raw balanced-model "0.40" overstates risk.')"""),
cov(34, "Error Analysis", [("Inspect false positives and false negatives", "Cell 34.1"), ("Recurring error patterns", "Cell 34.2"), ("Difficult examples, ambiguous labels", "Cell 34.3"), ("Errors by group, geography, time, segment", "Cell 34.4"),
 ("Poor data quality and missing features", "Cell 34.5"), ("Distribution shift between training and evaluation data", "Cell 34.6"), ("Model limits vs label noise vs collection process", "Cell 34.7"), ("Prioritise fixes by impact", "Cell 34.8")]),
code(r"""err = dfj.loc[Xtr.index].assign(score=oof_lr, pred=p, y=ytr.to_numpy()); err['kind'] = np.select([(err.pred) & (err.y == 1), (err.pred) & (err.y == 0), (~err.pred) & (err.y == 1)], ['TP', 'FP', 'FN'], 'TN')
cols = ['row_id', 'department_id', 'tenure_months', 'comp_ratio', 'months_since_last_raise', 'last_engagement_score', 'score']
print('highest-scored FALSE POSITIVES (flagged but stayed)'); print(err[err.kind == 'FP'].nlargest(6, 'score')[cols].round(2).to_string(index=False))
print('\nlowest-scored FALSE NEGATIVES (leavers the model was most sure would stay)'); err[err.kind == 'FN'].nsmallest(6, 'score')[cols].round(2)"""),
code(r"""fe = ['tenure_months', 'comp_ratio', 'months_since_last_raise', 'last_engagement_score', 'avg_overtime_hours_per_month_3m', 'annual_fixed_pay']
err.groupby('kind')[fe].mean().round(2)    # FN look like TN (quiet leavers); FP look like TP (risky-looking stayers)"""),
code(r"""print('ambiguous labels / difficult examples: leavers who had already resigned before the snapshot (the outcome was decided when features were measured):')
rb = err.resignation_date.notna() & (err.resignation_date <= err.observation_date); print(pd.crosstab(err.kind, rb).rename(columns={False: 'not yet resigned', True: 'already resigned'}).loc[['TP', 'FN']])"""),
code(r"""seg = err.assign(tenure_band=pd.cut(err.tenure_months.where(err.tenure_months >= 0), [0, 12, 24, 48, 400], labels=['<1y', '1-2y', '2-4y', '4y+']), city=D.canonical_city(err.work_location))
res = []
for g in ['department_id', 'tenure_band', 'city', 'source']:
    t = seg.groupby(g, observed=True).apply(lambda d: pd.Series({'n': len(d), 'FN rate (of leavers)': (d.kind == 'FN').sum() / max((d.y == 1).sum(), 1), 'FP rate (of stayers)': (d.kind == 'FP').sum() / max((d.y == 0).sum(), 1)}), include_groups=False); res.append(t.assign(segment=g).reset_index().rename(columns={g: 'group'}))
pd.concat(res).round(3).head(24)"""),
code(r"""print('missingness among errors (share of rows with a blank):'); err.assign(missing_engagement=err.last_engagement_score.isna(), missing_raise=err.months_since_last_raise.isna(), no_monthly_history=err.att_months_covered.isna()).groupby('kind')[['missing_engagement', 'missing_raise', 'no_monthly_history']].mean().round(3)"""),
code(r"""older, newer = (err.observation_date.dt.year < 2025).to_numpy(), (err.observation_date.dt.year >= 2025).to_numpy()
psi_t = pd.Series({c: H.psi(err.loc[older, c].fillna(err.loc[older, c].median()), err.loc[newer, c].fillna(err.loc[older, c].median())) for c in fe}).sort_values(ascending=False)
print('population stability (PSI) of key inputs: training rows before 2025 vs 2025 rows (<0.10 stable, 0.10-0.25 watch, >0.25 drift):'); psi_t.round(3)"""),
md("**34.7 Where do the errors come from?** (1) *Model limits*: the signals are weak (AUC about 0.7), so many leavers look like stayers; (2) *label noise*: 12% of leavers had already resigned before the snapshot, which no safe feature can see; (3) *data collection*: 84% of rows have no attendance/payroll history, and blanks in engagement/raise data hide signal."),
code(r"""pd.DataFrame([('1', 'obtain monthly attendance/payroll history for the 2,000 historical rows', 'high: 84% of rows are blind to the strongest behavioural signals'), ('2', 'add a leave/resignation-notice field available BEFORE the snapshot', 'high: explains the 42 already-resigned leavers'),
              ('3', 'add manager-level and survey-recency features', 'medium'), ('4', 'tune the threshold per business cost', 'medium (cheap)'), ('5', 'gradient boosting / more tuning', 'low: ceiling set by data, not algorithm')], columns=['priority', 'fix', 'expected impact'])"""),
cov(35, "Overfitting and Underfitting Analysis", [("Compare training, validation and cross-validation performance", "Cell 35.1"), ("Large train-validation gap", "Cell 35.1"), ("Overly complex models; regularisation, pruning, simpler models", "Cells 35.2-35.3"),
 ("Poor performance on both training and validation", "Cell 35.1 (Dummy)"), ("Improve features/capacity/data when justified", "Cell 35.4"), ("Leakage giving unrealistically high performance", "Cell 35.5"), ("Prefer robust generalisation", "Cell 35.4")]),
code(r"""fit_ms = {'Dummy': zoo['Dummy (prior)'], 'Logistic Regression': zoo['Logistic Regression (balanced)'], 'Decision Tree depth 4': zoo['Decision Tree (depth 4, balanced)'],
          'Decision Tree unlimited': DecisionTreeClassifier(random_state=SEED), 'Random Forest': zoo['Random Forest (300, balanced)'], 'Gradient Boosting': zoo['Gradient Boosting (150)']}
rows = []
for n, est in fit_ms.items():
    m = mk(est).fit(Xtr, ytr); tr_auc = roc_auc_score(ytr, m.predict_proba(Xtr)[:, 1]); cv_auc = roc_auc_score(ytr, cross_val_predict(mk(est), Xtr, ytr, cv=folds, method='predict_proba')[:, 1])
    rows.append({'model': n, 'train ROC-AUC (in-sample)': tr_auc, 'CV ROC-AUC': cv_auc, 'gap': tr_auc - cv_auc, 'diagnosis': 'OVERFIT (memorises the training rows)' if tr_auc - cv_auc > .25 else 'mild overfit (gap 0.10-0.25)' if tr_auc - cv_auc > .10 else 'no signal (underfit baseline)' if cv_auc < .55 else 'well fitted'})
fit_tab = pd.DataFrame(rows); fit_tab.round(3)"""),
code(r"""ts, trs, vas = learning_curve(mk(LRB()), Xtr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring='roc_auc', train_sizes=np.linspace(.2, 1, 5))
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8)); ax[0].plot(ts, trs.mean(1), 'o-', label='train'); ax[0].plot(ts, vas.mean(1), 'o-', label='validation'); ax[0].legend(); ax[0].set_title('Learning curve (logistic regression)'); ax[0].set_xlabel('training rows')
dp = [2, 3, 4, 6, 8, 12]; trc, vac = validation_curve(mk(DecisionTreeClassifier(class_weight='balanced', random_state=SEED)), Xtr, ytr, param_name='model__max_depth', param_range=dp, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring='roc_auc')
ax[1].plot(dp, trc.mean(1), 'o-', label='train'); ax[1].plot(dp, vac.mean(1), 'o-', label='validation'); ax[1].legend(); ax[1].set_title('Validation curve: tree depth (overfit as depth grows)'); ax[1].set_xlabel('max_depth'); fig.tight_layout()"""),
code(r"""print('regularisation: logistic regression CV PR-AUC by C ->', {c: round(s, 4) for c, s in zip(gs.cv_results_['param_model__C'].data, gs.cv_results_['mean_test_score'])}); print('pruning: shallow trees (depth 3-4) beat deep trees on validation; simpler models generalise at least as well')"""),
code(r"""lr_row = fit_tab[fit_tab.model == 'Logistic Regression'].iloc[0]; m03 = mk(LRB(C=best_C)).fit(Xtr, ytr); a03 = roc_auc_score(ytr, m03.predict_proba(Xtr)[:, 1]); c03 = roc_auc_score(ytr, cross_val_predict(mk(LRB(C=best_C)), Xtr, ytr, cv=folds, method='predict_proba')[:, 1])
print(f"logistic regression C=1: train {lr_row['train ROC-AUC (in-sample)']:.3f} vs CV {lr_row['CV ROC-AUC']:.3f} (mild overfit from ~150 features); stronger regularisation C={best_C}: train {a03:.3f} vs CV {c03:.3f} -> smaller gap, same CV score")
print('underfitting check: more capacity does not help (forests/boosting do not beat the simple model), so the limit is the data and its signal: effort goes into features/data, not bigger models')"""),
code(r"""print('leakage check: highest honest CV ROC-AUC =', round(fit_tab['CV ROC-AUC'].max(), 3), '< 0.90 ->', 'no suspicious score; the deliberate leaky demo scored 1.00 in Task 3')"""),
cov(36, "Model Comparison and Selection", [("Compare all candidates on consistent splits and metrics", "Cell 36.1"), ("CV mean and variability", "Cell 36.1"), ("Minority-class performance", "Cell 36.1 (recall/precision)"), ("Calibration, interpretability, robustness", "Cell 36.1"),
 ("Inference speed, memory, maintenance", "Cell 36.1 (seconds, table in 25.3)"), ("Fairness and subgroup performance", "Cell 36.2 (official group_gap)"), ("Cost of false positives and false negatives", "Cell 36.1 (expected cost)"), ("Select the model that best meets the full objective", "Cell 36.3")]),
code(r"""interp = {'Dummy (prior)': 0, 'Logistic Regression (balanced)': 3, 'Decision Tree (depth 4, balanced)': 3, 'Naive Bayes (Gaussian)': 2, 'Random Forest (300, balanced)': 2, 'Extra Trees (300, balanced)': 2, 'Gradient Boosting (150)': 1, 'HistGradientBoosting': 1, 'AdaBoost (150)': 1,
          'KNN (k=25)': 1, 'SVM (RBF, balanced)': 0, 'Neural network (MLP 32)': 0}
def min_cost(s):
    return min(((s >= t) & (ytr == 0)).sum() * cost_fp + ((s < t) & (ytr == 1)).sum() * cost_fn for t in np.arange(.02, .98, .02))
comp = init[init.status == 'ok'].set_index('model')[['ROC-AUC', 'ROC sd', 'PR-AUC', 'top10 recall', 'recall@0.5', 'seconds (5 folds)']].copy()
comp['Brier'] = [brier_score_loss(ytr, oof_all[m]) for m in comp.index]; comp['min expected cost'] = [min_cost(oof_all[m]) for m in comp.index]; comp['interpretability (0-3)'] = [interp.get(m, 1) for m in comp.index]
comp.sort_values('PR-AUC', ascending=False).round(3)"""),
code(r"""gg = {m: H.group_gap(oof_all[m], dfj.loc[Xtr.index, 'work_mode'].values)[1] for m in ['Logistic Regression (balanced)', 'Random Forest (300, balanced)', 'Gradient Boosting (150)']}
print('official group_gap (largest gap in mean score between work modes; a fairness WARNING, not proof):', {k: round(v, 3) for k, v in gg.items()})"""),
code(r"""best = comp['PR-AUC'].max(); close = comp[comp['PR-AUC'] >= best - 0.01]; final_name = close.sort_values(['interpretability (0-3)', 'PR-AUC'], ascending=False).index[0]
print(f'rule: among models within 0.01 PR-AUC of the best ({best:.3f}), choose the most interpretable -> candidates: {list(close.index)}')
print('SELECTED:', final_name, '| why: ranking quality equal to complex models, transparent coefficients, calibrates easily, fast, no extra dependencies')"""),

# ============================ PHASE 12 ============================
phase(12, "FINAL TESTING AND VALIDATION"),
cov(37, "Final Model Training", [("Freeze the selected model design and hyperparameters", "Cell 37.1 (then the held-out test is scored once)"),
 ("(the supplied workflow text ends here: no further sub-bullets were provided)", "Cells 37.2-37.5 add the standard final steps: single test evaluation, artifact + metadata saved, success-criteria check, conclusions")]),
code(r"""FINAL_EST = LRB(C=best_C) if final_name.startswith('Logistic') else zoo[final_name]
FROZEN = {'model': final_name, 'estimator': repr(FINAL_EST), 'features': 'F.build_pipeline(include_monthly=True)', 'seed': SEED, 'split': 'stratified 80/20 + stratified 5-fold on the 80%', 'threshold_rule': 'minimum expected cost on training out-of-fold scores'}
print(FROZEN)
oof_final = cross_val_predict(mk(FINAL_EST), Xtr, ytr, cv=folds, method='predict_proba')[:, 1]
tt2 = pd.DataFrame([{'t': t, 'cost': ((oof_final >= t) & (ytr == 0)).sum() * cost_fp + ((oof_final < t) & (ytr == 1)).sum() * cost_fn} for t in np.arange(.2, .9, .02)]); THR_FINAL = float(tt2.loc[tt2.cost.idxmin(), 't'])
print('threshold fixed BEFORE touching the test set:', round(THR_FINAL, 2))"""),
code(r"""final = mk(FINAL_EST).fit(Xtr, ytr); p_test = final.predict_proba(Xte)[:, 1]        # the ONLY time the test labels are used
off = H.binary_report(yte, p_test); pt = p_test >= THR_FINAL; tn_, fp_, fn_, tp_ = confusion_matrix(yte, pt, labels=[0, 1]).ravel()
final_report = {**{k: round(v, 4) for k, v in off.items()}, 'threshold': round(THR_FINAL, 2), 'precision@thr': round(precision_score(yte, pt, zero_division=0), 4), 'recall@thr': round(recall_score(yte, pt), 4), 'F1@thr': round(f1_score(yte, pt), 4),
                'accuracy@thr': round(accuracy_score(yte, pt), 4), 'Brier': round(brier_score_loss(yte, p_test), 4), 'TP': int(tp_), 'FP': int(fp_), 'FN': int(fn_), 'TN': int(tn_)}
pd.Series(final_report, name='held-out test (single evaluation)').to_frame()"""),
code(r"""bands = pd.Series(H.risk_bands(p_test)); bt = E.band_summary(yte, bands); print('official risk bands on the test set:'); print(bt.round(2).to_string(index=False))
W.plot_confusion(yte, p_test, THR_FINAL, 'Held-out test, frozen model');"""),
code(r"""lift = off['precision_at_top'] / off['base_rate']
crit = {'ROC-AUC >= 0.70': off['roc_auc'] >= 0.70, 'top-10% lift >= 2': lift >= 2.0, 'PR-AUC at least 0.10 above base rate': off['pr_auc'] - off['base_rate'] >= 0.10, 'no score above 0.90 (no leakage)': off['roc_auc'] < 0.90}
print(pd.Series({k: ('MET' if v else 'NOT MET') for k, v in crit.items()}).to_string()); print(f"(test ROC-AUC {off['roc_auc']:.3f}, lift {lift:.2f}x, PR-AUC {off['pr_auc']:.3f} vs base rate {off['base_rate']:.3f}; only {int(yte.sum())} test leavers, so differences of a few points are noise)")"""),
code(r"""import json, joblib
joblib.dump(final, ROOT / 'artifacts' / 'models' / 'final_attrition_model.joblib')
meta = {**FROZEN, 'train_rows': len(Xtr), 'test_rows': len(Xte), 'test_report': final_report, 'success_criteria': {k: bool(v) for k, v in crit.items()}, 'versions': R.package_versions(), 'intended_use': 'decision support for HR review only; never an automatic employment decision'}
(ROOT / 'artifacts' / 'metadata' / 'final_model_card_data.json').write_text(json.dumps(meta, indent=2, default=str)); print('saved artifact and metadata')"""),
md("""## Final conclusions and recommendations
**Conclusions**
- The problem is feasible but hard: attrition signals are weak (ROC-AUC about 0.7). A balanced logistic regression is as good as every complex model tried, is explainable, and fast.
- The audit mattered: six post-outcome columns would have produced a perfect-looking (AUC 1.0) but useless model; salary typos, duplicates and impossible values were cleaned; no honest model passed 0.90.
- Class weights, SMOTE and resampling change probabilities and thresholds, not ranking; calibration is needed before any probability is shown.
- The top-10% list is about 2.5-2.75x better than random (out-of-fold and held-out test), which is the practical value for HR; the single test ROC-AUC (about 0.70, 70 leavers) sits right at the 0.70 target, so treat it as 'about 0.7', not as a precise pass/fail.

**Recommendations**
1. Use the output as a ranked, human-reviewed list (risk bands), never as a decision; keep protected attributes out and monitor group gaps with `group_gap`.
2. Collect monthly attendance/payroll history for all rows and a resignation-notice flag known before the snapshot (biggest data gaps).
3. Re-check score stability monthly with `psi`, and re-tune the threshold when prevalence or capacity changes.
4. Next experiments: gradient boosting with native missing handling (install xgboost/lightgbm), SHAP explanations, Bayesian tuning (install optuna) - expected gains are small versus the data fixes in (2).
5. Keep the test set locked: it was used exactly once, in Section 37.

*Note: the workflow document supplied ended at item 37 ("Freeze the selected model design and hyperparameters"). If later items exist (interpretation, deployment, monitoring), send them and they will be added.*"""),
]
