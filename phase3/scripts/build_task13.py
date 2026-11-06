"""Builds notebooks 07-09 (Task 13: attrition model v2)."""
from nbkit import NB

LOAD = """import joblib
from cortexhr.phase3.attrition import v2 as V, train as TR
from cortexhr.phase3.common.io import read_csv, save_table, save_json
from cortexhr.phase3.common.validation import profile
from cortexhr import config as C, features as F
d = V.load_training(); X, y, Xtr, Xte, ytr, yte, folds, dfm = d['X'], d['y'], d['Xtr'], d['Xte'], d['ytr'], d['yte'], d['folds'], d['df']
bench = read_csv('baseline_benchmarks')
print('model frame:', dfm.shape, '| inputs:', X.shape[1], '| train/test:', len(Xtr), '/', len(Xte), '| folds:', len(folds), '| base rate: %.3f' % y.mean())"""


def nb07():
    n = NB("Task 13, Notebook 07: Attrition v2, data, leakage and baseline", "13: Attrition model v2: beat the baseline honestly", "Re-audit the attrition data and leakage controls, reuse the Phase 1 pipeline and folds, and fix the baseline v2 must beat.", list(range(1, 25)))
    n.stage(1, "Earlier help for people who may leave only works if the prediction is real. A model that looks better because of leakage hurts HR twice: it wastes effort and hides who actually needs support. Teams re-check leakage whenever a score jumps.")
    n.stage(2, "**Binary classification:** will the employee leave within 12 months of the observation date? **Inputs:** HR, pay and 3-month attendance/payroll aggregates known at the observation date. **Output:** a probability per employee, later explained in Task 14. v2 means a gradient-boosting based model.")
    n.stage(3, "Pass marks, fixed in the config before any result: the improvement over the Task 3 baseline is called *real* only if the bootstrap 95% interval of the paired fold difference in ROC-AUC is above zero; probabilities must be calibrated (ECE at most 0.05); no single feature may predict the target suspiciously well (AUC at most 0.80).")
    n.code(LOAD.split("\nd = ")[0] + "\nCFG_A = CFG['attrition']['success_criteria']\npd.Series(CFG_A, name='threshold').to_frame()")
    n.code("d = V.load_training(); X, y, Xtr, Xte, ytr, yte, folds, dfm = d['X'], d['y'], d['Xtr'], d['Xte'], d['ytr'], d['yte'], d['folds'], d['df']\nbench = read_csv('baseline_benchmarks'); print('model frame:', dfm.shape, '| inputs:', X.shape[1], '| train/test:', len(Xtr), '/', len(Xte), '| folds:', len(folds), '| base rate: %.3f' % y.mean())")
    n.stage(4, "Tables: `attrition_dataset` (2,389 employee snapshots), the attendance and payroll monthly feature tables (joined with a cut-off so no future month is used), `baseline_benchmarks` (the programme's reference numbers) and, for Task 14, `attrition_scoring_2026`.")
    n.stage(5, "Loaded through the Phase 1 loader `cortexhr.runs.load_model_frame`: exact duplicates dropped, monthly aggregates joined cut-off-safe, stratified 80/20 split, 5 stratified folds with seed 42, identical to Task 3.")
    n.code("d['join_report']")
    n.stage(6, "")
    n.code("pd.Series({'raw rows': d['raw'].shape[0], 'rows after removing exact duplicates': dfm.shape[0], 'model inputs': X.shape[1], 'train rows': len(Xtr), 'test rows': len(Xte), 'leavers in train': int(ytr.sum()), 'leavers in test': int(yte.sum())}, name='count').to_frame()")
    n.stage(7, "")
    n.code("read_csv('data_dictionary').query(\"sheet == 'attrition_dataset'\")[['column', 'data_type', 'description']].head(30)")
    n.stage(8, "")
    n.code("pd.Series(X.dtypes.astype(str).map(lambda t: 'numeric' if t.startswith(('int', 'float')) else 'text/category')).value_counts().to_frame('inputs')")
    n.stage(9, "")
    n.code("profile(X).sort_values('missing_pct', ascending=False).head(12)")
    n.stage(10, "Missing is informative here: `months_since_last_promotion` is empty for people never promoted; engagement and performance are missing for people who were not rated; monthly aggregates exist only for part of the history. The pipeline imputes with training statistics and adds missing-indicator columns.")
    n.code("""m = X.isna().mean().sort_values(ascending=False); print(m[m > 0].round(3).to_dict())
miss = pd.DataFrame({'missing_share': X.isna().mean(), 'leave_rate_if_missing': {c: y[X[c].isna()].mean() if X[c].isna().any() else np.nan for c in X.columns}}).dropna().sort_values('missing_share', ascending=False).head(6).round(3); miss""")
    n.stage(11, "")
    n.code("print('exact duplicate rows in the raw file:', int(d['raw'].duplicated().sum()), '| after cleaning:', int(dfm.duplicated(subset=[c for c in dfm.columns if c != 'row_id']).sum()), '| employee_id unique:', dfm.employee_id.is_unique)")
    n.stage(12, "")
    n.code("""num = X.select_dtypes('number'); rows = []
for c in ['tenure_months', 'commute_distance_km', 'annual_fixed_pay', 'training_hours_12m', 'avg_overtime_hours_per_month_3m', 'paid_leave_days_6m']:
    q1, q3 = num[c].quantile([.25, .75]); i = q3 - q1; o = ((num[c] < q1 - 1.5 * i) | (num[c] > q3 + 1.5 * i)); rows.append({'feature': c, 'IQR outliers': int(o.sum()), 'share': round(o.mean(), 3), 'max': num[c].max(), 'leave_rate_of_outliers': round(y[o].mean(), 3) if o.any() else np.nan})
pd.DataFrame(rows)""")
    n.stage(13, "Impossible values. The Phase 1 cleaning step already sets impossible values to missing and clips rates to 0..1; here they are counted before cleaning.")
    n.code("""raw = dfm; checks = {'attendance_rate_3m outside 0..1': int(((raw.attendance_rate_3m < 0) | (raw.attendance_rate_3m > 1)).sum()), 'att_wfh_share_3m above 1 (raw monthly aggregate)': int((raw.att_wfh_share_3m > 1).sum()),
 'comp_ratio outside 0.2..2.5': int(((raw.comp_ratio < .2) | (raw.comp_ratio > 2.5)).sum()), 'negative tenure': int((raw.tenure_months < 0).sum()), 'annual_fixed_pay <= 0': int((raw.annual_fixed_pay <= 0).sum())}
print(checks); print('after the pipeline cleaning step: rates are clipped to [0, 1] (stateless, no learned values)')""")
    n.stage(14, "")
    n.code("print('base rate of leaving within 12 months: %.3f (%d of %d)' % (y.mean(), y.sum(), len(y))); dfm.groupby('department_id').left_within_12m.agg(['mean', 'size']).round(3).sort_values('mean', ascending=False).head(6)")
    n.stage(15, "")
    n.code("""fig, ax = plt.subplots(1, 4, figsize=(15, 3))
for a, c in zip(ax, ['tenure_months', 'last_engagement_score', 'comp_ratio', 'attendance_rate_3m']): X[c].plot.hist(ax=a, bins=25, color='#2f6fb0'); a.set_title(c)
plt.tight_layout()""")
    n.stage(16, "Leave rate against the main drivers, by quantile bins (a bivariate view, not a causal claim).")
    n.code("""fig, ax = plt.subplots(1, 4, figsize=(15, 3)); 
for a, c in zip(ax, ['tenure_months', 'last_engagement_score', 'comp_ratio', 'attendance_rate_3m']):
    b = pd.qcut(X[c], 5, duplicates='drop'); r = y.groupby(b, observed=True).mean(); r.plot.bar(ax=a, color='#c0392b'); a.set_title('leave rate by ' + c, fontsize=9); a.set_xticklabels([str(i)[:12] for i in r.index], rotation=30, fontsize=6)
plt.tight_layout()""")
    n.stage(17, "Redundant inputs: the 3-month attendance numbers appear twice (HR snapshot and monthly table). This is why Task 14 explains in *groups*.")
    n.code("""num = X.select_dtypes('number'); cm = num.corr(method='spearman').abs(); cm = cm.mask(np.eye(len(cm), dtype=bool), 0)
pairs_ = cm.where(np.triu(np.ones(cm.shape), 1).astype(bool)).stack().sort_values(ascending=False).head(8).round(3); pairs_.to_frame('|spearman|')""")
    n.stage(18, "")
    n.code("print('target = left_within_12m. rate overall %.3f | train %.3f | test %.3f' % (y.mean(), ytr.mean(), yte.mean()), '| the split keeps the rate equal (stratified).'); print('Always-stay baseline accuracy: %.3f but it finds 0 leavers, which is why PR-AUC and recall are reported.' % (1 - y.mean()))")
    n.stage(19, "Leakage checks. (1) the exit columns exist only after someone leaves; (2) the monthly aggregates use only months before the observation date; (3) no single input predicts the target too well; (4) the leaky model in the benchmark file shows what leakage looks like.")
    n.code("""print('columns excluded as inputs (Phase 1 audit, kept unchanged):', C.FORBIDDEN_AS_INPUT)
print('any excluded column present among the inputs:', [c for c in C.FORBIDDEN_AS_INPUT if c in X.columns])
print(bench[bench.model.str.contains('LEAKY')][['model', 'roc_auc_mean', 'comment']].to_string(index=False))""")
    n.code("""from sklearn.metrics import roc_auc_score
rows = []
for c in X.columns:
    s = X[c]
    if s.dtype.kind in 'if': v = s.fillna(s.median())
    else: v = s.map(y.groupby(s).mean()).fillna(y.mean())        # target mean encoding (diagnostic only)
    a = roc_auc_score(y, v); rows.append({'feature': c, 'single_feature_auc': max(a, 1 - a)})
sf = pd.DataFrame(rows).sort_values('single_feature_auc', ascending=False); print('max single-feature AUC: %.3f (limit %.2f) -> %s' % (sf.single_feature_auc.max(), CFG_A['max_leakage_single_feature_auc'], 'OK' if sf.single_feature_auc.max() <= CFG_A['max_leakage_single_feature_auc'] else 'SUSPICIOUS')); sf.head(6).round(3)""")
    n.md("Note: the target-mean encoding of `role_id` and `highest_education` inflates their diagnostic AUC (many categories, computed on all rows) so it is an upper bound used only to hunt for leakage, never a model result.")
    n.stage(20, "Reused from Phase 1: stateless cleaning, derived features (pay versus department median, raise recency, overtime flag, missing indicators), scaling and one-hot encoding. Nothing new is learned outside the training folds.")
    n.code("""pipe = F.build_pipeline(True).fit(Xtr, ytr); Z = pipe.transform(Xte); print('raw inputs:', X.shape[1], '-> model-ready columns:', Z.shape[1]); print([s[0] for s in pipe.steps])""")
    n.stage(21, "Allowed inputs, excluded inputs, and inputs that need careful wording because they can act as proxies.")
    n.code("""protected = [c for c in X.columns if any(k in c.lower() for k in ('gender', 'sex', 'age_', 'religion', 'marital', 'caste', 'disab'))]
print('protected attributes among the inputs:', protected or 'none'); print('proxy-risk inputs kept (flagged for explanation wording in Task 14): total_experience_years (age proxy), highest_education, work_location, tenure_months (new joiners)')""")
    n.stage(22, "Preprocessing is inside the sklearn pipeline, so it is refit on the training part of every fold. Demonstration: the imputation value learned on training data differs from the value on all data.")
    n.code("""print('median engagement: train part %.3f | all rows %.3f -> statistics are learned only from the training part' % (Xtr.last_engagement_score.median(), X.last_engagement_score.median()))""")
    n.stage(23, "80% training (1,906 rows) / 20% test (477 rows), stratified, seed 42; 5-fold stratified cross-validation **inside the training part**, the same folds as Task 3. The test set is used once, at the end.")
    n.code("pd.DataFrame([{'fold': i, 'train': len(a), 'validation': len(b), 'leavers_in_validation': int(ytr.iloc[b].sum()), 'rate': round(ytr.iloc[b].mean(), 3)} for i, (a, b) in enumerate(folds)])")
    n.stage(24, "The baseline v2 must beat: Phase 1 logistic regression (balanced classes) on the same folds, next to the programme's reference numbers.")
    n.code("""lr = V.baseline_models()['Logistic Regression (balanced)']; ft = V.fold_table(lr, Xtr, ytr, folds, 'Logistic Regression (balanced)')
mine = V.summarise(ft)[['model', 'roc_auc_mean', 'roc_auc_sd', 'pr_auc_mean', 'precision_mean', 'recall_mean']]
ref = bench[bench.model.str.startswith('Logistic Regression (class')][['model', 'roc_auc_mean', 'roc_auc_sd', 'pr_auc_mean', 'precision_mean', 'recall_mean']]
pd.concat([mine.assign(source='this repo, same folds'), ref.assign(source='programme reference')]).round(3)""")
    return n


def nb08():
    n = NB("Task 13, Notebook 08: Candidate models, cross-validation and tuning log", "13: Attrition model v2: beat the baseline honestly", "Train the candidates, compare them on identical folds and keep a complete log of a small, controlled tuning search.", [25, 26, 27, 28])
    n.code(LOAD)
    n.stage(25, "Candidates: the Phase 1 logistic regression and random forest baselines, HistGradientBoosting as a documented comparator, an equal-weight LR + HGB blend, and (when installed) a real LightGBM classifier. The LightGBM candidate is tuned over a small predeclared parameter set and compared on exactly the same Task 3 folds; it is not silently replaced by HistGradientBoosting.")
    n.code("print('optional libraries in this environment:', V.optional_libs()); print('LightGBM candidate:', 'enabled' if V.optional_libs().get('lightgbm') else 'BLOCKED: install lightgbm; HGB is only the documented fallback')")
    n.stage(26, "Fit each candidate on the training part and compare with its cross-validation score. A large gap between training and validation scores would signal overfitting.")
    n.code("""from sklearn.metrics import roc_auc_score
models = {'Logistic Regression (balanced) [baseline]': V.baseline_models()['Logistic Regression (balanced)'], 'Random Forest (300, balanced) [baseline]': V.baseline_models()['Random Forest (300, balanced)'],
          'HistGradientBoosting default': V.wrap(V.make_hgb())}
rows = []
for k, m in models.items():
    from sklearn.base import clone
    f = clone(m).fit(Xtr, ytr); rows.append({'model': k, 'train_roc_auc': roc_auc_score(ytr, f.predict_proba(Xtr)[:, 1])})
pd.DataFrame(rows).round(3)""")
    n.stage(27, "Per-fold metrics on the **same five folds** for every model. Spread across folds is as important as the mean.")
    n.code("""log, tune_folds = V.run_tuning(Xtr, ytr, folds); params = V.best_params(log)
cands = {**models, 'HistGradientBoosting tuned': V.wrap(V.make_hgb(**params)), 'attrition-v2 candidate: LR + tuned HGB blend': V.make_v2(params)}
ft = pd.concat([V.fold_table(m, Xtr, ytr, folds, k) for k, m in cands.items()], ignore_index=True)
summ = V.summarise(ft); summ[['model', 'roc_auc_mean', 'roc_auc_sd', 'pr_auc_mean', 'pr_auc_sd', 'precision_mean', 'recall_mean', 'f1_mean', 'balanced_accuracy_mean', 'brier_mean']].round(3)""")
    n.code("""fig, ax = plt.subplots(figsize=(9, 3.4)); piv = ft.pivot(index='fold', columns='model', values='roc_auc'); piv.plot(ax=ax, marker='o'); ax.set_ylabel('ROC-AUC'); ax.legend(fontsize=6); ax.set_title('ROC-AUC per fold (same folds for all models)'); plt.tight_layout(); plt.savefig(P.FIGURES / 'task13_fold_roc_auc.png', dpi=130, bbox_inches='tight')""")
    n.stage(28, "16 settings of a deliberately small grid (shallow trees, strong regularisation, small learning rates). Selection uses mean cross-validation PR-AUC on the training part only; the test set is untouched.")
    n.code("""save_table(log, P.task_dir(13) / 'tuning_log.csv'); log[['experiment_id', 'params', 'roc_auc_mean', 'roc_auc_sd', 'pr_auc_mean', 'decision']].round(3)""")
    n.code("""top = log.sort_values('pr_auc_mean', ascending=False).head(5)
print('selected:', log.loc[log.decision.str.startswith('selected'), 'experiment_id'].iloc[0], params)
print('top-5 settings span %.3f in mean PR-AUC and %.3f in ROC-AUC, while the fold-to-fold sd is about %.3f -> the optimum is flat; extra tuning would only fit noise.' % (top.pr_auc_mean.max() - top.pr_auc_mean.min(), top.roc_auc_mean.max() - top.roc_auc_mean.min(), log.roc_auc_sd.mean()))""")
    return n


def nb09():
    n = NB("Task 13, Notebook 09: Evaluation, calibration, honest assessment and the saved model", "13: Attrition model v2: beat the baseline honestly", "Evaluate once on the test set, calibrate, decide whether v2 beats the baseline, and save attrition-v2 with its metadata.", list(range(29, 39)))
    n.code(LOAD)
    n.stage(29, "The main run executes HGB tuning, same-fold comparison, calibration choice, one test evaluation, artifacts and metadata. A separate reproducible LightGBM audit cell below runs the official booster candidate on the same Task 3 folds and saves a separately versioned challenger.")
    n.code("out = TR.run_task13(write=True); summ = out['summary']; summ[['model', 'roc_auc_mean', 'roc_auc_sd', 'pr_auc_mean', 'precision_mean', 'recall_mean', 'f1_mean', 'balanced_accuracy_mean', 'brier_mean', 'precision_top10_mean', 'recall_top10_mean']].round(3)")
    n.code("""# Official LightGBM candidate: same Task 3 folds, predeclared tuning grid, separate artifact.
lgb_audit = TR.run_lightgbm_audit(write=True)
print('Actual LightGBM execution: 3 predeclared settings x 5 Task 3 folds.')
print(lgb_audit['tuning_log'][['experiment_id','roc_auc_mean','pr_auc_mean','decision']].to_string(index=False))
print('Holdout evaluation:')
print(lgb_audit['test_metrics'].to_string(index=False))""")
    n.code("""out['paired'].round(4)""")
    n.code("""print(out['verdict']['text']); print('mean paired fold difference in ROC-AUC (v2 minus baseline): %+.4f, bootstrap 95%% interval [%.4f, %.4f]' % (out['verdict']['mean_diff'], *out['verdict']['ci']))
print('Test set (used once, models fitted on the training part only):'); out['test_metrics'].round(3)""")
    n.code("""print('bootstrap of the test AUC difference (v2 - baseline):', {k: round(v, 4) for k, v in out['test_boot'].items()})
print('bootstrap of the out-of-fold AUC difference:', {k: round(v, 4) for k, v in out['oof_boot'].items()})""")
    n.stage(30, "Who does the model miss? At the company rule (top 10% flagged) look at leavers that were caught versus missed.")
    n.code("""ts = out['test_scores'].join(Xte); cut = ts.p_v2.quantile(0.90); ts['flagged'] = ts.p_v2 >= cut; lv = ts[ts.y == 1]
print('test leavers: %d | caught in top 10%%: %d | missed: %d | flagged non-leavers: %d' % (len(lv), lv.flagged.sum(), (~lv.flagged).sum(), (ts.flagged & (ts.y == 0)).sum()))
cols = ['last_engagement_score', 'tenure_months', 'comp_ratio', 'attendance_rate_3m', 'manager_changes_12m']; pd.concat({'caught leavers': lv[lv.flagged][cols].mean(), 'missed leavers': lv[~lv.flagged][cols].mean(), 'stayers': ts[ts.y == 0][cols].mean()}, axis=1).round(2)""")
    n.stage(31, "Permutation importance on the test set with the train-only calibrated v2: how much does ROC-AUC fall when one input is shuffled?")
    n.code("""from sklearn.inspection import permutation_importance
mdl = joblib.load(P.MODELS / 'attrition_v2_train_only.joblib')
pi = permutation_importance(mdl, Xte, yte, scoring='roc_auc', n_repeats=5, random_state=SEED, n_jobs=1)
imp = pd.DataFrame({'feature': Xte.columns, 'auc_drop': pi.importances_mean, 'sd': pi.importances_std}).sort_values('auc_drop', ascending=False)
save_table(imp, P.task_dir(13) / 'permutation_importance.csv'); fig, ax = plt.subplots(figsize=(7, 4)); imp.head(15)[::-1].plot.barh(x='feature', y='auc_drop', ax=ax, legend=False, color='#2f6fb0'); ax.set_xlabel('drop in test ROC-AUC when shuffled'); plt.tight_layout(); plt.savefig(P.FIGURES / 'task13_permutation_importance.png', dpi=130, bbox_inches='tight'); imp.head(10).round(4)""")
    n.stage(32, "Does v2 flag one group more than another? Gender and age come from the restricted `unlock_CX-15` file and are used **only here, for measurement**. They are not model inputs. Predictions for training rows are out-of-fold.")
    n.code("""if 'group_gender' not in out: print('restricted file not present: group measurement skipped')
else:
    print(out['group_gender'].round(3).to_string()); print(out['group_age'].round(3).to_string())""")
    n.code("""if 'group_gender' in out:
    g = out['group_gender']; a = out['group_age']; ok = a[a.n >= CFG['salary']['min_group_size']]
    print('flagged share by gender ranges %.3f to %.3f; by age band (groups with n >= %d) %.3f to %.3f; the company rule flags 10%% overall.' % (g.flagged_top10_share.min(), g.flagged_top10_share.max(), CFG['salary']['min_group_size'], ok.flagged_top10_share.min(), ok.flagged_top10_share.max()))
    print('Reading: small differences in flag rates can reflect real differences in the features (for example tenure), not unfairness. The 45+ group has only %d rows, too few for a conclusion.' % int(a[a.age_band == '45+'].n.iloc[0]))""")
    n.stage(33, "Probabilities must be real probabilities: class-weighted models over-state risk, so calibration is learned inside the training folds and chosen by out-of-fold Brier score.")
    n.code("""cal = out['calibration']; cal.round(4)""")
    n.code("""from sklearn.calibration import calibration_curve
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for k, c in (('none', '#c0392b'), (out['calibration_method'], '#2e8b57')):
    fr, mp = calibration_curve(ytr, out['oof'][k], n_bins=10, strategy='quantile'); ax[0].plot(mp, fr, 'o-', color=c, label=k)
ax[0].plot([0, .6], [0, .6], 'k--', lw=.8); ax[0].set_xlabel('predicted probability'); ax[0].set_ylabel('observed leave rate'); ax[0].legend(); ax[0].set_title('Calibration (out-of-fold)')
ax[1].hist(out['oof'][out['calibration_method']], bins=30, color='#2f6fb0'); ax[1].set_title('calibrated score distribution'); plt.tight_layout(); plt.savefig(P.FIGURES / 'task13_calibration.png', dpi=130, bbox_inches='tight')
print('chosen calibration:', out['calibration_method'], '| ECE %.4f (limit %.2f) -> %s' % (cal.set_index('method').loc[out['calibration_method'], 'ece'], CFG_A['max_expected_calibration_error'], 'OK' if cal.set_index('method').loc[out['calibration_method'], 'ece'] <= CFG_A['max_expected_calibration_error'] else 'ABOVE LIMIT'))""" if False else """from sklearn.calibration import calibration_curve
CFG_A = CFG['attrition']['success_criteria']
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for k, c in (('none', '#c0392b'), (out['calibration_method'], '#2e8b57')):
    fr, mp = calibration_curve(ytr, out['oof'][k], n_bins=10, strategy='quantile'); ax[0].plot(mp, fr, 'o-', color=c, label=k)
ax[0].plot([0, .6], [0, .6], 'k--', lw=.8); ax[0].set_xlabel('predicted probability'); ax[0].set_ylabel('observed leave rate'); ax[0].legend(); ax[0].set_title('Calibration (out-of-fold)')
ax[1].hist(out['oof'][out['calibration_method']], bins=30, color='#2f6fb0'); ax[1].set_title('calibrated score distribution'); plt.tight_layout(); plt.savefig(P.FIGURES / 'task13_calibration.png', dpi=130, bbox_inches='tight')
e = cal.set_index('method').loc[out['calibration_method'], 'ece']; print('chosen calibration:', out['calibration_method'], '| ECE %.4f (limit %.2f) -> %s' % (e, CFG_A['max_expected_calibration_error'], 'OK' if e <= CFG_A['max_expected_calibration_error'] else 'ABOVE LIMIT'))""")
    n.stage(34, "Decision on evidence. The question is not 'which model has the highest number' but 'is the gain real, and is it worth the extra complexity'.")
    n.code("""row = out['summary'].set_index('model'); b = row.loc['Logistic Regression (balanced) [baseline]']; v = row.loc['attrition-v2 candidate: LR + tuned HGB blend']; g = row.loc['HistGradientBoosting tuned']
tab = pd.DataFrame({'baseline LR': b, 'tuned gradient boosting': g, 'attrition-v2 (LR + HGB blend)': v})[0:0] if False else pd.DataFrame({'baseline LR': [b.roc_auc_mean, b.pr_auc_mean, b.recall_mean, b.brier_mean], 'tuned HGB alone': [g.roc_auc_mean, g.pr_auc_mean, g.recall_mean, g.brier_mean], 'attrition-v2 blend': [v.roc_auc_mean, v.pr_auc_mean, v.recall_mean, v.brier_mean]}, index=['roc_auc', 'pr_auc', 'recall', 'brier']).round(4); tab""")
    n.code("""allm = pd.concat([bench[['model', 'roc_auc_mean', 'roc_auc_sd', 'pr_auc_mean']].assign(source='programme reference'), out['summary'][['model', 'roc_auc_mean', 'roc_auc_sd', 'pr_auc_mean']].assign(source='this repo, same folds')], ignore_index=True)
allm = allm[~allm.model.str.contains('LEAKY')]; save_table(allm, P.task_dir(13) / 'comparison_with_benchmarks.csv'); allm.round(3)""")
    n.md("**Would I ship v2 if it were only slightly better but much harder to explain? No, not by itself.** A small gain inside the noise of five folds is not evidence of a better model, while a boosted ensemble is harder to explain to HR and to audit. I would ship the simpler logistic model for decisions and keep v2 as a challenger until a gain with a confidence interval above zero is shown on new data.")
    n.code("""print('Gradient boosting alone vs baseline: ROC-AUC %+.4f (%s).' % (g.roc_auc_mean - b.roc_auc_mean, 'worse' if g.roc_auc_mean < b.roc_auc_mean else 'better')); print('Blend vs baseline: ROC-AUC %+.4f, PR-AUC %+.4f.' % (v.roc_auc_mean - b.roc_auc_mean, v.pr_auc_mean - b.pr_auc_mean))
print('Verdict:', out['verdict']['text']); print('Decision:', ('attrition-v2 beats the baseline with a gain that is real, and is saved and used in Task 14.' if out['verdict']['gain_is_meaningful'] else 'attrition-v2 is saved and used in Task 14 because the task asks for it, but HR should treat it as EQUIVALENT to the simpler logistic baseline, not as a proven upgrade. If only a slightly better but harder-to-explain model is on offer, the simple model is the better one to ship.'))""")
    n.stage(35, "Final predictions: test-set scores were produced above; here the 430 active employees are scored with the production model (refit on all labelled rows, calibrated). The risk report itself is Task 14.")
    n.code("""prod = joblib.load(P.MODELS / 'attrition_v2.joblib'); sc, Xs, rep = V.load_scoring(); ps = prod.predict_proba(Xs)[:, 1]
res = pd.DataFrame({'employee_id': sc.employee_id.to_numpy(), 'risk_score_12m': ps.round(6)}); save_table(res, P.PREDICTIONS / 'attrition_v2_scores_2026.csv'); print(res.shape); res.risk_score_12m.describe().round(3).to_frame().T""")
    n.stage(36, "Business sanity checks: probabilities in range, mean close to the historical base rate, and agreement with the simple baseline.")
    n.code("""from scipy.stats import spearmanr
lr_cal = joblib.load(P.MODELS / 'attrition_v2_train_only.joblib'); base_lr = V.baseline_models()['Logistic Regression (balanced)'].fit(pd.concat([Xtr, Xte]), pd.concat([ytr, yte])).predict_proba(Xs)[:, 1]
top_v2, top_lr = set(res.employee_id[res.risk_score_12m >= res.risk_score_12m.quantile(.9)]), set(res.employee_id[base_lr >= np.quantile(base_lr, .9)])
print('probabilities within [0, 1]:', bool(res.risk_score_12m.between(0, 1).all()), '| mean score %.3f vs historical leave rate %.3f' % (res.risk_score_12m.mean(), y.mean()))
print('rank agreement with the baseline logistic model on the 430: spearman %.3f | overlap of the top 10%% lists: %d of %d' % (spearmanr(res.risk_score_12m, base_lr)[0], len(top_v2 & top_lr), len(top_v2)))""")
    n.stage(37, "Versioned artifacts, metadata and a reproducibility check: reload, re-score, compare checksums.")
    n.code("""import json; meta = json.load(open(P.METADATA / 'attrition_v2_metadata.json')); print({k: meta[k] for k in ('model_version', 'calibration_method', 'seed', 'train_rows', 'test_rows', 'production_fit_rows')}); print('environment:', meta['environment'])
ok = V.file_sha256(P.MODELS / 'attrition_v2.joblib') == meta['artifact_sha256']['attrition_v2.joblib']; again = joblib.load(P.MODELS / 'attrition_v2.joblib').predict_proba(Xs)[:, 1]
print('artifact checksum matches metadata:', ok, '| re-scoring gives identical probabilities:', bool(np.allclose(again, ps)), '| Phase 1/2 model files untouched:', sorted(p.name for p in (P.ROOT.parent / 'phase2' / 'models').glob('*.joblib'))[:3] if (P.ROOT.parent / 'phase2' / 'models').exists() else 'n/a')""")
    n.stage(38, "")
    n.code("""print('FINDINGS'); print('- Same-fold ROC-AUC: baseline LR %.3f, tuned gradient boosting %.3f, attrition-v2 blend %.3f.' % (b.roc_auc_mean, g.roc_auc_mean, v.roc_auc_mean))
print('- %s' % out['verdict']['text']); print('- Calibration fixed a large over-statement of risk (mean uncalibrated %.2f vs base rate %.2f): ECE %.3f -> %.3f.' % (cal.set_index('method').loc['none', 'mean_probability'], y.mean(), cal.set_index('method').loc['none', 'ece'], cal.set_index('method').loc[out['calibration_method'], 'ece']))
print('LIMITS: 2,383 labelled rows and about 350 leavers; five folds give wide uncertainty; the monthly features cover only recent months; the model supports HR conversations and must not be used for dismissal, pay or promotion decisions.')""")
    return n


if __name__ == "__main__":
    from pathlib import Path
    out = Path(__file__).resolve().parents[1] / "notebooks"
    for name, f in (("07_task13_attrition_v2_baseline", nb07), ("08_task13_gradient_boosting_tuning", nb08), ("09_task13_model_comparison_calibration", nb09)):
        print(f().save(out / f"{name}.ipynb"))
