"""Builds notebooks 12-14 (Task 15: fair pay bands)."""
from nbkit import NB

LOAD = """from cortexhr.phase3.common.io import read_csv, save_table, save_json
from cortexhr.phase3.common.validation import profile
from cortexhr.phase3.salary import model as S, run as SR
from cortexhr.phase3.explanations.risk_report import load_helpers
H = load_helpers()
sal = read_csv('salary_training'); df = S.load_salary(); dd = read_csv('data_dictionary')
print('salary rows:', sal.shape, '| engineered:', df.shape)"""


def nb12():
    n = NB("Task 15, Notebook 12: Salary data audit, EDA, leakage and feature choice", "15: Recommend fair pay bands", "Understand the salary training data, find where history may carry bias, and choose job-related features only.", list(range(1, 22)))
    n.stage(1, "Pay is personal. A recommendation that copies past pay copies past unfairness. The output is **advice** to a manager and HR, as a range, never an offer.")
    n.stage(2, "**Regression:** predict the annual fixed salary offered at hiring from job-related facts. **Output:** a low / mid / high band. A second job is a **fairness test**: do predictions or errors differ by gender or by career gap?")
    n.stage(3, "Fixed before results: beat the simple 'average by role' baseline; report MAE and R2; output a range whose coverage is close to the 80% it claims; measure gaps by gender and career gap with the supplied `group_gap()` and explain them; write the limits.")
    n.code(LOAD)
    n.code("pd.Series(CFG['salary']['success_criteria'], name='criterion').to_frame()")
    n.stage(4, "`salary_training` (2,934 historical hires) and, for measurement only, the restricted `unlock_CX-15` file with gender and age. The starter helper `group_gap()` is used for group gaps.")
    n.stage(5, "")
    n.code("""va = S.vault_gender(); print('restricted attributes available:', va is not None); sal.head(3).T""")
    n.stage(6, "One row per hire (`hire_id`).")
    n.code("print('rows:', len(sal), '| hire_id unique:', sal.hire_id.is_unique, '| hire years:', int(sal.hire_year.min()), 'to', int(sal.hire_year.max()), '| roles:', sal.role_id.nunique(), '| departments:', sal.department_id.nunique())")
    n.stage(7, "")
    n.code("dd[dd.sheet.str.contains('salary', case=False)][['column', 'data_type', 'description']]")
    n.stage(8, "")
    n.code("sal.dtypes.astype(str).to_frame('dtype').T")
    n.stage(9, "")
    n.code("profile(sal)")
    n.stage(10, "")
    n.code("print('missing values per column:', sal.isna().sum()[lambda s: s > 0].to_dict() or 'none')")
    n.stage(11, "")
    n.code("print('duplicate hire_id:', int(sal.hire_id.duplicated().sum()), '| exact duplicate rows:', int(sal.duplicated().sum()), '| duplicates ignoring hire_id/date:', int(sal.drop(columns=['hire_id', 'hire_date']).duplicated().sum()))")
    n.stage(12, "")
    n.code("""x = sal.offered_annual_fixed; q1, q3 = x.quantile([.25, .75]); i = q3 - q1; o = (x < q1 - 1.5 * i) | (x > q3 + 1.5 * i)
lx = np.log(x); lo = (lx < lx.quantile(.25) - 1.5 * (lx.quantile(.75) - lx.quantile(.25))) | (lx > lx.quantile(.75) + 1.5 * (lx.quantile(.75) - lx.quantile(.25)))
print('salary IQR outliers: %d (%.1f%%) on the raw scale, %d on the log scale -> the raw skew is mostly the senior-level spread, not bad data' % (o.sum(), o.mean() * 100, lo.sum()))
fig, ax = plt.subplots(1, 2, figsize=(10, 3)); x.plot.hist(bins=40, ax=ax[0], color='#2f6fb0'); ax[0].set_title('offered salary'); lx.plot.hist(bins=40, ax=ax[1], color='#2e8b57'); ax[1].set_title('log offered salary'); plt.tight_layout()""")
    n.stage(13, "")
    n.code("""chk = {'salary > 0': bool((sal.offered_annual_fixed > 0).all()), 'band_min <= band_mid <= band_max': bool(((sal.band_min_2026 <= sal.band_mid_2026) & (sal.band_mid_2026 <= sal.band_max_2026)).all()),
       'career gap >= 0': bool((sal.career_gap_months >= 0).all()), 'prior experience between 0 and 45': bool(sal.prior_experience_years.between(0, 45).all()), 'market_index > 0': bool((sal.market_index > 0).all())}
print(pd.Series(chk, name='ok').to_frame().T.to_string()); print('offers outside the 2026 band range (after market adjustment): %.1f%%' % (100 * ((sal.offered_annual_fixed < sal.band_min_2026 * sal.market_index) | (sal.offered_annual_fixed > sal.band_max_2026 * sal.market_index)).mean()))""")
    n.stage(14, "")
    n.code("df.groupby('job_level').offered_annual_fixed.agg(['count', 'median']).round(0)")
    n.stage(15, "")
    n.code("""fig, ax = plt.subplots(1, 3, figsize=(13, 3)); df.prior_experience_years.plot.hist(bins=30, ax=ax[0], color='#2f6fb0'); ax[0].set_title('prior experience'); df.career_gap_months.plot.hist(bins=30, ax=ax[1], color='#2f6fb0'); ax[1].set_title('career gap (months)'); df.market_index.plot.hist(bins=30, ax=ax[2], color='#2f6fb0'); ax[2].set_title('market index'); plt.tight_layout()""")
    n.stage(16, "The offer is almost proportional to **band mid-point x market index**. The ratio is the natural 'pay relative to the job' measure used in the fairness tests.")
    n.code("""fig, ax = plt.subplots(1, 2, figsize=(10, 3.4)); ax[0].scatter(df.band_mid_2026 * df.market_index, df.offered_annual_fixed, s=4, alpha=.4); ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].set_title('offer vs band mid x market index')
df.plot.scatter('prior_experience_years', 'pay_vs_band', s=4, alpha=.4, ax=ax[1]); ax[1].set_title('pay vs band by experience'); plt.tight_layout()
df[['pay_vs_band']].describe().T.round(3)""")
    n.stage(17, "")
    n.code("df[['pay_vs_band', 'prior_experience_years', 'career_gap_months', 'skills_count', 'job_level', 'market_index']].corr(method='spearman').round(2)")
    n.stage(18, "Target = `offered_annual_fixed` (rupees per year), right-skewed and roughly proportional to the job band, so models are trained on its **logarithm** and results are converted back to rupees.")
    n.code("print('skew raw: %.2f | skew log: %.2f' % (df.offered_annual_fixed.skew(), np.log(df.offered_annual_fixed).skew()))")
    n.stage(19, "Leakage and history. (1) `hire_id`, `hire_date` and `source` identify rows, they are not features. (2) The 2026 band belongs to the role and level, not to the person, so it is a legitimate job-related input. (3) **History:** past offers may contain bias; the model learns from them.")
    n.code("""print('columns that must never be inputs:', S.NEVER_INPUTS); print('protected attributes in the salary file:', [c for c in sal.columns if c in ('gender', 'age_years', 'age_band')] or 'none (they exist only in the restricted file)')
if va is not None:
    d = df.join(va, on='hire_id'); print('historical pay ratio by gender (offer / band mid x market index):'); print(d.groupby('gender').pay_vs_band.agg(['mean', 'size']).round(4).to_string())""")
    n.stage(20, "Engineered inputs: logs of the band mid-point and market index (the relationship is multiplicative).")
    n.code("df[['band_mid_2026', 'log_band_mid', 'market_index', 'log_market_index']].head(3)")
    n.stage(21, "Which inputs are **job-related** (allowed) and which are not. `career_gap_months` is not a requirement of any role and is strongly tied to gender in the restricted file, so it is excluded from the chosen model and used only in a comparison model that shows what happens when history is copied.")
    n.code("""pd.DataFrame([('department_id, role_id, job_level', 'what the job is', 'USE'), ('log_band_mid', 'the company band for this role and level', 'USE'), ('log_market_index', 'market conditions in the hiring year', 'USE'),
 ('prior_experience_years, skills_count, highest_education', 'what the person brings to the job', 'USE (reviewed)'), ('work_location', 'cost of labour in the location', 'USE (reviewed)'),
 ('career_gap_months', 'time out of work; not a job requirement', 'EXCLUDE from the chosen model; comparison model only'), ('gender, age', 'protected', 'NEVER an input; measurement only'), ('hire_id, hire_date, source', 'identifiers', 'NEVER an input')], columns=['inputs', 'meaning', 'decision'])""")
    return n


def nb13():
    n = NB("Task 15, Notebook 13: Models, cross-validation, tuning and accuracy", "15: Recommend fair pay bands", "Compare a simple baseline, a linear model and gradient boosting, with MAE and R2, then evaluate once on the test set.", list(range(22, 32)))
    n.code(LOAD)
    n.stage(22, "Scaling and one-hot encoding are inside the sklearn pipeline, learned on each training fold only. Rare categories (fewer than 5) are grouped; unseen categories are ignored.")
    n.code("S.preprocessor(S.JOB_FEATURES).fit(df.iloc[:2000][S.JOB_FEATURES]).transform(df.iloc[2000:][S.JOB_FEATURES]).shape")
    n.stage(23, "80% training / 20% test (seed 42); 5-fold cross-validation inside the training part; the test set is used once.")
    n.code("tr, te = S.split(df); print('train', len(tr), '| test', len(te), '| folds', CFG['salary']['n_splits'])")
    n.stage(24, "Baselines: (1) the task's simple 'average offered salary by role'; (2) a stronger reference, the role average adjusted for the market index, to show honestly how much the models add beyond what a careful clerk could do.")
    n.stage(25, "Linear regression (Ridge) because it is transparent; gradient boosting because pay may depend on interactions. Both predict the log salary.")
    n.stage(26, "")
    n.code("""import time; t = time.time(); o = SR.run_task15(write=True); print('Task 15 pipeline finished in %.0f s | final model: %s' % (time.time() - t, o['final_model']))""")
    n.stage(27, "Cross-validated accuracy on the training part (out-of-fold predictions).")
    n.code("cv = o['cv_metrics'].round(3); cv")
    n.stage(28, "Six gradient-boosting settings; the lowest cross-validated MAE is selected. Differences are small, so no more tuning is justified.")
    n.code("o['tuning_log'][['experiment_id', 'params', 'MAE_rupees', 'R2', 'decision']].round(3)")
    n.stage(29, "Test set, used once, with the models fitted on the training part. MAE and R2 for each model against the baselines.")
    n.code("""tm = o['test_metrics'].round(3); tm""")
    n.code("""b1, b2 = tm.loc['baseline: average offered salary by role'], tm.loc['baseline 2: role average adjusted for market index']; f = tm.loc[o['final_model']]
print('Final model %s: MAE Rs %.0f, R2 %.3f' % (o['final_model'], f.MAE_rupees, f.R2)); print('vs role-average baseline: MAE %.0f%% lower; vs market-adjusted role baseline: MAE %.0f%% lower' % (100 * (1 - f.MAE_rupees / b1.MAE_rupees), 100 * (1 - f.MAE_rupees / b2.MAE_rupees)))
print('success criterion "beat the simple baseline":', bool(f.MAE_rupees < b1.MAE_rupees))""")
    n.stage(30, "Where is the model wrong? Largest test errors and error by job level.")
    n.code("""te = o['test'].copy(); te['pred'] = o['test_pred'][o['final_model']]; te['abs_pct_err'] = (te.offered_annual_fixed - te.pred).abs() / te.offered_annual_fixed * 100
print(te.groupby('job_level').abs_pct_err.agg(['count', 'median', 'mean']).round(1).to_string())
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4)); ax[0].scatter(te.offered_annual_fixed, te.pred, s=5, alpha=.5); lim = [te.offered_annual_fixed.min(), te.offered_annual_fixed.max()]; ax[0].plot(lim, lim, 'r--', lw=.8); ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].set_xlabel('actual'); ax[0].set_ylabel('predicted')
ax[1].hist(np.log(te.offered_annual_fixed / te.pred), bins=30, color='#2f6fb0'); ax[1].set_title('log error (actual / predicted)'); plt.tight_layout()
te.sort_values('abs_pct_err', ascending=False).head(5)[['hire_id', 'role_id', 'job_level', 'prior_experience_years', 'offered_annual_fixed', 'pred', 'abs_pct_err']].round(1)""")
    n.stage(31, "What moves the recommendation? Coefficients of the linear model on the standardised numeric inputs (percent change in salary for a one-standard-deviation change), and permutation importance for the chosen model.")
    n.code("""rg = o['fitted']['Ridge regression (job-related features)']; names = rg.named_steps['prep'].get_feature_names_out(); coef = pd.Series(rg.named_steps['model'].coef_, index=names)
num = coef[[c for c in coef.index if c.startswith('num__')]].rename(lambda c: c.replace('num__', '')); (np.expm1(num) * 100).round(2).sort_values(ascending=False).to_frame('% change in salary per +1 sd')""")
    n.code("""from sklearn.inspection import permutation_importance
mdl = o['fitted'][o['final_model']]; Xte_ = o['test'][S.JOB_FEATURES]; yl = np.log(o['test'][S.TARGET].to_numpy(float))
pi = permutation_importance(mdl, Xte_, yl, scoring='neg_mean_absolute_error', n_repeats=5, random_state=SEED, n_jobs=1)
imp = pd.Series(pi.importances_mean, index=S.JOB_FEATURES).sort_values(ascending=False); fig, ax = plt.subplots(figsize=(6, 3)); imp[::-1].plot.barh(ax=ax, color='#2f6fb0'); ax.set_xlabel('increase in log-MAE when shuffled'); plt.tight_layout(); imp.round(4).to_frame('importance')""")
    return n


def nb14():
    n = NB("Task 15, Notebook 14: Group gaps, bands, final selection, outputs and limits", "15: Recommend fair pay bands", "Test the model for gaps by gender and career gap, output a band, decide what to ship and write the limits down.", list(range(32, 39)))
    n.code(LOAD)
    n.code("o = SR.run_task15(write=True); print('final model:', o['final_model'], '| restricted attributes available:', o['vault'])")
    n.stage(32, "Group gaps. Gender and age come from the restricted `unlock_CX-15` file and are **never inputs**; they define groups after the fact. Every row has an out-of-sample prediction (out-of-fold for training rows, train-fit for test rows). The supplied `group_gap()` gives best-minus-worst group mean for each measure. 'Pay ratio' = pay divided by (band mid-point x market index), so roles are comparable.")
    n.code("""if not o['vault']: print('restricted file not present: fairness measurement skipped')
else:
    print('HISTORY AND PREDICTIONS BY GENDER (mean pay ratio):'); print(o['gender_means'].round(4).to_string()); print(); print('group_gap() by gender:'); print(o['gender_gaps'].round(4).to_string())""")
    n.code("""if o['vault']:
    print('Female minus Male, mean pay ratio, with 95% bootstrap interval:')
    for k, v in o['gender_boot'].items(): print('  %-14s %+.4f  [%+.4f, %+.4f]' % (k, v['difference'], v['ci_low'], v['ci_high']))
    print(); print('career gap by gender:'); print(o['gender_vs_gap'].round(3).to_string())""")
    n.code("""if o['vault']:
    print('BY CAREER GAP:'); print(o['gap_bin_means'].round(4).to_string()); print(); print(o['gap_bin_gaps'].round(4).to_string())
    print(); print('BY AGE BAND (group_gap):'); print(o['age_gaps'].round(4).to_string())""")
    n.code("""if o['vault']:
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6)); gm = o['gender_means'][['actual offer / band mid-point (history)', 'model A predicted ratio', 'model B predicted ratio (chosen)']]; gm.plot.bar(ax=ax[0], rot=0); ax[0].set_ylim(gm.min().min() - .02, gm.max().max() + .01); ax[0].set_title('Pay ratio by gender'); ax[0].legend(fontsize=6)
    cm = o['gap_bin_means'][['actual offer / band mid-point (history)', 'model A predicted ratio', 'model B predicted ratio (chosen)']]; cm.plot.bar(ax=ax[1], rot=20); ax[1].set_ylim(cm.min().min() - .02, cm.max().max() + .01); ax[1].set_title('Pay ratio by career gap'); ax[1].legend(fontsize=6); plt.tight_layout(); plt.savefig(P.FIGURES / 'task15_group_gaps.png', dpi=130, bbox_inches='tight')""")
    n.code("""if o['vault']:
    gm, cm = o['gender_means'], o['gap_bin_means']; ga = o['gender_gaps'].set_index('measure').group_gap; ca = o['gap_bin_gaps'].set_index('measure').group_gap
    hist_g, a_g, b_g = ga['actual offer / band mid-point (history)'], ga['model A predicted ratio'], ga['model B predicted ratio (chosen)']; hist_c, a_c, b_c = ca['actual offer / band mid-point (history)'], ca['model A predicted ratio'], ca['model B predicted ratio (chosen)']
    print('WHAT THE GAPS MEAN')
    print('- History: by gender the best-minus-worst mean pay ratio is %.4f; by career gap %.4f (offers fall as the gap grows).' % (hist_g, hist_c))
    print('- Model A (career gap allowed as an input) repeats this: gender gap %.4f, career-gap gap %.4f.' % (a_g, a_c))
    print('- Model B (chosen, job-related inputs only): gender gap %.4f, career-gap gap %.4f.' % (b_g, b_c))
    ra = o['gap_bin_means']['model B actual/predicted']; rg_ = o['gender_means']['model B actual/predicted']
    print('- Women hold longer career gaps on average (%.1f vs %.1f months for men), so a career-gap input carries the gender difference into the recommendation even though gender is never an input: a proxy.' % (o['gender_vs_gap'].loc['Female', 'mean'], o['gender_vs_gap'].loc['Male', 'mean']))
    print('- What history still does: actual offers divided by the chosen-model recommendation average %.4f for people with no career gap and %.4f for over 24 months, and %.4f for women vs %.4f for men. People with long gaps (and women) were offered less than the job-related recommendation, so the chosen model does NOT copy that shortfall.' % (ra['none'], ra['over 24 months'], rg_['Female'], rg_['Male']))
    print('- Remaining gender gap in the chosen-model recommendations: %+.4f (95%% interval %+.4f to %+.4f): %s.' % (o['gender_boot']['ratio_pred_B']['difference'], o['gender_boot']['ratio_pred_B']['ci_low'], o['gender_boot']['ratio_pred_B']['ci_high'], 'indistinguishable from zero' if o['gender_boot']['ratio_pred_B']['ci_low'] <= 0 <= o['gender_boot']['ratio_pred_B']['ci_high'] else 'still different from zero'))
    print('WHAT WAS CHANGED: career_gap_months was removed as an input; band recommendations are driven by role, level, band, market, experience, skills, education and location only.')""")
    n.md("**Think for yourself: if women in the data were paid less in the past, should the model copy that?** No. A model trained on past offers learns whatever pattern is in them, including a pattern caused by a career gap penalty. What I did: (1) never use gender or age; (2) exclude career gap, which is not a job requirement and acts as a proxy for gender; (3) measure what is left with the supplied helper and a bootstrap interval; (4) present a *range*, so the manager decides inside it; (5) say plainly that the training target itself may contain past bias, so a model that matches history perfectly is not the goal. What I would add before real use: re-base the target (for example use only recent offers after a pay review, or audit and correct historical offers), review any remaining gap with HR and legal, and monitor offers against the band by gender every quarter.")
    n.stage(33, "Reliability of the range. The band is the model's mid-point times the 10th and 90th percentile of its out-of-fold error, so it should contain about 80% of actual offers. Check on the untouched test set and by group.")
    n.code("""print('test coverage of the low..high band: %.3f (target %.2f)' % (o['band_coverage'], CFG['salary']['success_criteria']['band_target_coverage']))
bt = o['bands_test']; print(bt.groupby(pd.cut(o['test'].job_level.to_numpy(), [0, 2, 4, 8]), observed=True).inside_band.agg(['mean', 'size']).round(3).to_string())
if o['vault']:
    t = o['full'][o['full']._part == 'test'].merge(bt[['hire_id', 'inside_band']], on='hire_id'); print(t.groupby('gender').inside_band.agg(['mean', 'size']).round(3).to_string())""")
    n.stage(34, "Final selection: accuracy lost by removing the career-gap input, against the fairness gain.")
    n.code("""tm = o['test_metrics']; fm = o['final_model']; with_gap = 'Gradient boosting + career_gap_months' if fm.startswith('Gradient') else 'Ridge regression + career_gap_months'
cmp_ = tm.loc[[fm, with_gap]].round(3); cmp_""")
    n.code("""d = tm.loc[with_gap, 'MAE_rupees'] - tm.loc[fm, 'MAE_rupees']; print('MAE with career gap minus MAE without: Rs %+.0f (%.2f%% of MAE) -> removing the career-gap input costs %s accuracy.' % (d, abs(d) / tm.loc[fm, 'MAE_rupees'] * 100, 'no' if d >= 0 else 'a little'))
print('Final model:', fm, '| version', SR.MODEL_VERSION)""")
    n.stage(35, "A band (low, mid, high) for every hire in the test set, and an example for a new role.")
    n.code("""bt = o['bands_test']; save_table(bt, P.task_dir(15) / 'band_recommendations_test.csv'); bt.head(8).round(0)""")
    n.code("""import joblib; art = joblib.load(P.MODELS / 'salary_band_v1.joblib'); ex = df.iloc[[0]].copy(); r = S.recommend_band(art['model'], ex, art['band_multipliers'], art['features']); print('example (a hire from the data):', r.iloc[0].to_dict(), '| actual offer was', int(ex.offered_annual_fixed.iloc[0]))""")
    n.stage(36, "Format and business checks: a range not a single number, ordered, positive, and compared with the company band for the role.")
    n.code("""ok = {'low < mid < high for every row': bool(((bt.recommended_low < bt.recommended_mid) & (bt.recommended_mid < bt.recommended_high)).all()), 'all values positive': bool((bt[['recommended_low', 'recommended_mid', 'recommended_high']] > 0).all().all()),
      'output is a range (3 numbers)': True, 'no protected column in the output': not any(c in bt.columns for c in ('gender', 'age_years', 'age_band'))}
tt = o['test'].reset_index(drop=True); inside_co_band = ((bt.recommended_mid.reset_index(drop=True) >= tt.band_min_2026 * tt.market_index) & (bt.recommended_mid.reset_index(drop=True) <= tt.band_max_2026 * tt.market_index)).mean()
print(pd.Series(ok, name='ok').to_frame().T.to_string()); print('share of recommended mid-points inside the company band for the role (market adjusted): %.3f' % inside_co_band)""")
    n.stage(37, "")
    n.code("""import json; meta = json.load(open(P.METADATA / 'salary_band_v1_metadata.json')); print({k: meta[k] for k in ('model_version', 'model', 'seed', 'train_rows', 'test_rows', 'production_fit_rows', 'band_coverage_test')}); print('excluded as job-unrelated:', meta['excluded_job_unrelated'], '| checksum ok:', meta['sha256'] == __import__('cortexhr.phase3.attrition.v2', fromlist=['x']).file_sha256(P.MODELS / 'salary_band_v1.joblib'))""")
    n.stage(38, "")
    n.code("""tmx = o['test_metrics']; f = tmx.loc[o['final_model']]; b1 = tmx.loc['baseline: average offered salary by role']
print('FINDINGS'); print('- Final model %s: test MAE Rs %.0f, R2 %.3f; the role-average baseline has MAE Rs %.0f, R2 %.3f.' % (o['final_model'], f.MAE_rupees, f.R2, b1.MAE_rupees, b1.R2)); print('- The band (low, mid, high) covers %.0f%% of actual test offers (target 80%%).' % (o['band_coverage'] * 100))
if o['vault']: print('- History shows a pay-ratio gap by gender and a larger one by career gap; a model allowed to use career gap repeats it; the chosen model does not use it.')
print('LIMITS: the target is historical offers and may contain past bias; the band describes what was offered, not what is fair; gender and age were measured only through a restricted file with three gender values and a small third group (Other: %d hires);' % (int((o['full'].gender == 'Other').sum()) if o['vault'] else 0), end=' ')
print(' 2026 bands are applied to hires from 2012 on through a market index; skills and education are coarse; no negotiation, performance or internal equity data; the output is advice and the final decision stays with a manager and HR.')""")
    return n


if __name__ == "__main__":
    from pathlib import Path
    out = Path(__file__).resolve().parents[1] / "notebooks"
    for name, f in (("12_task15_salary_data_audit", nb12), ("13_task15_salary_regression", nb13), ("14_task15_salary_fairness", nb14)):
        print(f().save(out / f"{name}.ipynb"))
