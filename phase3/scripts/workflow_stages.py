"""The full 38-stage Phase 1/2 workflow for notebook 15, applied end to end to all five Phase 3 tasks.
Every number is read from the data files or from the artifacts written by notebooks 01 to 14; nothing is typed in by hand."""


def add_stages(n):
    n.md("## Part 1: The 38-stage workflow for all of Phase 3 (Tasks 11 to 15)\n\nEach stage below covers the five tasks together, using the same stage names as Phase 1 and Phase 2. "
         "The detailed per-task work is in notebooks 01 to 14; this notebook re-reads their saved results so one notebook shows the whole workflow from business question to conclusion. "
         "Part 2 (Appendix A to G) keeps the artifact checks and the one-case walkthroughs.")
    # ---------------------------------------------------------------- 1-4
    n.stage(1, "Five decisions are supported in Phase 3. All outputs are advice for a human, never an automatic decision.")
    n.code("""biz = pd.DataFrame([
 ('11 Resume ranking', 'Recruiters with hundreds of resumes per opening', 'Which candidates to read first for a job'),
 ('12 Fairness audit', 'HR leadership, legal, candidates', 'Whether the screener treats identical candidates identically'),
 ('13 Attrition model v2', 'HR business partners', 'Which employees may need an earlier conversation'),
 ('14 Risk explanations', 'Managers and HR', 'What to talk about with each of the 430 active employees'),
 ('15 Pay bands', 'Hiring managers and compensation team', 'A fair offer range for a new hire')], columns=['task', 'who needs the result', 'decision it supports'])
biz""")
    n.stage(2, "Each task is one precisely stated problem with an input and an output.")
    n.code("""prob = pd.DataFrame([
 ('11', 'Rank 185 resumes against each of 10 job descriptions', 'resume text, job text, skills', 'ranked list per job with matched / missing skills', 'ranking'),
 ('12', 'Do 25 twin resumes (one detail changed) get the same score?', 'resume_bias_pairs + screener', 'score gap per pair, before and after a fix', 'measurement'),
 ('13', 'Will an employee leave within 12 months?', '42 columns of the attrition table (as-of date)', 'probability of leaving', 'binary classification'),
 ('14', 'Why does each of 430 active employees get their score?', 'attrition_scoring_2026 + model v2', 'score, band, 3 reasons, review note', 'explanation'),
 ('15', 'What offer range fits a new hire for a role?', 'job-related columns of salary_training', 'low / mid / high band', 'regression + range')], columns=['task', 'question', 'input', 'output', 'type'])
prob""")
    n.stage(3, "Success criteria were fixed from the task brief before results were read. Where each is measured is listed.")
    n.code("""crit = pd.DataFrame([
 ('11', 'Quality reported against human labels; 5 misranks explained; no name, gender, age or college tier in the score', 'reports/task11/'),
 ('12', 'Average twin gap before and after; ranking quality still reported; honest statement', 'reports/task12/'),
 ('13', 'Same folds as Task 3; gain proven or its absence explained; tuning log; versioned model', 'reports/task13/'),
 ('14', '430 rows with score, band, 3 plain-English reasons, sample format, human-review note', 'reports/task14/'),
 ('15', 'Beats the role-average baseline; group gaps measured; output is a range; limits written', 'reports/task15/')], columns=['task', 'success criterion', 'evidence folder'])
crit""")
    n.stage(4, "The seven tables Phase 3 uses, and the two restricted files that are used only to measure fairness.")
    n.code("""from cortexhr.phase3.common.io import read_csv
import json
used = ['resumes', 'job_descriptions', 'resume_job_labels', 'resume_bias_pairs', 'attrition_dataset', 'attrition_scoring_2026', 'salary_training', 'sample_attrition_risk', 'baseline_benchmarks', 'data_dictionary']
disc = pd.DataFrame([(t, 'task ' + {'resumes': '11-12', 'job_descriptions': '11', 'resume_job_labels': '11', 'resume_bias_pairs': '12', 'attrition_dataset': '13', 'attrition_scoring_2026': '14', 'salary_training': '15', 'sample_attrition_risk': '14', 'baseline_benchmarks': '13', 'data_dictionary': 'all'}[t], (ROOT / '03_Data' / 'csv' / (t + '.csv')).exists()) for t in used], columns=['table', 'used by', 'file present'])
vault = ROOT / '04_Validation_Vault'
print('restricted validation files present locally (names only, used for measurement, never as model inputs):', sorted(p.name[:14] for p in vault.glob('unlock_CX*')) if vault.exists() else 'folder not present')
disc""")
    # ---------------------------------------------------------------- 5-9
    n.stage(5, "All tables are loaded with the project helper and the expected row counts are checked.")
    n.code("""res = read_csv('resumes'); jd = read_csv('job_descriptions'); lab = read_csv('resume_job_labels'); pairs = read_csv('resume_bias_pairs')
att = read_csv('attrition_dataset'); sc = read_csv('attrition_scoring_2026'); sal = read_csv('salary_training'); sample = read_csv('sample_attrition_risk'); dd = read_csv('data_dictionary')
R = {t: P.task_dir(t) for t in (11, 12, 13, 14, 15)}; risk = pd.read_csv(R[14] / 'attrition_risk_2026.csv')
expected = {'resumes': 185, 'job_descriptions': 10, 'resume_job_labels': 555, 'resume_bias_pairs': 25, 'attrition_scoring_2026': 430, 'salary_training': 2934}
got = {'resumes': len(res), 'job_descriptions': len(jd), 'resume_job_labels': len(lab), 'resume_bias_pairs': len(pairs), 'attrition_scoring_2026': len(sc), 'salary_training': len(sal)}
chk = pd.DataFrame({'expected': expected, 'loaded': got}); chk['ok'] = chk.expected == chk.loaded; print('attrition_dataset rows:', len(att)); chk""")
    n.stage(6, "Rows, columns and the key of every table; a key that is not unique would break joins.")
    n.code("""T = {'resumes': (res, 'resume_id'), 'job_descriptions': (jd, 'job_id'), 'resume_job_labels': (lab, ['resume_id', 'job_id']), 'resume_bias_pairs': (pairs, 'pair_id'),
     'attrition_dataset': (att, 'row_id'), 'attrition_scoring_2026': (sc, 'employee_id'), 'salary_training': (sal, 'hire_id')}
pd.DataFrame([(k, d.shape[0], d.shape[1], str(key), not d.duplicated(key).any()) for k, (d, key) in T.items()], columns=['table', 'rows', 'columns', 'key', 'key is unique'])""")
    n.stage(7, "Meaning of the columns comes from the supplied data dictionary, checked against the real columns.")
    n.code("""print('dictionary rows:', len(dd), '| sheets:', dd.sheet.nunique())
focus = ['left_within_12m', 'tenure_months', 'comp_ratio', 'offered_annual_fixed', 'career_gap_months', 'human_suitability_score', 'required_skill_coverage', 'college_tier']
dd[dd.column.isin(focus)][['sheet', 'column', 'data_type', 'description']].drop_duplicates('column').reset_index(drop=True)""")
    n.stage(8, "Every column has the type the analysis expects; dates parse.")
    n.code("""rows = []
for k, (d, _) in T.items():
    ty = d.dtypes.astype(str).value_counts().to_dict(); rows.append((k, ty))
date_ok = {'attrition observation_date': pd.to_datetime(att.observation_date, errors='coerce').notna().mean(), 'salary hire_date': pd.to_datetime(sal.hire_date, errors='coerce').notna().mean()}
print('share of dates that parse:', {k: round(v, 3) for k, v in date_ok.items()}); pd.DataFrame(rows, columns=['table', 'column types'])""")
    n.stage(9, "A one-page quality profile: missing cells, exact duplicate rows and constant columns per table.")
    n.code("""rows = []
for k, (d, _) in T.items():
    rows.append((k, round(d.isna().mean().mean() * 100, 2), int(d.duplicated().sum()), [c for c in d.columns if d[c].nunique(dropna=False) <= 1]))
pd.DataFrame(rows, columns=['table', 'missing cells %', 'duplicate rows', 'constant columns'])""")
    # ---------------------------------------------------------------- 10-13
    n.stage(10, "Where values are missing and why. Missing exit information for people who stayed is by design, and it is exactly the leakage Task 13 excludes.")
    n.code("""miss = pd.concat({k: d.isna().mean().mul(100).round(1)[lambda s: s > 0].sort_values(ascending=False).head(4) for k, (d, _) in T.items()}).rename('missing %')
stay = att[att.left_within_12m == 0]; left = att[att.left_within_12m == 1]
print('exit_date missing: stayers %.1f%% | leavers %.1f%%  (by design: only leavers have an exit)' % (stay.exit_date.isna().mean() * 100, left.exit_date.isna().mean() * 100)); miss.to_frame()""")
    n.stage(11, "Exact and key duplicates, including identical resume texts, which would inflate ranking quality.")
    n.code("""dup = pd.DataFrame([
 ('resumes: resume_id', int(res.resume_id.duplicated().sum())), ('resumes: identical resume_text', int(res.resume_text.duplicated().sum())),
 ('labels: (resume_id, job_id)', int(lab.duplicated(['resume_id', 'job_id']).sum())), ('attrition: (employee_id, observation_date)', int(att.duplicated(['employee_id', 'observation_date']).sum())),
 ('scoring: employee_id', int(sc.employee_id.duplicated().sum())), ('salary: hire_id', int(sal.hire_id.duplicated().sum()))], columns=['check', 'duplicates'])
from cortexhr.data_loading import prepare_attrition, clean_raw
att_clean, dlog = prepare_attrition(att); print('handling (project cleaning step, same in training and scoring): exact duplicate snapshots ignoring row_id ->', dlog)
print('duplicate (employee, date) keys after cleaning:', int(att_clean.duplicated(['employee_id', 'observation_date']).sum())); dup""")
    n.stage(12, "Unusual values by the 1.5 x IQR rule. They are inspected, not deleted: salaries and experience are real, skewed quantities.")
    n.code("""cols = [('resume experience (years)', res.total_experience_years), ('tenure (months)', att.tenure_months), ('annual fixed pay (Rs)', att.annual_fixed_pay), ('offered salary (Rs)', sal.offered_annual_fixed)]
def iqr(s):
    q1, q3 = s.quantile([.25, .75]); i = q3 - q1; return int(((s < q1 - 1.5 * i) | (s > q3 + 1.5 * i)).sum())
out = pd.DataFrame([(a, len(s), iqr(s), round(s.min(), 1), round(s.median(), 1), round(s.max(), 1)) for a, s in cols], columns=['variable', 'n', 'IQR outliers', 'min', 'median', 'max'])
fig, ax = plt.subplots(1, 4, figsize=(13, 3)); [ax[i].boxplot(s.dropna(), vert=True) or ax[i].set_title(a, fontsize=9) for i, (a, s) in enumerate(cols)]; plt.tight_layout(); out""")
    n.stage(13, "Impossible values and broken links between tables.")
    n.code("""bad = pd.DataFrame([
 ('resume experience < 0 or > 45 years', int(((res.total_experience_years < 0) | (res.total_experience_years > 45)).sum())),
 ('attrition tenure_months < 0', int((att.tenure_months < 0).sum())), ('attrition comp_ratio <= 0', int((att.comp_ratio <= 0).sum())),
 ('salary offered <= 0', int((sal.offered_annual_fixed <= 0).sum())), ('salary career_gap_months < 0', int((sal.career_gap_months < 0).sum())),
 ('labels pointing to unknown resume', int((~lab.resume_id.isin(res.resume_id)).sum())), ('labels pointing to unknown job', int((~lab.job_id.isin(jd.job_id)).sum())),
 ('twin pairs pointing to unknown resume', int((~pairs.resume_id_original.isin(res.resume_id)).sum() + (~pairs.resume_id_twin.isin(res.resume_id)).sum())),
 ('scoring rows whose employee is missing in risk report', int((~sc.employee_id.isin(risk.employee_id)).sum()))], columns=['rule', 'violations'])
c = clean_raw(att); print('handling: clean_raw turns impossible tenure into missing (then imputed inside each fold). negative tenure before: %d, after: %d (now missing: %d)' % ((att.tenure_months < 0).sum(), (c.tenure_months.dropna() < 0).sum(), int(c.tenure_months.isna().sum() - att.tenure_months.isna().sum())))
print('human score range:', lab.human_suitability_score.min(), 'to', lab.human_suitability_score.max()); bad""")
    # ---------------------------------------------------------------- 14-19
    n.stage(14, "Look at the data before trusting any model.")
    n.code("""fig, ax = plt.subplots(2, 2, figsize=(11, 6))
ax[0, 0].hist(res.total_experience_years, bins=20, color='#2e75b6'); ax[0, 0].set_title('Resume experience (years)')
ax[0, 1].hist(att.tenure_months, bins=30, color='#2e75b6'); ax[0, 1].set_title('Employee tenure (months)')
ax[1, 0].hist(sal.offered_annual_fixed / 1e5, bins=40, color='#2e75b6'); ax[1, 0].set_title('Offered salary (lakh Rs)')
ax[1, 1].hist(risk.risk_score_12m, bins=30, color='#c55a11'); ax[1, 1].set_title('Model v2 risk score, 430 active employees'); plt.tight_layout(); print('four distributions drawn')""")
    n.stage(15, "One variable at a time, across the tasks.")
    n.code("""pd.concat({'resume experience': res.total_experience_years.describe(), 'tenure months': att.tenure_months.describe(), 'comp ratio': att.comp_ratio.describe(), 'offered salary': sal.offered_annual_fixed.describe(), 'human score': lab.human_suitability_score.describe(), 'risk score 2026': risk.risk_score_12m.describe()}, axis=1).round(2)""")
    n.stage(16, "Two variables together: each against the label it is meant to predict.")
    n.code("""fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
att.assign(b=pd.cut(att.tenure_months, [0, 12, 24, 48, 96, 1000])).groupby('b', observed=True).left_within_12m.mean().plot.bar(ax=ax[0], color='#2e75b6'); ax[0].set_title('Attrition rate by tenure (months)'); ax[0].set_xlabel('')
sal.groupby('job_level').offered_annual_fixed.mean().div(1e5).plot.bar(ax=ax[1], color='#2e75b6'); ax[1].set_title('Mean offer by job level (lakh Rs)'); ax[1].set_xlabel('')
ax[2].scatter(lab.required_skill_coverage, lab.human_suitability_score, s=8, alpha=.5); ax[2].set_title('Human score vs required-skill coverage'); plt.tight_layout()
print('corr(human score, required skill coverage) = %.3f' % lab.human_suitability_score.corr(lab.required_skill_coverage))""")
    n.stage(17, "Several variables together: correlation among the salary inputs shows which ones carry overlapping information.")
    n.code("""num = sal[['band_mid_2026', 'prior_experience_years', 'career_gap_months', 'skills_count', 'market_index', 'offered_annual_fixed']]; cm = num.corr()
fig, ax = plt.subplots(figsize=(5.5, 4.4)); im = ax.imshow(cm, cmap='coolwarm', vmin=-1, vmax=1); ax.set_xticks(range(len(cm))); ax.set_xticklabels(cm.columns, rotation=60, ha='right', fontsize=8); ax.set_yticks(range(len(cm))); ax.set_yticklabels(cm.columns, fontsize=8)
[ax.text(j, i, '%.2f' % cm.iloc[i, j], ha='center', va='center', fontsize=7) for i in range(len(cm)) for j in range(len(cm))]; plt.colorbar(im); plt.tight_layout(); cm.round(2)""")
    n.stage(18, "The label of each task: how it is distributed and how reliable it is.")
    n.code("""lab_tab = pd.DataFrame([
 ('attrition left_within_12m', 'share = 1', round(att.left_within_12m.mean(), 3)), ('attrition voluntary_left_within_12m', 'share = 1', round(att.voluntary_left_within_12m.mean(), 3)),
 ('resume shortlist (human)', 'share = Y', round((lab.shortlist == 'Y').mean(), 3)), ('human suitability score', 'mean', round(lab.human_suitability_score.mean(), 2)),
 ('offered salary', 'mean / median (right-skewed)', '%.0f / %.0f' % (sal.offered_annual_fixed.mean(), sal.offered_annual_fixed.median()))], columns=['label', 'statistic', 'value']); lab_tab""")
    n.stage(19, "Anything that carries the answer or protected information into a model is excluded and checked here with an assertion.")
    n.code("""am = json.load(open(P.METADATA / 'attrition_v2_metadata.json')); sm = json.load(open(P.METADATA / 'salary_band_v1_metadata.json')); rm = json.load(open(P.METADATA / 'resume_ranker_v2_metadata.json'))
leak = set(am['excluded_as_inputs']) & set(am['features_raw']); prot = set(sm['never_inputs']) & set(sm['features'])
assert not leak and not prot, (leak, prot)
print('attrition: excluded as inputs ->', am['excluded_as_inputs']); print('salary never inputs ->', sm['never_inputs']); print('resume ranker never used ->', rm['never_used'])
print('overlap between excluded and used inputs: attrition', leak or 'none', '| salary', prot or 'none')""")
    # ---------------------------------------------------------------- 20-24
    n.stage(20, "The inputs each model really uses, built from the raw data.")
    n.code("""print('resume score inputs:', rm['score_inputs']); print('attrition raw features:', len(am['features_raw']), '| preprocessing:', am['preprocessing'])
print('salary features (%d): %s' % (len(sm['features']), sm['features']))""")
    n.stage(21, "Which inputs matter. Permutation importance on the attrition model (drop in ROC-AUC when a column is shuffled).")
    n.code("""pi = pd.read_csv(R[13] / 'permutation_importance.csv').sort_values('auc_drop', ascending=False).head(10)
fig, ax = plt.subplots(figsize=(6.5, 3.6)); ax.barh(pi.feature[::-1], pi.auc_drop[::-1], xerr=pi.sd[::-1], color='#2e75b6'); ax.set_xlabel('drop in ROC-AUC'); ax.set_title('Top 10 attrition features'); plt.tight_layout(); pi.round(4)""")
    n.stage(22, "Cleaning and encoding are fitted on training data only, inside every fold, so no information leaks from the test part.")
    n.code("""print('attrition:', am['preprocessing']); print('attrition note:', am['note'][:300]); print('salary note:', sm['note'][:300])""")
    n.stage(23, "How data was split so no result is measured on data the model has seen.")
    n.code("""print('attrition: train rows', am['train_rows'], '| test rows', am['test_rows'], '| production refit rows', am['production_fit_rows'], '|', am['cv'], '| seed', am['seed'])
print('salary: train rows', sm['train_rows'], '| test rows', sm['test_rows'], '| production refit rows', sm['production_fit_rows'], '| seed', sm['seed'])
print('resume ranking: no training. The engine is rule based, scored on the human-labelled pool; weights chosen from', len(pd.read_csv(R[11] / 'weight_experiments.csv')), 'logged experiments')""")
    n.stage(24, "The simple method every model must beat, per task.")
    n.code("""q = pd.read_csv(R[11] / 'ranking_quality_summary.csv').set_index('engine'); cv = pd.read_csv(R[13] / 'cv_summary.csv').set_index('model'); sm_t = pd.read_csv(R[15] / 'test_metrics.csv').set_index('model')
base = pd.DataFrame([('11', 'random ordering', 'precision@5 %.3f, NDCG@5 %.3f' % tuple(q.loc['random ordering', ['precision@5', 'ndcg@5']])),
 ('13', cv.index[cv.index.str.contains('baseline')][0], 'CV ROC-AUC %.3f' % cv[cv.index.str.contains('baseline')].roc_auc_mean.iloc[0]),
 ('15', sm_t.index[0], 'test MAE Rs %.0f, R2 %.3f' % (sm_t.MAE_rupees.iloc[0], sm_t.R2.iloc[0]))], columns=['task', 'baseline', 'its score']); base""")
    # ---------------------------------------------------------------- 25-31
    n.stage(25, "Why these candidates: a transparent one first, then a stronger one, so any gain can be measured against something simple.")
    n.code("""print('Task 11 engines:', list(q.index)); print('Task 13 models:', list(cv.index)); print('Task 15 models:', list(sm_t.index))
lgb = pd.read_csv(R[13] / 'lightgbm_cv_summary.csv'); print('Task 13 separate LightGBM challenger rows:', len(lgb))""")
    n.stage(26, "Fit the candidates. Parameters and weights that were actually used are printed from the saved metadata.")
    n.code("""print('attrition HistGradientBoosting params:', am['hgb_params']); print('resume score weights:', rm['weights']); print('salary final model:', sm['model'], '| band multipliers (log):', sm['band_multipliers_log'])""")
    n.stage(27, "Repeat the evaluation on several splits to see the spread, not only one lucky number.")
    n.code("""fig, ax = plt.subplots(figsize=(7, 3.2)); ax.errorbar(cv.roc_auc_mean, range(len(cv)), xerr=cv.roc_auc_sd, fmt='o', color='#1f4e79'); ax.set_yticks(range(len(cv))); ax.set_yticklabels([m[:45] for m in cv.index], fontsize=8); ax.set_xlabel('CV ROC-AUC, mean +/- sd (5 folds)'); plt.tight_layout()
print('salary CV (5 folds) is stored in metadata:', list(sm['cv_metrics'].keys())[:3], '...'); cv[['roc_auc_mean', 'roc_auc_sd', 'pr_auc_mean', 'brier_mean']].round(4)""")
    n.stage(28, "A small, logged search. Nothing is tuned on the test set; every experiment and the decision about it is kept.")
    n.code("""t13 = pd.read_csv(R[13] / 'tuning_log.csv'); t15 = pd.read_csv(R[15] / 'tuning_log.csv'); wx = pd.read_csv(R[11] / 'weight_experiments.csv')
print('Task 11 weight experiments:', len(wx), '| Task 13 tuning log:', len(t13), '| Task 13 LightGBM log:', len(pd.read_csv(R[13] / 'lightgbm_tuning_log.csv')), '| Task 15 tuning log:', len(t15))
t13[['experiment_id', 'model', 'roc_auc_mean', 'roc_auc_sd', 'decision']].round(4)""")
    n.stage(29, "Metrics suited to each problem, against the baseline.")
    n.code("""tm = pd.read_csv(R[13] / 'test_metrics.csv')[['model', 'roc_auc', 'pr_auc', 'brier', 'ece', 'recall', 'precision_top10']].round(3)
print('Task 11 ranking quality against human labels'); print(q.round(3).to_string()); print(); print('Task 13 held-out test'); print(tm.to_string()); print(); print('Task 15 test'); print(sm_t.round(3).to_string())""")
    n.stage(30, "Look at the cases the models get wrong and say why.")
    n.code("""mis = pd.read_csv(R[11] / 'misranking_analysis.csv'); print('Task 11: five misranked resumes'); print(mis[['job_id', 'resume_id', 'predicted_rank', 'human_score', 'misrank_type', 'root_cause']].to_string(index=False))
c = pd.read_csv(R[13] / 'test_metrics.csv').iloc[-1]; print('\\nTask 13 confusion matrix (test): tn %d fp %d fn %d tp %d' % (c.tn, c.fp, c.fn, c.tp))
br = pd.read_csv(R[15] / 'band_recommendations_test.csv'); print('Task 15: actual offers inside the recommended band: %.1f%% of %d' % (br.inside_band.mean() * 100, len(br)))""")
    n.stage(31, "What drives the output, in plain words. The 430-row report uses grouped Shapley reasons; the native SHAP library was also run on all 430 employees as a separate comparison.")
    n.code("""reasons = pd.concat([risk.top_reason_1, risk.top_reason_2, risk.top_reason_3]).str.split(' \\\\(').str[0].value_counts().head(8)
print('most frequent reasons across the 430 reports:'); print(reasons.to_string())
sf = R[14] / 'shap_full430_summary.json'; cf = R[14] / 'shap_full430_comparison.csv'
if sf.exists():
    s = json.load(open(sf)); print('\\nnative SHAP', s.get('shap_version', '?'), 'on', s.get('rows', '?'), 'employees; additivity error', s.get('additivity_max_error', '?'))
    if cf.exists():
        k = pd.read_csv(cf); print('computed from the per-employee comparison file (%d rows): mean top-3 overlap with the grouped method %.2f | same top reason %.1f%% | all three identical %.1f%%' % (len(k), k.top3_overlap.mean(), k.same_top1.mean() * 100, (k.top3_overlap == 1).mean() * 100))
else: print('\\nnative SHAP summary not found: run python scripts/run_shap_full430.py')""")
    # ---------------------------------------------------------------- 32-38
    n.stage(32, "Gaps between groups, measured with protected attributes only after the fact, never as inputs.")
    n.code("""fa = json.load(open(R[12] / 'fairness_results.json')); print('Task 12 mean twin gap %.6f -> %.6f (max %.4f -> %.4f); NDCG@5 %.3f -> %.3f' % (fa['before']['mean_abs_gap'], fa['after']['mean_abs_gap'], fa['before']['max_abs_gap'], fa['after']['max_abs_gap'], fa['quality_before']['ndcg@5'], fa['quality_after']['ndcg@5']))
print('\\nTask 13 attrition by gender'); print(pd.read_csv(R[13] / 'group_gender.csv').round(3).to_string(index=False)); print('\\nTask 15 pay gaps by gender'); print(pd.read_csv(R[15] / 'gender_gaps.csv').round(4).to_string(index=False))
print('\\nEqual accuracy is not equal treatment: a model can be equally accurate for two groups and still treat two individuals differently, so the twin test, not the accuracy table, is the fairness evidence.')""")
    n.stage(33, "Are the numbers trustworthy as numbers? Calibration of the attrition scores and the spread of the final bands.")
    n.code("""cc = pd.read_csv(R[13] / 'calibration_comparison.csv'); print(cc.round(4).to_string(index=False)); print()
print('final report: mean score by band'); risk.groupby('risk_band').risk_score_12m.agg(['count', 'mean', 'min', 'max']).round(3)""")
    n.stage(34, "Choose on evidence, not on the highest score alone.")
    n.code("""print('Task 13 verdict:', am['verdict']['text']); print(); pd.read_csv(R[13] / 'paired_differences.csv').round(4)""")
    n.stage(35, "The deliverables, produced by the saved models.")
    n.code("""print('Task 14 risk report rows:', len(risk), '| bands:', risk.risk_band.value_counts().to_dict()); print('Task 15 band recommendations on the test hires:', len(br)); print('Task 11 top-5 lists written:', len(pd.read_csv(R[11] / 'top5_per_job_all_resumes.csv')), 'rows'); risk[['employee_id', 'risk_score_12m', 'risk_band', 'top_reason_1']].head(5)""")
    n.stage(36, "The output is checked against the required format and read as the user would read it.")
    n.code("""same_cols = list(risk.columns) == list(sample.columns); ex = json.load(open(R[14] / 'explanation_run.json'))
print('columns identical to sample_attrition_risk:', same_cols); print('430 rows, unique employees, all in the scoring file:', len(risk) == 430, risk.employee_id.is_unique, set(risk.employee_id) == set(sc.employee_id))
print('every row carries a human-review note:', bool(risk.model_note.str.contains('review', case=False).all())); print('Task 14 automated checks passed: %d of %d' % (sum(c['passed'] for c in ex['checks']), len(ex['checks'])))
print('\\nfirst row read aloud:'); r = risk.iloc[0]; print(r.employee_id, r.risk_band, round(r.risk_score_12m, 3))
for k in (1, 2, 3): print('  -', r['top_reason_%d' % k])""")
    n.stage(37, "Artifacts, checksums, seeds and the environment each result was produced in.")
    n.code("""from cortexhr.phase3.attrition.v2 import file_sha256
rec = {'attrition_v2.joblib': am['artifact_sha256']['attrition_v2.joblib'], 'salary_band_v1.joblib': sm['sha256'], 'resume_ranker_v2.joblib': rm['sha256']}
print(pd.DataFrame([(k, file_sha256(P.MODELS / k) == v) for k, v in rec.items()], columns=['artifact', 'checksum matches metadata']).to_string(index=False))
print('seed:', SEED, '| model versions:', am['model_version'], sm['model_version'], rm['engine_version']); print('environment when attrition-v2 was trained:', am['environment'])""")
    n.stage(38, "What was found, what remains and what the work must not be used for.")
    n.code("""fin = pd.DataFrame([
 ('11', 'Ranking engine', 'precision@5 %.3f vs random %.3f' % (q.loc[q.index != 'random ordering', 'precision@5'].iloc[0], q.loc['random ordering', 'precision@5'])),
 ('12', 'Fairness audit', 'mean twin gap %.6f -> %.6f; NDCG@5 %.3f -> %.3f' % (fa['before']['mean_abs_gap'], fa['after']['mean_abs_gap'], fa['quality_before']['ndcg@5'], fa['quality_after']['ndcg@5'])),
 ('13', 'Attrition v2', 'gain over baseline: ' + ('proven' if am['verdict']['gain_is_meaningful'] else 'not proven') + ' (paired AUC diff %+.4f)' % am['verdict']['mean_diff']),
 ('14', 'Risk report', '%d employees with score, band, 3 reasons and a review note' % len(risk)),
 ('15', 'Pay bands', 'test MAE Rs %.0f, R2 %.3f; %.1f%% of offers inside the band' % (sm['test_metrics'][sm['model']]['MAE_rupees'], sm['test_metrics'][sm['model']]['R2'], sm['band_coverage_test'] * 100))], columns=['task', 'deliverable', 'finding']); print(fin.to_string(index=False))
print('''
Limitations: the attrition gain is not proven; reasons are drivers, not certainty; the pay target may contain past bias; group fairness measured on the supplied attributes only.
Must not be used for: automatic hiring, firing, promotion or pay decisions. A human reviews every output.''')""")
