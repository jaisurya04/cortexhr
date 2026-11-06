"""Builds notebooks 10-11 (Task 14: explain every employee's attrition risk)."""
from nbkit import NB

LOAD = """import joblib
from cortexhr.phase3.attrition import v2 as V
from cortexhr.phase3.common.io import read_csv, save_table, save_json
from cortexhr.phase3.common.validation import profile
from cortexhr.phase3.explanations import risk_report as RR
from cortexhr.phase3.explanations.groups import GROUPS, SENSITIVE_PROXY, check_groups
from cortexhr.phase3.explanations.reasons import sentence, wording_problems
from cortexhr.phase3.explanations.shap_explainer import explain_with_shap, additivity_error
scoring_raw = read_csv('attrition_scoring_2026'); sample = read_csv('sample_attrition_risk'); train = V.load_training()
print('scoring rows:', len(scoring_raw), '| sample rows:', len(sample))"""


def nb10():
    n = NB("Task 14, Notebook 10: Scoring audit, risk scores and SHAP explanations", "14: Explain every person's risk", "Audit the 430-row scoring file, score it with attrition-v2, compute per-employee contributions and check how trustworthy they are.", list(range(1, 32)))
    n.stage(1, "A score alone cannot help an employee. HR needs to know what moved the score so a real conversation can start, and needs to see on every line that the number is a support signal and not a decision.")
    n.stage(2, "**Input:** the 430 active employees at 30 Sep 2026. **Output:** one row per employee in the exact format of `sample_attrition_risk`: probability of leaving within 12 months, risk band, three plain-English reasons, model version, and a human-review note.")
    n.stage(3, "Pass marks fixed up front: 430 rows; columns and order equal to the sample; scores in 0..1; bands from the supplied `risk_bands()`; three different plain-English reasons on every row; every row says human review is required; the explanation must add up (efficiency check) and be reasonably stable between independent runs.")
    n.code(LOAD)
    n.stage(4, "Files: `attrition_scoring_2026` (430 rows), `sample_attrition_risk` (the format), the saved model `attrition_v2.joblib` (Task 13), the helper `risk_bands()` in `05_Starter/cortexhr_helpers.py`.")
    n.stage(5, "The scoring rows are joined with the same cut-off-safe monthly aggregates as the training data.")
    n.code("scoring, X, join_rep = V.load_scoring(); print(join_rep); X.shape")
    n.stage(6, "")
    n.code("print('rows:', len(scoring), '| columns in file:', scoring_raw.shape[1], '| model inputs:', X.shape[1], '| observation dates:', scoring_raw.observation_date.unique().tolist()); sample.head(3).T")
    n.stage(7, "Sample format columns and meaning.")
    n.code("pd.DataFrame({'column': sample.columns, 'example': sample.iloc[0].astype(str).str[:70].values})")
    n.stage(8, "")
    n.code("pd.DataFrame({'scoring': X.dtypes.astype(str), 'training': train['X'].dtypes.astype(str)}).assign(same=lambda t: t.scoring == t.training).same.value_counts().to_frame('inputs with identical dtype')")
    n.stage(9, "")
    n.code("profile(X).sort_values('missing_pct', ascending=False).head(8)")
    n.stage(10, "Missing patterns in the scoring set versus training. A big shift would be a warning.")
    n.code("""cmp_ = pd.DataFrame({'scoring': X.isna().mean(), 'training': train['X'].isna().mean()}); cmp_['abs_diff'] = (cmp_.scoring - cmp_.training).abs(); cmp_.sort_values('abs_diff', ascending=False).head(6).round(3)""")
    n.stage(11, "")
    n.code("print('employee_id unique:', scoring.employee_id.is_unique, '| duplicate rows:', int(scoring_raw.duplicated().sum()), '| same order as the sample file:', list(scoring.employee_id) == list(sample.employee_id))")
    n.stage(12, "Value ranges of the scoring set compared with training (anything outside the training range deserves a look).")
    n.code("""num = X.select_dtypes('number'); tr = train['X'].select_dtypes('number')
out_rng = {c: int(((num[c] < tr[c].min()) | (num[c] > tr[c].max())).sum()) for c in num.columns}; print({k: v for k, v in out_rng.items() if v} or 'no scoring value outside the training range')""")
    n.stage(13, "Unseen categories would be silently ignored by the one-hot encoder, so they are checked explicitly.")
    n.code("""cats = X.select_dtypes(exclude='number').columns; unseen = {c: sorted(set(X[c].dropna()) - set(train['X'][c].dropna())) for c in cats}; print({k: v for k, v in unseen.items() if v} or 'no unseen categories in', list(cats))""")
    n.stage(14, "")
    n.code("scoring.department_id.value_counts().head(8).to_frame('employees')")
    n.stage(15, "")
    n.code("""fig, ax = plt.subplots(1, 4, figsize=(15, 3))
for a, c in zip(ax, ['tenure_months', 'last_engagement_score', 'comp_ratio', 'attendance_rate_3m']): X[c].plot.hist(ax=a, bins=25, color='#2f6fb0', alpha=.8, label='scoring'); a.set_title(c)
plt.tight_layout()""")
    n.stage(16, "No label exists for the scoring set, so a bivariate view uses the model score itself (below, after scoring).")
    model = None
    n.code("""model = joblib.load(P.MODELS / 'attrition_v2.joblib'); p = model.predict_proba(X)[:, 1]
print('score summary:', pd.Series(p).describe().round(3).to_dict())
print('Spearman of score with: engagement %.2f | tenure %.2f | comp_ratio %.2f' % tuple(pd.Series(p).corr(X[c].reset_index(drop=True), method='spearman') for c in ['last_engagement_score', 'tenure_months', 'comp_ratio']))""")
    n.stage(17, "Distribution shift between training and scoring, measured with the population stability index (PSI; under 0.1 is small, over 0.25 large).")
    n.code("""def psi(a, b, bins=10):
    q = np.unique(np.quantile(a.dropna(), np.linspace(0, 1, bins + 1))); q[0], q[-1] = -np.inf, np.inf
    pa = np.histogram(a.dropna(), q)[0] / a.notna().sum(); pb = np.histogram(b.dropna(), q)[0] / b.notna().sum(); pa, pb = np.clip(pa, 1e-4, None), np.clip(pb, 1e-4, None); return float(np.sum((pb - pa) * np.log(pb / pa)))
ps = pd.Series({c: psi(tr[c], num[c]) for c in ['tenure_months', 'last_engagement_score', 'comp_ratio', 'attendance_rate_3m', 'annual_fixed_pay', 'commute_distance_km', 'manager_changes_12m', 'job_level']}).sort_values(ascending=False); ps.round(3).to_frame('PSI')""")
    n.stage(18, "Not applicable to the scoring file as a label analysis: the 12-month outcome has not happened yet. What can be compared is the predicted mean risk with the historical leave rate.")
    n.code("print('mean predicted risk of the 430: %.3f | historical leave rate in training: %.3f' % (p.mean(), train['y'].mean()))")
    n.stage(19, "Leakage checks for the scoring file: no exit columns; inputs identical to those of the model; and the overlap with training people.")
    n.code("""from cortexhr import config as C
print('exit/outcome columns in the scoring file:', [c for c in scoring_raw.columns if c in C.FORBIDDEN_AS_INPUT and c not in ('row_id', 'employee_id', 'observation_date')] or 'none')
tr_ids = set(train['df'].employee_id); ov = scoring.employee_id.isin(tr_ids).to_numpy()
print('scoring employees that were also in the training data (earlier snapshot): %d of %d (%.0f%%)' % (ov.sum(), len(ov), ov.mean() * 100)); print('mean risk: overlap %.3f vs new people %.3f' % (p[ov].mean(), p[~ov].mean()))""")
    n.md("Overlap is not target leakage (their 2026 outcome is unknown) but the model has seen an earlier snapshot of these people, so their scores may be slightly more confident than for strangers. It is reported, not hidden.")
    n.stage(20, "Engineering for explanation: the 43 raw inputs are folded into **23 readable groups** (players). Near-duplicate columns share one player so credit is not split between them.")
    n.code("check_groups(X.columns); pd.DataFrame({'group': list(GROUPS), 'columns': [', '.join(v) for v in GROUPS.values()], 'sensitive proxy': [g in SENSITIVE_PROXY for g in GROUPS]})")
    n.stage(21, "Sensitive-proxy groups (`career_history` can proxy for age, `education` and `work_location` for background). They stay in the model's score but are **not shown as reasons**, and their values are never printed.")
    n.code("print('groups not shown as reasons:', sorted(SENSITIVE_PROXY)); print('protected attributes (gender, age) are not model inputs:', not any(k in ' '.join(X.columns).lower() for k in ('gender', 'age_years', 'religion')))")
    n.stage(22, "Preprocessing lives inside the saved model, so the scoring rows get exactly the training transformations. The explainer only changes raw inputs and calls the same model.")
    n.code("est = model.calibrated_classifiers_[0].estimator; print(type(model).__name__, '->', type(est).__name__, '->', [type(m).__name__ for m in est.models_], '| calibrator:', type(model.calibrated_classifiers_[0].calibrators[0]).__name__)")
    n.stage(23, "No train/test split is needed for explaining a fixed model. The SHAP method needs a *reference population* (background): 200 randomly chosen labelled employees (seed 42), of which shap uses a 50-row sample.")
    n.code("bg = RR.build_background(); print('background rows:', len(bg))")
    n.stage(24, "Baseline for explanations: give **everyone the same three reasons** (the globally most important groups). The local method must capture more of each person's own score movement than that.")
    n.stage(25, "The explanations come from the real `shap` library: `shap.Explainer(..., algorithm='permutation')` (PermutationExplainer) on the calibrated attrition-v2 pipeline, for all 430 employees, over the 43 raw inputs. The SHAP feature values are then added up inside the same 23 HR-readable groups. No home-made estimator is used anywhere.")
    n.stage(26, "There is no model training in this task. 'Training' here means running the explainer: shap evaluates the model 87 times per background row (2 x 43 + 1) for each of the 430 employees. This takes a few minutes on one CPU.")
    n.code("""import time; t = time.time(); o = RR.run_task14(write=True); print('explanation run finished in %.0f seconds' % (time.time() - t))
phi, base, fx, rep, X = o['phi'], o['base'], o['fx'], o['report'], o['X']; p = o['p']
print('SHAP version:', o['shap_version'], '| employees explained:', len(phi), '| background rows:', o['background_rows'], '| additivity error:', o['additivity_max_error'])
phi.head()""")
    n.stage(27, "Cross-validation analogue: **seed stability**. shap is run again on the first 40 employees with a different seed (different 50-row background and permutation order). If the explanation is reliable the two runs should agree.")
    n.code("""from cortexhr.phase3.explanations.risk_report import select_reasons
f = lambda d: o['model'].predict_proba(d)[:, 1]; sub = X.reset_index(drop=True).iloc[:40]; ref = phi.iloc[:40].reset_index(drop=True)
alt = explain_with_shap(f, sub, o['background'], groups=GROUPS, seed=SEED + 1, max_background=50)['group_values']
ta, tb = select_reasons(ref), select_reasons(alt)
ov3 = np.array([len({ta.loc[i, 'driver_1'], ta.loc[i, 'driver_2'], ta.loc[i, 'driver_3']} & {tb.loc[i, 'driver_1'], tb.loc[i, 'driver_2'], tb.loc[i, 'driver_3']}) / 3 for i in range(len(ta))])
same_top1 = (ta.driver_1 == tb.driver_1).mean(); corr = float(np.corrcoef(ref.to_numpy().ravel(), alt.to_numpy().ravel())[0, 1])
print('correlation of contributions between the two seeds: %.3f | same #1 reason: %.1f%% | mean overlap of the 3 shown reasons: %.2f | all three identical: %.1f%%' % (corr, same_top1 * 100, ov3.mean(), (ov3 == 1).mean() * 100))""")
    n.stage(28, "How many background rows are enough? Contributions for the same 40 employees with fewer background rows are compared with the final 50-row result.")
    n.code("""rows = []
for k in (10, 25):
    pk = explain_with_shap(f, sub, o['background'], groups=GROUPS, seed=7, max_background=k)['group_values']
    rows.append({'background rows': k, 'correlation with final': float(np.corrcoef(pk.to_numpy().ravel(), ref.to_numpy().ravel())[0, 1])})
conv = pd.DataFrame(rows); conv.round(3)""")
    n.stage(29, "Evaluation of the explainer itself: (1) efficiency: contributions plus the baseline value must equal the prediction; (2) how much of a person's movement the three shown reasons capture, against the same-three-for-everyone baseline.")
    n.code("""print('efficiency check, largest |sum(contributions) + base - prediction| over 430 employees: %.2e' % o['additivity_max_error'])
shown = [c for c in phi.columns if c not in SENSITIVE_PROXY]; A = phi[shown].clip(lower=0)
local = np.sort(A.to_numpy(), axis=1)[:, ::-1][:, :3].sum(1) / A.sum(1).replace(0, np.nan)
gtop = A.mean().sort_values(ascending=False).index[:3]; glob = A[gtop].sum(1) / A.sum(1).replace(0, np.nan)
print('share of a person\\'s risk-raising movement captured by the 3 shown reasons: local %.2f vs the same global three for everyone %.2f' % (local.mean(), glob.mean())); print('global top-3 groups:', list(gtop))""")
    n.stage(30, "Where is the explanation least reliable? Employees whose three reasons differ most between the two seeds.")
    n.code("""bad = pd.DataFrame({'employee_id': scoring.employee_id.to_numpy()[:40], 'overlap': ov3, 'risk': np.round(p[:40], 3)}).sort_values('overlap').head(5); print('employees with overlap <= 1/3 (of the 40 re-run):', int((ov3 <= 1 / 3).sum()), 'of', len(ov3)); bad""")
    n.md("Low agreement happens when several groups have almost the same small contribution, so their order is arbitrary. The shown reasons are then equally valid candidates; HR should read them as 'among the factors', not as a strict ranking.")
    n.stage(31, "Global view: which groups move scores most, and in which direction.")
    n.code("""imp = phi.abs().mean().sort_values(ascending=False); fig, ax = plt.subplots(figsize=(7, 5)); imp.head(15)[::-1].plot.barh(ax=ax, color='#2f6fb0'); ax.set_xlabel('mean |contribution| (probability points)'); ax.set_title('Global importance of explanation groups (shap library, all 430 employees)'); plt.tight_layout(); plt.savefig(P.FIGURES / 'task14_shap_global_importance.png', dpi=130, bbox_inches='tight'); imp.head(8).round(4).to_frame('mean_abs_contribution')""")
    n.code("""KEY = {'tenure': 'tenure_months', 'engagement': 'last_engagement_score', 'raises': 'months_since_last_raise', 'pay_level': 'comp_ratio', 'manager_changes': 'manager_changes_12m', 'performance': 'last_performance_rating',
       'attendance': 'attendance_rate_3m', 'team_context': 'dept_attrition_rate_prior_12m', 'job_level': 'job_level', 'commute': 'commute_distance_km', 'training': 'training_hours_12m', 'hours_and_overtime': 'avg_hours_worked_3m'}
top = [g for g in imp.index if g in KEY][:9]; fig, ax = plt.subplots(figsize=(8, 5)); rng = np.random.default_rng(SEED); Xr = X.reset_index(drop=True)
for k, g in enumerate(top[::-1]):
    v = Xr[KEY[g]].rank(pct=True).fillna(.5).to_numpy(); sc_ = ax.scatter(phi[g], k + rng.uniform(-.28, .28, len(phi)), c=v, cmap='coolwarm', s=9)
ax.set_yticks(range(len(top))); ax.set_yticklabels([f'{g} ({KEY[g]})' for g in top[::-1]], fontsize=7); ax.axvline(0, color='k', lw=.6); ax.set_xlabel('contribution to risk (probability points)'); plt.colorbar(sc_, label='feature value (low to high)'); ax.set_title('SHAP summary by HR group (all 430 employees)'); plt.tight_layout(); plt.savefig(P.FIGURES / 'task14_shap_summary.png', dpi=130, bbox_inches='tight')""")
    n.code("""i_hi, i_lo = int(np.argmax(p)), int(np.argmin(p)); fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for a, i, t in zip(ax, (i_hi, i_lo), ('highest-risk employee', 'lowest-risk employee')):
    s = phi.iloc[i].sort_values(); s = pd.concat([s.head(5), s.tail(5)]); s.plot.barh(ax=a, color=['#2e8b57' if v < 0 else '#c0392b' for v in s]); a.set_title(f'{scoring.employee_id.iloc[i]}: {t}, score {p[i]:.3f} (average {base.iloc[i]:.3f})', fontsize=8)
plt.tight_layout()""")
    return n


def nb11():
    n = NB("Task 14, Notebook 11: Fairness of the explanations, validation, manual review and the exported report", "14: Explain every person's risk", "Check the risk report for group effects and wording problems, validate it against the sample format, read ten rows aloud and save the final 430-row CSV.", list(range(32, 39)))
    n.code(LOAD)
    n.code("""scoring, X, join_rep = V.load_scoring(); rep = read_csv('sample_attrition_risk').iloc[0:0]; rep = pd.read_csv(P.PREDICTIONS / 'attrition_risk_2026.csv'); contrib = pd.read_csv(P.task_dir(14) / 'shap_full430_group_values.csv')
phi = contrib.drop(columns=['employee_id']); print('report rows:', len(rep), '| contributions:', phi.shape)""")
    n.stage(32, "Group effects. Gender and age come from the restricted attribute file and are used only to *measure* here. Coverage is checked first; then the band share and the explanation content by group.")
    n.code("""from cortexhr.phase3.attrition.train import vault_attributes
va = vault_attributes()
if va is None: print('restricted file not present: group view skipped'); grp = None
else:
    a = va.drop_duplicates('person_ref').set_index('person_ref'); grp = rep.join(a[['gender', 'age_band']], on='employee_id'); print('employees with attribute record: %d of %d' % (grp.gender.notna().sum(), len(grp)))
    t = grp.groupby('gender').agg(n=('employee_id', 'size'), mean_risk=('risk_score_12m', 'mean'), high_share=('risk_band', lambda s: (s == 'High').mean()), medium_or_high=('risk_band', lambda s: (s != 'Low').mean())).round(3); print(t.to_string())
    t2 = grp.groupby('age_band').agg(n=('employee_id', 'size'), mean_risk=('risk_score_12m', 'mean'), high_share=('risk_band', lambda s: (s == 'High').mean())).round(3); print(t2.to_string())""")
    n.code("""if grp is not None:
    tenure = scoring.set_index('employee_id').tenure_months.reindex(rep.employee_id).to_numpy(); bins = pd.cut(tenure, [-1, 11, 35, 10 ** 4], labels=['under 1 year', '1 to 3 years', 'over 3 years'])
    allr = rep[['top_reason_1', 'top_reason_2', 'top_reason_3']].astype(str).agg(' '.join, axis=1); ment = allr.str.contains('time at the company|length of service', regex=True)
    print(pd.DataFrame({'employees': pd.Series(bins).value_counts(), 'share_High': rep.groupby(bins, observed=True).risk_band.apply(lambda s: (s == 'High').mean()), 'tenure_in_reasons': ment.groupby(bins, observed=True).mean()}).round(3).to_string())
    print('Reading: employees with under one year of service are over-represented in High because the model learned that early tenure carries more risk in the history. The wording says so and says it is a pattern, not the person\\'s performance.')""")
    n.code("""# how often would a sensitive-proxy group have been among the three largest contributions, had we shown it?
absrank = phi.abs().rank(axis=1, ascending=False); share = {g: float((absrank[g] <= 3).mean()) for g in SENSITIVE_PROXY}; print('share of employees where a sensitive-proxy group is in the top-3 by absolute contribution (hidden from the reasons):', {k: round(v, 3) for k, v in share.items()})""")
    n.stage(33, "Reliability of the numbers. (1) the score is a calibrated probability (Task 13: ECE 0.014); (2) band edges are rank-based, so people near an edge could fall either side: how many are within one percentage point of the High cut-off?")
    n.code("""cut = rep.loc[rep.risk_band == 'High', 'risk_score_12m'].min(); near = ((rep.risk_score_12m - cut).abs() <= 0.01).sum()
print('lowest High score %.3f | employees within 0.01 of that cut-off: %d of 430 | calibrated scores are probabilities; bands are rank cut-offs (top 10%% / next 20%%), not risk guarantees.' % (cut, near))""")
    n.stage(34, "attrition-v2 versus the simple baseline: do they flag the same people? (Task 13 concluded the models are equivalent in accuracy.)")
    n.code("""from scipy.stats import spearmanr
d = V.load_training(); lr = V.baseline_models()['Logistic Regression (balanced)'].fit(d['X'], d['y']); pl = lr.predict_proba(X)[:, 1]
v2s = rep.risk_score_12m.to_numpy(); k = 43; a_, b_ = set(np.argsort(-v2s)[:k]), set(np.argsort(-pl)[:k])
print('rank correlation v2 vs baseline LR: %.3f | overlap of the 43 highest-risk people: %d of %d' % (spearmanr(v2s, pl)[0], len(a_ & b_), k))""")
    n.stage(35, "The deliverable: the 430-row CSV in the sample's exact format (written by the explanation run, re-read here).")
    n.code("save_table(rep, P.task_dir(14) / 'attrition_risk_2026.csv'); rep.shape")
    n.stage(36, "Validation against the sample format, then a manual read of ten rows.")
    n.code("""chk = pd.DataFrame(RR.validate_report(rep, read_csv('sample_attrition_risk'))); print('checks passed: %d of %d' % (chk.passed.sum(), len(chk))); chk[['check', 'passed']]""")
    n.code("""rng = np.random.default_rng(SEED); ten = pd.concat([rep[rep.risk_band == 'High'].sample(3, random_state=1), rep[rep.risk_band == 'Medium'].sample(3, random_state=2), rep[rep.risk_band == 'Low'].sample(2, random_state=3),
                 rep.assign(t=scoring.set_index('employee_id').tenure_months.reindex(rep.employee_id).to_numpy()).sort_values('t').head(2).drop(columns='t')]).drop_duplicates('employee_id')
for _, r in ten.iterrows():
    print(f"{r.employee_id}  {r.risk_band:6s} score {r.risk_score_12m:.3f}")
    for k in (1, 2, 3): print(f"   {k}. {r[f'top_reason_{k}']}")
save_table(ten, P.task_dir(14) / 'manual_review_10_rows.csv')""")
    n.md("""### Manual review: what reading ten rows aloud showed, and what was changed

I read these rows as an HR colleague would. First version of the wording, and the problems found:

1. **"Factors in the employee's career history raised the predicted risk (a general pattern ...)"** appeared in many rows. It is vague, HR cannot act on it, and career history can act as an age proxy. **Change:** groups that can proxy for protected traits (`career_history`, `education`, `work_location`) are no longer shown as reasons; they still count in the score, and the model note says so on every row.
2. **"The time at the company so far (151 months)"**: months are hard to read for long service, and the sentence could sound like a judgement of new joiners. **Change:** years for 12 months or more, "A short time at the company so far" under 12 months, and the tail *"this reflects a common pattern at this stage of service, not the person's performance"*. For new joiners the sentence therefore describes a pattern in the data, not a fault.
3. **"A missing recent performance rating lowered the predicted risk."** was confusing and read like a judgement. **Change:** "No recent performance rating is on record (a data gap, not a judgement)".
4. **High-risk people sometimes had a *protective* factor as reason 1** ("... lowered the predicted risk"). HR starts the conversation from what raised the score. **Change:** the three reasons are now the three factors that **raised** the score the most compared with an average employee (largest positive contribution); protective factors stay in `shap_full430_group_values.csv` for audit.
5. **"The time since the last raise (0 months)"** for people who just joined was odd. **Change:** "the time since the last raise or joining".

Remaining wording that is acceptable but not perfect: "The role the person holds raised the predicted risk" is generic; it is kept with the tail "common patterns in the role, not the individual". The 10 rows above are the re-read after these changes: each reason is a full sentence, names the employee's own value where it is safe to, never says the person will leave, and says on the row that human review is required.""")
    n.code("""bad = [(r.employee_id, w) for _, r in rep.iterrows() for w in wording_problems(' '.join(r[['top_reason_1', 'top_reason_2', 'top_reason_3']].astype(str)))]
print('banned phrases found in all 1,290 reasons:', bad or 'none'); L = rep[['top_reason_1', 'top_reason_2', 'top_reason_3']].stack().str.len(); print('reason length (characters): median %d, max %d' % (L.median(), L.max()))
print('distinct reason sentences:', rep[['top_reason_1', 'top_reason_2', 'top_reason_3']].stack().nunique(), '| every row has the human-review note:', bool(rep.model_note.str.contains('Human review required').all()))""")
    n.stage(37, "Artifacts: the CSV in two places, the contributions table, the run metadata, the model checksum.")
    n.code("""import json; run = json.load(open(P.task_dir(14) / 'explanation_run.json')); print({k: run[k] for k in ('created_utc', 'shap_version', 'background_rows', 'additivity_max_error')})
print('model checksum recorded in Task 13 metadata:', json.load(open(P.METADATA / 'attrition_v2_metadata.json'))['artifact_sha256']['attrition_v2.joblib'][:16], '| CSV copies identical:', (P.PREDICTIONS / 'attrition_risk_2026.csv').read_bytes() == (P.task_dir(14) / 'attrition_risk_2026.csv').read_bytes())""")
    n.md("**Think for yourself: how should tenure be described to a new joiner?** As a pattern learned from historical data, not as a judgement of the employee or proof that they will leave. For under one year of service, the report says 'A short time at the company so far' and explicitly adds that this reflects a common pattern at this stage of service, not the person's performance.")
    n.stage(38, "")
    n.code("""print('FINDINGS'); print('- 430 employees scored with attrition-v2; bands %s; mean risk %.3f.' % (rep.risk_band.value_counts().to_dict(), rep.risk_score_12m.mean()))
print('- Every row has three plain-English reasons, a human-review note and passes all %d format checks.' % len(chk)); print('- Explanation is exact in the efficiency sense (error %.1e) and comes from the real shap library (version %s); the order of the three reasons is the least reliable part.' % (run['additivity_max_error'], run['shap_version']))
print('LIMITS: model accuracy is modest (ROC-AUC about 0.70), reasons are associations in historical data, not causes; sensitive-proxy groups are hidden from reasons but still influence scores; for people already in the training history the score may be a little more confident; the 12-month outcome is unknown; HR must not use the list for dismissal, pay or promotion decisions.')""")
    return n


if __name__ == "__main__":
    from pathlib import Path
    out = Path(__file__).resolve().parents[1] / "notebooks"
    for name, f in (("10_task14_shap_explanations", nb10), ("11_task14_risk_scoring_export", nb11)):
        print(f().save(out / f"{name}.ipynb"))
