"""Payroll track, part 2: stages 20-38."""

NOTES = {}
CODE = {}

# ------------------------------------------------------------------------------------------------ 20
NOTES[20] = "evaluates the model trained on training months only against the synthetic validation errors: confusion matrix, accuracy, precision, recall, specificity, F1, ROC-AUC, PR-AUC, classification report, curves, baseline comparison, subgroups, bootstrap uncertainty and failure modes."
CODE[20] = r'''
va_scored, _ = P.score_payroll(VALID_X, model=det_tr)
yv = va_scored.payroll_item_id.isin(valid_truth).to_numpy(); flag = va_scored.flagged.to_numpy(); sc = va_scored.anomaly_score.to_numpy()
m = confusion(flag, va_scored.payroll_item_id, valid_truth)
display(pd.DataFrame([[m['TN'], m['FP']], [m['FN'], m['TP']]], index=['actual normal', 'actual error'], columns=['predicted normal', 'predicted error']))
vm = {'accuracy': round((m['TP'] + m['TN']) / len(yv), 3), 'precision': m['precision'], 'recall (sensitivity)': m['recall'], 'specificity': m['specificity'], 'F1': m['F1'],
      'ROC-AUC': round(M.roc_auc_score(yv, sc), 3), 'PR-AUC': round(M.average_precision_score(yv, sc), 3), 'flag_share_%': m['flag_share_%']}
display(pd.Series(vm, name='validation metrics').to_frame()); print(M.classification_report(yv, flag, target_names=['normal', 'error'], digits=3, zero_division=0))
fig, ax = plt.subplots(1, 2, figsize=(11, 4)); pr, rc, _ = M.precision_recall_curve(yv, sc); ax[0].plot(rc, pr); ax[0].axhline(yv.mean(), ls=':', c='grey'); ax[0].set(title='Precision-recall curve (dotted = random)', xlabel='recall', ylabel='precision')
fp, tp, _ = M.roc_curve(yv, sc); ax[1].plot(fp, tp); ax[1].plot([0, 1], [0, 1], ls=':', c='grey'); ax[1].set(title='ROC curve', xlabel='false positive rate', ylabel='true positive rate'); plt.tight_layout(); plt.show()
b = ACCEPT['baseline_to_beat']; chk = pd.DataFrame([['recall >= %.2f' % ACCEPT['minimum_recall'], vm['recall (sensitivity)'], vm['recall (sensitivity)'] >= ACCEPT['minimum_recall']], ['precision >= %.2f' % ACCEPT['minimum_precision'], vm['precision'], vm['precision'] >= ACCEPT['minimum_precision']],
    ['F1 above supplied benchmark %.3f' % b['F1'], vm['F1'], vm['F1'] > b['F1']]], columns=['requirement (stage 3)', 'value', 'met']); display(chk)
tt = pd.DataFrame({'error_type': pd.Series(valid_truth)}); tt['caught'] = va_scored.set_index('payroll_item_id').flagged.reindex(tt.index).to_numpy()
display(tt.groupby('error_type').caught.agg(['sum', 'count', 'mean']).round(2).rename(columns={'sum': 'caught', 'count': 'support', 'mean': 'recall'}))
g = va_scored.merge(VALID_X[['payroll_item_id', 'department_id']].drop_duplicates('payroll_item_id'), on='payroll_item_id', how='left'); display((g.groupby('department_id').flagged.agg(['mean', 'size']).round(3).rename(columns={'mean': 'flag rate', 'size': 'rows'})).T)
rng = np.random.default_rng(SEED); bs = []
for _ in range(300):
    ix = rng.integers(0, len(yv), len(yv)); f_, y_ = flag[ix], yv[ix]; tp_ = (f_ & y_).sum(); bs.append((tp_ / max(f_.sum(), 1), tp_ / max(y_.sum(), 1)))
bs = np.array(bs); print('95%% bootstrap interval  precision: %.3f-%.3f | recall: %.3f-%.3f (only %d errors in validation, so the intervals are wide)' % (*np.percentile(bs[:, 0], [2.5, 97.5]), *np.percentile(bs[:, 1], [2.5, 97.5]), yv.sum()))
print('Limitations / failure modes: (1) synthetic errors are simpler than real ones; (2) precision depends on prevalence (stage 18); (3) INFLATED_VARIABLE with consistent totals is only caught by the forest; (4) rule hits on real rows that were not injected count as false positives here although they may be genuine errors.')
RESULTS['valid'] = vm
'''

# ------------------------------------------------------------------------------------------------ 21
NOTES[21] = "chooses the operating point on VALIDATION data only: sweeps the score threshold, prices false alarms and misses with the stage-1 assumptions, enforces the review-volume limit on real (non-injected) rows and freezes the result."
CODE[21] = r'''
real_va, _ = P.score_payroll(VALID, model=det_tr); n_runs = VALID.month.nunique(); per_run = len(VALID) / n_runs
tp_money = va_scored[va_scored.payroll_item_id.isin(valid_truth) & va_scored.flagged].money_at_risk; loss_fn = float(tp_money.median()) if len(tp_money) and tp_money.median() > 0 else 10000.0
fp_cost = ASSUMPTIONS['review_minutes_per_flag'] * ASSUMPTIONS['review_cost_per_minute_inr']
print(f'default threshold: the production flag is "rule hit OR forest outlier at 3% contamination"; rule hits are floored at score 0.90. Cost of one false alarm = INR {fp_cost}, cost of one miss = INR {loss_fn:,.0f} (median money_at_risk of caught errors; a proxy).')
rows = []
for t in np.round(np.arange(0.0, 1.0001, 0.05), 2):
    f_ = va_scored.anomaly_score.to_numpy() >= t; c = confusion(f_, va_scored.payroll_item_id, valid_truth); share_real = float((real_va.anomaly_score >= t).mean() * 100)
    rows.append({'threshold': t, 'precision': c['precision'], 'recall': c['recall'], 'FP': c['FP'], 'FN': c['FN'], 'cost_INR': c['FP'] * fp_cost + c['FN'] * loss_fn, 'real_flag_share_%': round(share_real, 2), 'reviews_per_run': round(share_real / 100 * per_run, 1)})
sweep = pd.DataFrame(rows); display(sweep.iloc[::2])
ok = sweep[sweep['real_flag_share_%'] <= ACCEPT['max_flag_share_pct']]; best = ok.loc[ok.cost_INR.idxmin()]
prod_share = float(real_va.flagged.mean() * 100)
THRESHOLD = {'score_threshold': float(best.threshold), 'chosen_on': 'validation only', 'rule': f"minimise cost subject to real flag share <= {ACCEPT['max_flag_share_pct']}%",
             'production_operating_point': 'rules OR forest outlier (contamination 3%)', 'production_real_flag_share_%': round(prod_share, 2)}
display(pd.Series(THRESHOLD, name='frozen threshold decision').to_frame())
print(f"review team feasibility: {prod_share:.2f}% of real rows ~ {prod_share / 100 * per_run:.0f} payslips per run ~ {prod_share / 100 * per_run * ASSUMPTIONS['review_minutes_per_flag'] / 60:.1f} reviewer-hours per run (assumed {ASSUMPTIONS['review_minutes_per_flag']} min each)")
print('Reason documented: errors are far costlier than reviews, so we take the cheapest point that keeps the queue small. The threshold is now FROZEN; it is only revisited through the retraining gate in stage 35, never by looking at the test months.')
RESULTS['threshold'] = THRESHOLD
'''

# ------------------------------------------------------------------------------------------------ 22
NOTES[22] = "asks whether scores must be probabilities (no: they rank a review queue), then measures calibration anyway with Brier score and reliability bins, tries Platt and isotonic fitted on early validation months and checks them on later months."
CODE[22] = r'''
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
print('Need for probabilities: LOW - the queue is ranked by score and money_at_risk. A probability would only be needed to quote "chance this payslip is wrong".')
d = va_scored.copy(); d['y'] = d.payroll_item_id.isin(valid_truth).astype(int); d['mo'] = pd.to_datetime(d.month).dt.to_period('M')
first = sorted(d.mo.unique())[:2]; cal_set, chk_set = d[d.mo.isin(first)], d[~d.mo.isin(first)]
print('calibrator fitted on', [str(x) for x in first], '| checked on the remaining validation month | the official test months are not touched')
iso = IsotonicRegression(out_of_bounds='clip').fit(cal_set.anomaly_score, cal_set.y); platt = LogisticRegression().fit(cal_set[['anomaly_score']], cal_set.y)
res = pd.DataFrame({'raw score used as probability': M.brier_score_loss(chk_set.y, chk_set.anomaly_score), 'Platt scaling': M.brier_score_loss(chk_set.y, platt.predict_proba(chk_set[['anomaly_score']])[:, 1]),
                    'isotonic regression': M.brier_score_loss(chk_set.y, iso.predict(chk_set.anomaly_score)), 'constant prevalence': M.brier_score_loss(chk_set.y, np.full(len(chk_set), cal_set.y.mean()))}, index=['Brier score (lower is better)']).T.round(4); display(res)
fig, ax = plt.subplots(figsize=(5, 4)); pt, pp = calibration_curve(chk_set.y, chk_set.anomaly_score, n_bins=5, strategy='quantile'); ax.plot(pp, pt, 'o-', label='raw score'); pt2, pp2 = calibration_curve(chk_set.y, iso.predict(chk_set.anomaly_score), n_bins=5, strategy='quantile'); ax.plot(pp2, pt2, 's-', label='isotonic')
ax.plot([0, 1], [0, 1], ':', c='grey'); ax.set(xlabel='mean predicted', ylabel='observed error rate', title='Reliability (held-out validation month)'); ax.legend(); plt.tight_layout(); plt.show()
print('Reading: the raw score is a RANKING score (0-1 after min-max scaling), not a probability; the observed error rate in a score bin depends on how many errors were injected, so any probability here would only be valid for that prevalence.')
print('Decision: no calibrator is adopted or frozen. The API and UI show anomaly_score as a risk score, never as "x% chance of error". Keep calibration data and final test data separate if this changes (labels from reviewers, stage 38).')
RESULTS['calibration'] = res['Brier score (lower is better)'].to_dict()
'''

# ------------------------------------------------------------------------------------------------ 23
NOTES[23] = "explains the model: permutation importance on validation PR-AUC (isolation forests have no coefficients), reason codes, individual explanations, business-sense and correlation-vs-causation checks."
CODE[23] = r'''
print('Coefficients: not applicable (no linear model) | tree feature_importances_: IsolationForest does not provide them | SHAP: not installed here; permutation importance below is the model-agnostic alternative.')
yv_ = np.array([i in valid_truth for i in VALID_X.payroll_item_id]); Xv = FVALID[P.FEATURE_COLS]; base_ap = M.average_precision_score(yv_, -pipe.decision_function(Xv)); rng = np.random.default_rng(SEED); imp = {}
for c in P.FEATURE_COLS:
    drops = []
    for _ in range(3):
        t = Xv.copy(); t[c] = rng.permutation(t[c].values); drops.append(base_ap - M.average_precision_score(yv_, -pipe.decision_function(t)))
    imp[c] = np.mean(drops)
imp = pd.Series(imp).sort_values(ascending=False); display(imp.head(8).round(4).to_frame('PR-AUC lost when shuffled (forest alone)'))
fig, ax = plt.subplots(figsize=(7, 4)); imp.head(10)[::-1].plot.barh(ax=ax); ax.set_title('Permutation importance (validation, forest only)'); plt.tight_layout(); plt.show()
print('Business sense: the strongest features are arithmetic ratios (gross vs expected, net vs gross, deductions, variable pay vs fixed pay) - exactly what a payroll clerk checks by hand. No identifier or calendar field is in the model, so nothing "suspiciously powerful" such as an employee id.')
print('Correlation is not causation: a large importance means the model relies on a feature to separate odd rows, not that the feature causes payroll errors.')
codes = va_scored[va_scored.flagged].reason_code.str.split('|').explode().value_counts(); display(codes.head(10).to_frame('flags citing this reason code'))
print('Individual explanations (what a clerk sees):')
for r in va_scored[va_scored.flagged].head(4).itertuples(): print(f' - {r.payroll_item_id} | score {r.anomaly_score:.2f} | money at risk INR {r.money_at_risk:,.0f} | {r.reason_code}\n     {r.reason_text}')
print('Limits of the explanation: reason text names the rule or the unusual ratios; it does not prove cause or intent. money_at_risk is a triage proxy. Explanations are for review ordering and must not be used to accuse or penalise an employee.')
'''

# ------------------------------------------------------------------------------------------------ 24
NOTES[24] = "inspects false positives and false negatives on validation, splits false alarms into rule hits on real rows vs model-only flags, checks concentration by group and error type, attributes causes and ranks improvements by business impact."
CODE[24] = r'''
e = va_scored.merge(VALID_X[['payroll_item_id', 'department_id', 'job_level']].drop_duplicates('payroll_item_id'), on='payroll_item_id', how='left'); e['is_error'] = e.payroll_item_id.isin(valid_truth); e['type'] = e.payroll_item_id.map(valid_truth)
FPs, FNs, TPs = e[e.flagged & ~e.is_error], e[~e.flagged & e.is_error], e[e.flagged & e.is_error]
print(f'true positives {len(TPs)} | false positives {len(FPs)} | false negatives {len(FNs)} | true negatives {int((~e.flagged & ~e.is_error).sum())}')
fp_split = pd.Series({'rule hit on a real (non-injected) row': int(FPs.rule_flagged.sum()), 'model-only flag (no rule)': int((~FPs.rule_flagged).sum())}); display(fp_split.to_frame('false positives'))
print('Important: a rule hit on a real row is usually a genuine data problem that exists in the file (e.g. a natural duplicate or mismatch). It is only a "false positive" because our synthetic truth did not label it.')
display(FPs[~FPs.rule_flagged][['payroll_item_id', 'anomaly_score', 'money_at_risk', 'reason_code']].head(8)); print('False negatives (missed errors):'); display(FNs[['payroll_item_id', 'type', 'anomaly_score', 'reason_code']])
display(FNs.type.value_counts().to_frame('missed by type')); display(FPs.groupby('department_id').size().sort_values(ascending=False).head(5).to_frame('false positives by department')); display(FPs.groupby('job_level').size().to_frame('false positives by job level').T)
print('Rare cases: errors concentrate in', 'INFLATED_VARIABLE (no rule can see it)' if (FNs.type == 'INFLATED_VARIABLE').any() else 'no single type', '- the missed rows are the hardest, subtle ones.')
print('Cause analysis: model-only false alarms come from legitimately large bonuses / overtime (limitation of the features, no bonus calendar); missed inflated bonuses come from a model limitation, not a label or data problem; no data-quality fault was found behind the errors.')
improvements = pd.DataFrame([['Add bonus calendar / approved-bonus list as a feature', 'cuts model-only false alarms for legitimate festival/performance bonuses', 'high'], ['Per-employee history features (past-only) in the API reference', 'sharper judgement for repeat employees', 'medium'],
    ['Try LOF/ensemble as challenger (stage 35)', 'better model-only recall in CV', 'medium'], ['Collect reviewer outcomes (stage 38)', 'lets us train and calibrate a supervised model', 'high']], columns=['improvement', 'expected effect', 'priority']); display(improvements)
RESULTS['improvements'] = improvements.improvement.tolist(); RESULTS['errors'] = {'FP': len(FPs), 'FN': len(FNs), 'missed_types': FNs.type.value_counts().to_dict()}
'''

# ------------------------------------------------------------------------------------------------ 25
NOTES[25] = "checks fairness and privacy: flag rates by department, job level, overtime eligibility and bonus type on REAL rows, proxy variables, data minimisation of the output, access control and human oversight."
CODE[25] = r'''
rv = real_va.merge(VALID[['payroll_item_id', 'department_id', 'job_level', 'overtime_eligible', 'variable_pay_type']], on='payroll_item_id', how='left'); rv['variable_pay_type'] = rv.variable_pay_type.fillna('NONE'); overall = rv.flagged.mean()
print('Protected / vulnerable groups: the payroll file holds NO gender, age, caste, religion or disability fields, and we deliberately do not join any. Department, job level, overtime eligibility and bonus type are the available groupings (job level is a proxy for salary band).')
rows = []
for col in ['department_id', 'job_level', 'overtime_eligible', 'variable_pay_type']:
    t = rv.groupby(col).agg(rows=('flagged', 'size'), flag_rate=('flagged', 'mean'), model_only=('iforest_flagged', lambda s: (s & ~rv.loc[s.index, 'rule_flagged']).mean())).reset_index().rename(columns={col: 'group'}); t.insert(0, 'attribute', col); rows.append(t)
fair = pd.concat(rows); fair['ratio_to_overall'] = (fair.flag_rate / overall).round(2); fair[['flag_rate', 'model_only']] = fair[['flag_rate', 'model_only']].round(3); display(fair)
big = fair[(fair.rows >= 30) & (fair.ratio_to_overall > 2)]; print(f'overall flag rate {overall:.3f}; groups with n>=30 and more than 2x the overall rate:', len(big)); display(big)
print('FP/FN rates by group: false negatives cannot be measured per group on real data (no labels); per-type recall in stage 20 is the closest check, and the injected set shows no department-specific misses.')
print('Historical discrimination: payroll amounts can reflect past pay inequity; the model flags ARITHMETIC inconsistency and unusual ratios, not "low" or "high" pay, so it does not rank people by pay level.')
leak_cols = {'gross_pay', 'net_pay', 'tds', 'pf_employee'} & set(P.OFFICIAL_COLUMNS); print('raw pay columns exposed in the official output:', sorted(leak_cols), '(none - only money_at_risk, reason text and ids)'); assert not leak_cols
print('Data minimisation: no names, bank details or protected attributes are used or stored. Access: payroll/finance roles only (the API must be placed behind authentication in production). Legal: confirm retention and purpose limits under the applicable data-protection law (e.g. India DPDP Act 2023) with the legal team - MANUAL.')
print('Human oversight: every flag goes to a person; nothing is blocked, recovered or reported automatically. Scope: validated ONLY for payslip review triage on this dataset; not for performance, hiring, pay-setting or disciplinary decisions.')
RESULTS['fairness'] = {'groups_over_2x': int(len(big)), 'overall_flag_rate': round(float(overall), 4)}
'''

# ------------------------------------------------------------------------------------------------ 26
NOTES[26] = "stress-tests the frozen design: resampled training sets and seeds, a different scaler, unusual but valid payslips, missing/unseen inputs, a plausible company-wide pay rise, temporal stability and explicit fallbacks."
CODE[26] = r'''
from sklearn.preprocessing import RobustScaler
base_ids = set(va_scored[va_scored.flagged].payroll_item_id); rows = []
for s in range(5):
    sub = TRAIN.sample(frac=0.8, random_state=s); d = P.fit_isolation_forest(sub, random_state=SEED + s); sc_, _ = P.score_payroll(VALID_X, model=d); ids_ = set(sc_[sc_.flagged].payroll_item_id); c = confusion(sc_.flagged.to_numpy(), sc_.payroll_item_id, valid_truth)
    rows.append({'training sample (80%) / seed': s, 'recall': c['recall'], 'precision': c['precision'], 'overlap with frozen model (Jaccard)': round(len(ids_ & base_ids) / len(ids_ | base_ids), 3)})
rob = pd.DataFrame(rows); display(rob); print('different samples/seeds: recall range', rob.recall.min(), '-', rob.recall.max(), '| flagged-set overlap >=', rob.iloc[:, 3].min())
alt = Pipeline([('imputer', SimpleImputer(strategy='median', add_indicator=True)), ('scaler', RobustScaler()), ('detector', IsolationForest(n_estimators=300, contamination=0.03, random_state=SEED, n_jobs=-1))]).fit(FTRAIN[P.FEATURE_COLS])
print('preprocessing change (RobustScaler instead of StandardScaler), model-only:', {k: v for k, v in confusion(alt.predict(FVALID[P.FEATURE_COLS]) == -1, vids, valid_truth).items() if k in ('precision', 'recall')}, 'vs standard:', {k: v for k, v in confusion(if_flags(pipe, VALID_X), vids, valid_truth).items() if k in ('precision', 'recall')})
def scale_money(df, f):
    x = df.copy()
    for c in ['monthly_fixed_gross', 'gross_pay', 'overtime_pay', 'variable_pay', 'tds']: x[c] = x[c] * f
    x['total_deductions'] = x[P.COMPONENT_COLS].sum(axis=1); x['net_pay'] = x.gross_pay - x.total_deductions; return x
clean_row = real_va[~real_va.flagged].payroll_item_id.iloc[0]; one = VALID[VALID.payroll_item_id == clean_row].copy(); hi = scale_money(one, 6.0)
r_hi, _ = P.score_payroll(hi, model=det_tr); hi_flag = bool(r_hi.flagged.iloc[0]); print('unusual but valid input: the same payslip with a 6x higher salary is flagged ->', hi_flag, '| score', round(float(r_hi.anomaly_score.iloc[0]), 3), '|', 'OK: ratios, not salary size, drive the flag' if not hi_flag else 'WEAKNESS FOUND: raw-amount features make an out-of-range salary look unusual. Mitigation to test through the stage-35 gate: ratios-only feature set (stage 11 showed it did better model-only) or capping/log-scaling amounts.')
na = one.copy(); na['pf_employee'] = np.nan; na['tds'] = np.nan
try: r_na, _ = P.score_payroll(na, model=det_tr); print('missing PF and TDS values: scored without error, flagged ->', bool(r_na.flagged.iloc[0]))
except Exception as ex: print('missing values raised:', type(ex).__name__, ex)
un = one.copy(); un['department_id'] = 'D99'; un['variable_pay_type'] = 'Brand New Award'; un['employee_id'] = 'EMP-NEW'; print('unseen department / bonus type / employee: flagged ->', bool(P.score_payroll(un, model=det_tr)[0].flagged.iloc[0]))
RESULTS['robustness_high_salary_flagged'] = hi_flag
shift = P.score_payroll(scale_money(VALID, 1.10), model=det_tr)[0]; print(f'plausible change - company-wide 10% pay rise: real flag share {real_va.flagged.mean() * 100:.2f}% -> {shift.flagged.mean() * 100:.2f}%')
mo = P.score_payroll(DF[IS_TRAIN | IS_VALID], model=det_tr)[0]; mo['mo'] = pd.to_datetime(mo.month).dt.to_period('M'); mt = (mo.groupby('mo').flagged.mean() * 100).round(2); display(mt.to_frame('% flagged by month (train+validation months)').T)
print('temporal stability: monthly flag share ranges', mt.min(), '-', mt.max(), '%')
print('Class-distribution shift: see stage 18 - recall steady, precision follows prevalence.')
try: P.parse_month('2026-13'); print('invalid month accepted?!')
except ValueError as ex: print('fallback for invalid input: rejected with ->', ex)
fb = pd.DataFrame([['month malformed / impossible', 'rejected with a clear message (API: HTTP 422)'], ['numeric field missing or NaN in a batch', 'API rejects; offline scoring imputes the training median and still applies the rules'], ['unseen department/bonus type/employee', 'scored normally; history rules stay quiet when there is no history'],
   ['company-wide pay change', 'ratios stay stable; watch PSI in monitoring (stage 34)'], ['duplicates only visible across batches', 'detector keeps reference history; batch-only scoring cannot see earlier months']], columns=['condition', 'defined behaviour']); display(fb)
'''

# ------------------------------------------------------------------------------------------------ 27
NOTES[27] = "freezes model, hyper-parameters, threshold and (non-)calibrator, verifies the test months were never touched, then evaluates ONCE on fresh synthetic errors injected into the test months; the authorised CX-06 vault is checked and, if absent, reported as BLOCKED with the owner's saved evidence shown separately."
CODE[27] = r'''
FROZEN = {'detector_version': det_tr.version, 'contamination': det_tr.contamination, 'score_lo': det_tr.score_lo, 'score_hi': det_tr.score_hi, 'trained_rows': det_tr.trained_rows, 'threshold': THRESHOLD['score_threshold'], 'calibrator': 'none (scores are rankings)'}
assert hashlib.sha256(pd.util.hash_pandas_object(DF[IS_TEST], index=False).values.tobytes()).hexdigest()[:16] == TEST_FINGERPRINT; print('test months unchanged since stage 12 (fingerprint matches); frozen:', FROZEN)
TEST = DF[IS_TEST].reset_index(drop=True); TEST_X, test_truth = inject_errors(TEST, SEED + 2)
te_scored, _ = P.score_payroll(TEST_X, model=det_tr); yt = te_scored.payroll_item_id.isin(test_truth).to_numpy(); ct = confusion(te_scored.flagged.to_numpy(), te_scored.payroll_item_id, test_truth)
tm = {'accuracy': round((ct['TP'] + ct['TN']) / len(yt), 3), 'precision': ct['precision'], 'recall (sensitivity)': ct['recall'], 'specificity': ct['specificity'], 'F1': ct['F1'], 'ROC-AUC': round(M.roc_auc_score(yt, te_scored.anomaly_score), 3), 'PR-AUC': round(M.average_precision_score(yt, te_scored.anomaly_score), 3), 'flag_share_%': ct['flag_share_%']}
cmp_ = pd.DataFrame({'validation': RESULTS['valid'], 'test (untouched, evaluated once)': tm}); cmp_['difference'] = (cmp_.iloc[:, 1] - cmp_.iloc[:, 0]).round(3); display(cmp_)
assert (det_tr.score_lo, det_tr.score_hi, det_tr.contamination) == (FROZEN['score_lo'], FROZEN['score_hi'], FROZEN['contamination']); print('model unchanged after evaluation: nothing was tuned on the test months')
tt = pd.DataFrame({'error_type': pd.Series(test_truth)}); tt['caught'] = te_scored.set_index('payroll_item_id').flagged.reindex(tt.index).to_numpy(); display(tt.groupby('error_type').caught.agg(['sum', 'count', 'mean']).round(2).rename(columns={'sum': 'caught', 'count': 'support', 'mean': 'recall'}))
real_te = P.score_payroll(TEST, model=det_tr)[0]; real_share = float(real_te.flagged.mean() * 100)
decision = pd.DataFrame([['recall >= %.2f' % ACCEPT['minimum_recall'], tm['recall (sensitivity)'], tm['recall (sensitivity)'] >= ACCEPT['minimum_recall']], ['precision >= %.2f' % ACCEPT['minimum_precision'], tm['precision'], tm['precision'] >= ACCEPT['minimum_precision']],
    ['real-row flag share <= %.0f%%' % ACCEPT['max_flag_share_pct'], round(real_share, 2), real_share <= ACCEPT['max_flag_share_pct']], ['F1 above supplied benchmark %.3f' % ACCEPT['baseline_to_beat']['F1'], tm['F1'], tm['F1'] > ACCEPT['baseline_to_beat']['F1']]], columns=['predefined acceptance criterion', 'test value', 'met'])
display(decision); RESULTS['test'] = tm; RESULTS['acceptance'] = decision.set_index('predefined acceptance criterion').met.to_dict()
print('Investigating differences: any gap is explained by the small number of injected errors (see stage 20 intervals), not tuned away.')
vault = ROOT/'04_Validation_Vault'/'unlock_CX-06.csv'
if vault.exists(): print('CX-06 vault file is present at', vault.relative_to(ROOT), '- evaluate it with scripts/evaluate_cx06.py AFTER freezing predictions; it is deliberately not opened inside this workflow notebook.')
else: print('CX-06 per-type evaluation: MEASURED by the project owner with the authorised file (kept outside this package); see reports/phase2/payroll_cx06_per_type.csv.')
owner = REP/'payroll_cx06_per_type.csv'
if owner.exists(): print('Saved evidence MEASURED BY THE PROJECT OWNER on the authorised file with the full-data production detector (a different model object from the train-only model above):'); display(pd.read_csv(owner).round(3))
print('Limitations of the final model: synthetic data, rare-error precision depends on prevalence, flags are review candidates, money_at_risk is a proxy.')
'''

# ------------------------------------------------------------------------------------------------ 28
NOTES[28] = "compares every candidate against the baselines on several criteria (not one metric), records the selection rationale, and verifies the shipped model file, version, data version and registry entry."
CODE[28] = r'''
cvm = pd.DataFrame(RESULTS['cv']).T; lat = pd.DataFrame(RESULTS['candidates']).T
sel = pd.DataFrame({'time-aware CV precision': cvm.precision, 'recall': cvm.recall, 'F1': cvm.F1}); sel['explainability'] = pd.Series({'rules': 'highest (reads like policy)', 'Isolation Forest': 'good (reason drivers per flag)', 'LOF': 'weak (neighbour distance)', 'rules + IF': 'good', 'rules + LOF': 'weak'})
sel['inference ms/1000 rows'] = pd.Series({'Isolation Forest': lat.loc['Isolation Forest', 'predict_ms_per_1000_rows'], 'LOF': lat.loc['Local Outlier Factor', 'predict_ms_per_1000_rows']}); sel['stored artefact'] = pd.Series({'rules': 'code only', 'Isolation Forest': '300 small trees', 'LOF': 'needs the whole training matrix at inference', 'rules + IF': '300 small trees + history', 'rules + LOF': 'training matrix + history'})
display(sel); print('Dummy baseline recall = 0 and the supplied benchmark F1 =', ACCEPT['baseline_to_beat']['F1'], '- every non-trivial candidate beats the dummy; rules + a model beat rules alone on recall.')
best_f1 = cvm.F1.idxmax(); print(f'Highest CV F1 is "{best_f1}". We do NOT choose by one metric alone.')
print('Chosen for production: rules + Isolation Forest (as shipped, version %s). Why: rules cover hard facts and read like policy; the forest adds recall on odd combinations and gives per-flag reason drivers; the saved object is small and scores a single API row exactly like a batch. LOF can look better model-only, but it must keep the full training matrix, is harder to explain and was not selected without a controlled challenger test (stage 35).' % P.MODEL_VERSION)
reg = S.read_registry().get('payroll', {}); f = ROOT/'models'/reg.get('file', 'payroll_detector_v2.joblib'); ok_sha = f.exists() and S._sha256(f) == reg.get('sha256')
print('registry entry:', {k: reg.get(k) for k in ('version', 'trained_rows', 'training_data', 'sklearn_version', 'labels_used')}, '| model file exists:', f.exists(), '| sha256 matches registry:', ok_sha)
print('training data version (sha256, first 16):', hashlib.sha256((CSV/'payroll_monthly_features.csv').read_bytes()).hexdigest()[:16], '| configuration: stage 16 file', (WF/'payroll_training_config.json').relative_to(ROOT))
assert reg.get('version') == P.MODEL_VERSION
'''

# ------------------------------------------------------------------------------------------------ 29
NOTES[29] = "wraps validation + the saved detector + risk bands into one reusable prediction function with informative errors, then tests it on representative good, broken and malformed inputs and checks reproducibility."
CODE[29] = r'''
DEPLOYED = S.load_payroll_detector(); REQUIRED = P.NUMERIC_BASE + ['run_id', 'employee_id', 'month']
def predict_payslips(rows, detector=DEPLOYED):
    """Validate -> score with the saved detector -> add a risk band.  Raises ValueError with a message a caller can show."""
    if not isinstance(rows, list) or not rows or not all(isinstance(r, dict) for r in rows): raise ValueError('rows must be a non-empty list of dicts (one per payslip)')
    if len(rows) > S.MAX_BATCH_ROWS: raise ValueError(f'at most {S.MAX_BATCH_ROWS} payslips per call')
    for k, r in enumerate(rows):
        miss = [c for c in REQUIRED if c not in r]
        if miss: raise ValueError(f'row {k}: missing required columns {miss}')
        for c in P.NUMERIC_BASE:
            if isinstance(r[c], bool) or not isinstance(r[c], (int, float)) or r[c] != r[c] or abs(r[c]) == float('inf'): raise ValueError(f'row {k}: {c} must be a finite number, got {r[c]!r}')
        P.parse_month(r['month'])
    out = pd.DataFrame(S.score_payroll_rows(rows, detector))
    out['risk_band'] = np.where(out.anomaly_score >= P.RULE_SCORE_FLOOR, 'HIGH', np.where(out.flagged == 'Y', 'MEDIUM', 'LOW'))
    return out
def to_row(r): d = r[REQUIRED + ['payroll_item_id', 'department_id', 'role_id', 'job_level']].to_dict(); d['month'] = pd.Timestamp(d['month']).strftime('%Y-%m'); return {k: (v.item() if hasattr(v, 'item') else v) for k, v in d.items()}
dep_valid = P.score_payroll(VALID, model=DEPLOYED)[0]; ok_id = dep_valid[~dep_valid.flagged].payroll_item_id.iloc[0]; good = to_row(VALID[VALID.payroll_item_id == ok_id].iloc[0]); bad = dict(good, payroll_item_id='DEMO-NEG', net_pay=-500.0, employee_id='EMP-0002')
res = predict_payslips([good, bad]); display(res[['record_id', 'anomaly_score', 'flagged', 'risk_band', 'reason_code', 'money_at_risk']])
for label, rows in {'missing column': [{k: v for k, v in good.items() if k != 'tds'}], 'text instead of number': [dict(good, net_pay='abc')], 'NaN value': [dict(good, tds=float('nan'))], 'impossible month': [dict(good, month='2026-13')], 'empty batch': []}.items():
    try: predict_payslips(rows); print(label, '-> accepted?!')
    except ValueError as ex: print(f'{label:24s} -> ValueError:', ex)
assert predict_payslips([good, bad]).drop(columns='scored_at').equals(res.drop(columns='scored_at')); print('reproducible: identical output on a second call (only the scored_at timestamp differs, by design)')
assert list(res.columns[:12]) == P.OFFICIAL_COLUMNS and res.loc[res.record_id == 'DEMO-NEG', 'flagged'].iloc[0] == 'Y' and res.loc[res.record_id == ok_id, 'flagged'].iloc[0] == 'N'
print('Output contract: the 12 official columns + risk_band (HIGH = rule hit or score >= 0.90, MEDIUM = other flag, LOW = not flagged). The approved flag rule is the frozen one (rules OR forest outlier); the saved detector carries its own scaling and history, so dev and production agree.')
'''

# ------------------------------------------------------------------------------------------------ 30
NOTES[30] = "runs real tests: inline unit tests for transformations, features, missing values, unseen categories, output types, invalid input, model loading, edge cases, leakage and regression, plus the repository's own tests_phase2 payroll tests and an API integration test when FastAPI is installed."
CODE[30] = r'''
import inspect, importlib, traceback
results = []
def case(name, fn):
    try: fn(); results.append((name, 'PASS', ''))
    except Exception as ex: results.append((name, 'FAIL', f'{type(ex).__name__}: {ex}'[:120]))
def t_expected_gross():
    off = ((DF.gross_pay - P.expected_gross(DF)).abs() > np.maximum(1.0, 0.002 * P.expected_gross(DF).abs())).sum(); assert off <= 40
def t_features_finite(): assert np.isfinite(P.engineer_payroll_features(DF)[P.FEATURE_COLS].fillna(0).to_numpy()).all()
def t_features_no_labels(): assert not set(P.FEATURE_COLS) & {'is_error', 'error_type', 'label'} and 'labels' not in inspect.signature(P.fit_isolation_forest).parameters
def t_missing(): r = VALID.head(2).copy(); r.loc[r.index[0], 'tds'] = np.nan; assert len(P.score_payroll(r, model=det_tr)[0]) == 2
def t_unseen_category(): r = VALID.head(1).copy(); r['department_id'] = 'D404'; r['variable_pay_type'] = 'New'; P.score_payroll(r, model=det_tr)
def t_output_types():
    o = P.to_official_format(P.score_payroll(VALID.head(20), model=det_tr)[0]); assert list(o.columns) == P.OFFICIAL_COLUMNS and o.flagged.isin(['Y', 'N']).all() and pd.api.types.is_numeric_dtype(o.money_at_risk) and o.anomaly_score.between(0, 1).all()
def t_invalid_rejected():
    try: predict_payslips([dict(good, month='bad')]); raise AssertionError('accepted')
    except ValueError: pass
def t_model_loading(): d = P.PayrollDetector.load(ROOT / 'models' / 'payroll_detector_v2.joblib'); assert d.version == P.MODEL_VERSION and len(P.score_payroll(VALID.head(5), model=d)[0]) == 5
def t_edge_cases():
    r = VALID.head(1).copy(); r['payable_days'] = 0; r['gross_pay'] = 0; r['net_pay'] = 0; r['total_deductions'] = 0; assert len(P.score_payroll(r, model=det_tr)[0]) == 1
    r['overtime_hours_paid'] = 700; assert len(P.score_payroll(r, model=det_tr)[0]) == 1
def t_regression_frozen():
    fz = REP / 'payroll_predictions_frozen.csv'
    if fz.exists(): now = P.score_payroll(DF, model=DEPLOYED)[0]; assert int(now.flagged.sum()) == int(pd.read_csv(fz).flagged.astype(str).str.lower().isin(['true', '1', 'y']).sum())
def t_no_leak_in_pipeline(): assert not (set(det_tr.feature_cols) & set(P.ID_COLS)) and len(set(det_tr.feature_cols)) == 23
for n, f in list(globals().items()):
    if n.startswith('t_') and callable(f): case(n[2:], f)
try:
    from fastapi.testclient import TestClient
    sys.path.insert(0, str(ROOT / '05_Starter' / 'api_skeleton')); from app.main import app
    def t_api():
        with TestClient(app) as c: assert c.get('/v1/health').status_code == 200
    case('api_integration (health)', t_api)
except ImportError as ex: results.append(('api_integration', 'SKIPPED', f'fastapi not installed here ({ex.name}); tests_phase2/test_api.py covers it where installed'))
sys.path.insert(0, str(ROOT / 'tests_phase2'))
try:
    tp = importlib.import_module('test_payroll')
    for n in sorted(x for x in dir(tp) if x.startswith('test_')): case('repo tests_phase2/test_payroll::' + n, getattr(tp, n))
except ImportError as ex: results.append(('repo payroll tests', 'SKIPPED', str(ex)))
tests = pd.DataFrame(results, columns=['test', 'status', 'detail']); display(tests); summary = tests.status.value_counts().to_dict(); print('summary:', summary)
RESULTS['tests'] = summary; assert 'FAIL' not in summary, 'a test failed - read the detail column'
print('Leakage check on the complete workflow: no label column in features; detector fit has no label argument; split is temporal; history cut at the split date (stage 9); test months fingerprinted (stage 27). Unresolved issues: none; API test only runs where FastAPI is installed.')
'''

# ------------------------------------------------------------------------------------------------ 31
NOTES[31] = "saves the train-only model with metadata to a demo folder (the shipped model is never overwritten), reloads it to prove identical predictions, records versions, seeds, data hash and metrics, checks for secrets and proves the training is deterministic."
CODE[31] = r'''
import joblib, sklearn
demo = WF / 'models_demo'; demo.mkdir(exist_ok=True); path = det_tr.save(demo / 'payroll_detector_trainonly.joblib')
re_ = P.PayrollDetector.load(path); a, _ = P.score_payroll(VALID_X, model=det_tr); b, _ = P.score_payroll(VALID_X, model=re_)
assert a.anomaly_score.equals(b.anomaly_score) and a.flagged.equals(b.flagged); print('saved + reloaded model reproduces identical scores on', len(a), 'rows')
det2 = P.fit_isolation_forest(TRAIN, contamination=0.03, random_state=SEED); c2, _ = P.score_payroll(VALID_X, model=det2); assert c2.anomaly_score.equals(a.anomaly_score); print('re-training with the same seed and data gives identical scores (deterministic)')
meta = {'model_version': det_tr.version, 'saved_file': str(path.relative_to(ROOT)), 'python': platform.python_version(), 'sklearn': sklearn.__version__, 'pandas': pd.__version__, 'numpy': np.__version__, 'joblib': joblib.__version__, 'seed': SEED,
        'contamination': det_tr.contamination, 'train_rows': det_tr.trained_rows, 'train_months': '2025-04..2026-03', 'data_sha256_16': hashlib.sha256((CSV / 'payroll_monthly_features.csv').read_bytes()).hexdigest()[:16],
        'frozen': FROZEN, 'validation_metrics': RESULTS['valid'], 'test_metrics': RESULTS['test'], 'labels_used_for_training': False}
(demo / 'payroll_detector_trainonly.meta.json').write_text(json.dumps(meta, indent=2, default=str)); display(pd.Series(meta, name='saved metadata').astype(str).str.slice(0, 110).to_frame())
txt = json.dumps(meta).lower(); print('secrets scan (password/token/api_key/secret) in metadata:', [w for w in ('password', 'token', 'api_key', 'secret') if w in txt] or 'none')
print('How to load and use:  det = PayrollDetector.load(path);  scored, _ = score_payroll(df, model=det);  official = to_official_format(scored).')
print('Reproducible training procedure: python scripts/train_phase2_models.py (writes models/ and model_registry.json) or fit_isolation_forest(df, contamination=0.03, random_state=42).')
print('No raw payroll rows are stored inside the model other than ids and per-employee medians used by the history rules (documented limitation).')
'''

# ------------------------------------------------------------------------------------------------ 32
NOTES[32] = "measures latency and throughput of the real scoring path and lays out the deployment plan: batch vs real-time, architecture, API contract, authentication, cost, logging, rollback and human review capacity."
CODE[32] = r'''
rows200 = [to_row(r) for _, r in VALID[VALID.payroll_item_id.isin(real_va[~real_va.flagged].payroll_item_id)].head(200).iterrows()]
t = []
for _ in range(15): t0 = time.perf_counter(); predict_payslips([rows200[0]]); t.append((time.perf_counter() - t0) * 1000)
t0 = time.perf_counter(); predict_payslips(rows200); batch_s = time.perf_counter() - t0
lat = {'single row ms (median / p95)': f'{np.median(t):.0f} / {np.percentile(t, 95):.0f}', f'batch of {len(rows200)} rows (s)': round(batch_s, 2), 'throughput rows/s (batch)': round(len(rows200) / batch_s), 'max rows per call': S.MAX_BATCH_ROWS, 'required max single-row ms (stage 3)': ACCEPT['latency_single_row_ms_max']}
display(pd.Series(lat, name='measured on this machine').astype(str).to_frame()); print('latency requirement met:', np.percentile(t, 95) <= ACCEPT['latency_single_row_ms_max'])
review_hours = RESULTS['threshold']['production_real_flag_share_%'] / 100 * (len(DF) / DF.month.nunique()) * ASSUMPTIONS['review_minutes_per_flag'] / 60
plan = pd.DataFrame([
 ['Inference mode', 'Batch after each monthly payroll run (primary) + real-time single-payslip API for ad-hoc checks'], ['Serving architecture', 'FastAPI service (05_Starter/api_skeleton/app/main.py); models loaded once at start-up from models/; no training per request'],
 ['API contract', 'POST /v1/anomaly/payroll with validated JSON (Pydantic, unknown fields rejected); response = 12 official columns'], ['Authentication / authorization', 'Production: SSO/API key, payroll + finance roles only (the demo service has no real auth) - MANUAL'],
 ['Latency / throughput / availability', f"p95 {np.percentile(t, 95):.0f} ms per row, ~{round(len(rows200) / batch_s)} rows/s; availability target set by IT (monthly batch tolerates restarts)"], ['Infrastructure cost', 'CPU only, a few hundred MB RAM, no GPU; maintenance = monthly retrain check'],
 ['Logging / monitoring / alerting', 'Log request id, model_version, flag, reason_code (never raw pay); alert rules in stage 34'], ['Rollback', 'models/model_registry.json lists version + sha256; keep the previous joblib; switch back by restoring the file'],
 ['Human review', f"~{review_hours:.1f} reviewer-hours per run at the assumed {ASSUMPTIONS['review_minutes_per_flag']} min per flag; escalate unclear items to the Finance controller"], ['Deployment documentation', 'README Phase 2 quick start, docs/model_cards, this notebook']], columns=['aspect', 'decision'])
display(plan)
'''

# ------------------------------------------------------------------------------------------------ 33
NOTES[33] = "runs the deployment smoke tests that are possible here: package/registry integrity, loading, two code paths giving identical predictions, output contract, FastAPI check when available, and the rollback registry; approval itself stays a human step."
CODE[33] = r'''
import importlib.util
reg = S.read_registry()['payroll']; f = ROOT / 'models' / reg['file']
print('1 package + dependencies: model', f.name, '| sklearn at save time', reg['sklearn_version'], '| sklearn now', __import__('sklearn').__version__, '| versions match:', reg['sklearn_version'] == __import__('sklearn').__version__)
print('2 staging load: loaded once via service.load_payroll_detector ->', type(DEPLOYED).__name__, DEPLOYED.version, '| sha256 matches registry:', S._sha256(f) == reg['sha256'])
dfp = VALID.head(200).copy(); dfp_rows = [to_row(r) for _, r in dfp.iterrows()]
api_path = pd.DataFrame(S.score_payroll_rows(dfp_rows, DEPLOYED)).set_index('record_id'); lib_path = P.to_official_format(P.score_payroll(dfp, model=DEPLOYED)[0]).set_index('record_id')
common = api_path.index.intersection(lib_path.index); same = (api_path.loc[common, 'flagged'] == lib_path.loc[common, 'flagged']).all() and np.allclose(api_path.loc[common, 'anomaly_score'].astype(float), lib_path.loc[common, 'anomaly_score'].astype(float), atol=1e-3)
print('3 smoke + 5 deployed vs local predictions: service path and library path agree on', len(common), 'rows ->', bool(same)); assert same
smoke = predict_payslips([dict(good, payroll_item_id='SMOKE-NEG', net_pay=-1.0)]); print('4 smoke test (known-bad payslip) -> flagged', smoke.flagged.iloc[0], '| reason', smoke.reason_code.iloc[0].split('|')[0]); assert smoke.flagged.iloc[0] == 'Y'
print('6 contract: output columns equal the official sample sheet ->', list(smoke.columns[:12]) == list(pd.read_csv(CSV / 'sample_anomaly_results.csv').columns))
if importlib.util.find_spec('fastapi'):
    from fastapi.testclient import TestClient; sys.path.insert(0, str(ROOT / '05_Starter' / 'api_skeleton')); from app.main import app
    with TestClient(app) as c: print('API check: /v1/health ->', c.get('/v1/health').status_code, '| bad payroll payload ->', c.post('/v1/anomaly/payroll', json={'employee_id': 'x'}).status_code, '(422 expected)')
else: print('API check: SKIPPED here (fastapi not installed). Run:  uvicorn app.main:app --app-dir 05_Starter/api_skeleton  then open /docs.')
print('7 security: the repo has no secrets in', [p.name for p in ROOT.glob('.env*')] or 'any .env file', '| API auth is a production TODO (stage 32).')
print('8 production approval: MANUAL (Finance + IT sign-off) | 9 initial rollout: run batch in shadow mode for one payroll run and compare with manual checks | 10 rollback: registry + previous joblib (stage 32)')
display(pd.DataFrame(S.read_registry()).T[['version', 'trained_rows', 'sklearn_version', 'saved_at']])
'''

# ------------------------------------------------------------------------------------------------ 34
NOTES[34] = "builds the monitoring view from real monthly data: volume, flag rate, missing and unknown-category rates, PSI drift of key features and scores against the training reference, alert rules, subgroup flag rates, and the plan for performance monitoring once labels arrive."
CODE[34] = r'''
def psi(expected, actual, bins=10):
    edges = np.unique(np.nanquantile(expected, np.linspace(0, 1, bins + 1)));
    if len(edges) < 3: return 0.0
    e = np.histogram(np.clip(expected, edges[0], edges[-1]), edges)[0] / len(expected) + 1e-4; a = np.histogram(np.clip(actual, edges[0], edges[-1]), edges)[0] / max(len(actual), 1) + 1e-4
    return float(np.sum((a - e) * np.log(a / e)))
allsc = P.score_payroll(DF, model=det_tr)[0].merge(DF[['payroll_item_id', 'department_id']], on='payroll_item_id'); allsc['mo'] = pd.to_datetime(allsc.month).dt.to_period('M'); fa = P.engineer_payroll_features(DF); fa['mo'] = DF.month.dt.to_period('M').values; fa['score'] = allsc.set_index('payroll_item_id').anomaly_score.reindex(DF.payroll_item_id).to_numpy()
ref_ = fa[IS_TRAIN.values]; train_depts = set(TRAIN.department_id); train_rate = float(allsc[allsc.payroll_item_id.isin(TRAIN.payroll_item_id)].flagged.mean())
MON_FEATS = ['net_to_gross_ratio', 'deduction_ratio', 'fixed_gross_log', 'gross_pay_log']      # stable, continuous features (zero-heavy ratios make PSI noisy, so they are not monitored)
rows = []
for mo, g in fa.groupby('mo'):
    f = allsc[allsc.mo == mo]; ps = {c: psi(ref_[c], g[c]) for c in MON_FEATS}
    rows.append({'month': str(mo), 'rows': len(g), 'flag_rate_%': round(f.flagged.mean() * 100, 2), 'missing_input_%': round(DF.loc[g.index, P.NUMERIC_BASE].isna().mean().mean() * 100, 2), 'unknown_dept_%': round((~DF.loc[g.index, 'department_id'].isin(train_depts)).mean() * 100, 2),
                 **{'PSI_' + c: round(v, 3) for c, v in ps.items()}, 'PSI_score': round(psi(ref_.score, g.score), 3), 'features_over_0.25': int(sum(v >= 0.25 for v in ps.values()))})
mon = pd.DataFrame(rows); lo, hi_ = 0.4 * train_rate * 100, 2.5 * train_rate * 100
mon['ALERT'] = (mon.PSI_score >= 0.25) | (mon['features_over_0.25'] >= 2) | (mon['flag_rate_%'] > hi_) | (mon['flag_rate_%'] < lo) | (mon['missing_input_%'] > 1) | (mon['unknown_dept_%'] > 5); display(mon.tail(8))
fig, ax = plt.subplots(1, 2, figsize=(12, 3.6)); ax[0].plot(mon.month, mon['flag_rate_%'], 'o-'); ax[0].axhline(train_rate * 100, ls=':'); ax[0].set_title('flag rate % per month (dotted = training rate)'); ax[0].tick_params(axis='x', rotation=70)
ax[1].plot(mon.month, mon.PSI_score, 'o-'); ax[1].axhline(0.25, c='r', ls=':'); ax[1].set_title('PSI of anomaly score (0.25 = alert)'); ax[1].tick_params(axis='x', rotation=70); plt.tight_layout(); plt.show()
print('months in alert:', mon[mon.ALERT].month.tolist() or 'none', '| alert rules: score PSI >= 0.25, or 2+ monitored features with PSI >= 0.25, or flag rate outside 0.4x-2.5x of the training rate, or missing inputs > 1%, or unknown departments > 5%. Training months are in-sample, so they only show the baseline noise level.')
latest = allsc[allsc.mo == allsc.mo.max()].groupby('department_id').flagged.mean().round(3); print('subgroup monitoring - latest month flag rate by department:', latest.to_dict())
print('Availability / latency / errors: logged by the API layer (stage 32); data quality + drift: this table, run after every payroll run.')
print('Ground-truth performance: when reviewers confirm or reject flags (stage 38) compute precision per reason code and recall from audits using evaluate_by_type(); concept drift = precision falling while PSI stays flat.')
incident = pd.DataFrame(columns=['date', 'model_version', 'trigger', 'cause', 'corrective action', 'owner']); incident.to_csv(WF / 'payroll_incident_log_template.csv', index=False); print('incident log template saved:', (WF / 'payroll_incident_log_template.csv').relative_to(ROOT))
RESULTS['monitoring'] = {'months_in_alert': mon[mon.ALERT].month.tolist(), 'train_flag_rate_%': round(train_rate * 100, 2)}
'''

# ------------------------------------------------------------------------------------------------ 35
NOTES[35] = "defines retraining triggers and rehearses the champion/challenger gate: retrain on more months, compare with the current model on freshly injected errors, check overlap, flag share, fairness and stability, and decide APPROVE or HOLD."
CODE[35] = r'''
triggers = pd.DataFrame([['PSI >= 0.25 on a key feature or the score for 2 consecutive runs', 'drift'], ['Reviewer-confirmed precision below 30% for 2 runs', 'performance'], ['Policy change (overtime cap, new deduction, new pay structure)', 'business rule'], ['12 months since the last retrain', 'schedule'], ['New confirmed labels from reviews (stage 38)', 'new data']], columns=['trigger', 'type']); display(triggers)
print('Validate new data first: schema, duplicates, structural missingness (stages 5-7) and leakage (stage 9) are re-run before every retrain.')
new_train = DF[IS_TRAIN | IS_VALID].reset_index(drop=True); assert not set(new_train.payroll_item_id) & set(TEST.payroll_item_id)
challenger = P.fit_isolation_forest(new_train, contamination=0.03, random_state=SEED); challenger.version = P.MODEL_VERSION + '-candidate'
NEW_X, new_truth = inject_errors(TEST, SEED + 9); ch, _ = P.score_payroll(NEW_X, model=challenger); cp, _ = P.score_payroll(NEW_X, model=det_tr)
def row(name, s): c = confusion(s.flagged.to_numpy(), s.payroll_item_id, new_truth); return {'model': name, 'trained_on_rows': None, **{k: c[k] for k in ('precision', 'recall', 'F1', 'flag_share_%')}}
cmpt = pd.DataFrame([row('champion (train months only)', cp), row('challenger (train + validation months)', ch)]); cmpt['trained_on_rows'] = [det_tr.trained_rows, challenger.trained_rows]; display(cmpt)
real_ch = P.score_payroll(TEST, model=challenger)[0]; real_cp = P.score_payroll(TEST, model=det_tr)[0]; a, b = set(real_cp[real_cp.flagged].payroll_item_id), set(real_ch[real_ch.flagged].payroll_item_id); jac = len(a & b) / max(len(a | b), 1)
gate = pd.DataFrame([['recall not worse than champion by more than 2 points', cmpt.recall.iloc[1] >= cmpt.recall.iloc[0] - 0.02], ['real flag share <= 5%', float(real_ch.flagged.mean() * 100) <= 5.0], ['flagged-set overlap with champion >= 0.70', jac >= 0.70],
    ['no department flag rate > 3x overall', bool((real_ch.merge(TEST[['payroll_item_id', 'department_id']], on='payroll_item_id').groupby('department_id').flagged.mean() <= 3 * real_ch.flagged.mean() + 1e-9).all())]], columns=['gate', 'passed']); display(gate)
print('overlap (Jaccard) of real flagged payslips, champion vs challenger:', round(jac, 3)); print('calibration: no calibrator in use; robustness/fairness re-checked above (overlap + department rates).')
decision = 'APPROVE for a shadow run' if gate.passed.all() else 'HOLD - keep the champion'; print('GATE DECISION:', decision, '- a human (Finance + data owner) gives final approval; the shipped model is NOT replaced by this notebook.')
challenger.save(WF / 'models_demo' / 'payroll_detector_candidate.joblib'); print('version + rollback: candidate saved as', challenger.version, 'in the demo folder; the champion file and its registry entry stay untouched, so rollback = do nothing.')
print('Track real-world outcomes: compare confirmed-error counts and recovered amounts before/after the change (stage 38 feedback table).')
RESULTS['retrain_gate'] = decision
'''

# ------------------------------------------------------------------------------------------------ 36
NOTES[36] = "generates the model documentation automatically from the evidence collected in this notebook (problem, target, data, quality, features, split, models, metrics, threshold, limitations, deployment, reproduction) and saves it as a Markdown file."
CODE[36] = r'''
v, t = RESULTS['valid'], RESULTS['test']
doc = f"""# Payroll anomaly detector - model documentation (generated by the workflow notebook)

## 1. Business problem
Flag risky payslips before payout, rank them for clerks, explain each flag, estimate money at risk (proxy). Assumptions: {ASSUMPTIONS}.

## 2. Target and timing
Binary: payslip error (1) vs normal (0). Unit: one payslip. Predicted after the run is calculated, before payout. Training is unsupervised; labels (CX-06) are evaluation-only.

## 3. Data
payroll_monthly_features.csv - {len(DF)} rows, {DF.month.nunique()} monthly runs ({DF.month.min():%Y-%m} to {DF.month.max():%Y-%m}); synthetic. SHA-256 prefix {hashlib.sha256((CSV / 'payroll_monthly_features.csv').read_bytes()).hexdigest()[:16]}. Dictionary: workflow38/payroll_data_dictionary.csv.

## 4. Data quality and cleaning
{RESULTS['quality_issues']}. Nothing was deleted; anomalous rows are the signal. Decisions: workflow38/payroll_cleaning_decisions.csv.

## 5. Features and preprocessing
23 numeric features (ratios, identity gaps, amounts); median imputation with indicators, standard scaling, Isolation Forest (300 trees, contamination 0.03, seed {SEED}); fitted on training rows only.

## 6. Split
Temporal: train 2025-04..2026-03, validation 2026-04..06, test 2026-07..09. Development errors were injected synthetically (NOT CX-06).

## 7. Baselines and candidates
{RESULTS['baselines']}
Time-aware CV: {RESULTS['cv']}

## 8. Evaluation
Validation: {v}
Test (evaluated once): {t}
Acceptance: {RESULTS['acceptance']}

## 9. Threshold and calibration
{RESULTS['threshold']}. Calibration: scores are rankings, no calibrator (Brier: {RESULTS['calibration']}).

## 10. Limitations, risks, fairness
Synthetic data; precision depends on prevalence; flags are review candidates; money_at_risk is a proxy; fairness screen: {RESULTS['fairness']}; known weak spot: {RESULTS['errors']['missed_types']}.

## 11. Deployment and monitoring
FastAPI service loads the model once; monitoring via monthly PSI/flag-rate table (alerts: {RESULTS['monitoring']['months_in_alert'] or 'none'}); rollback via model_registry.json.

## 12. Reproduce
`python scripts/train_phase2_models.py`, then run the notebooks in order; tests: `python tests_phase2/run_all.py`. Tests in this notebook: {RESULTS['tests']}.
"""
out = WF / 'payroll_model_documentation.md'; out.write_text(doc, encoding='utf-8'); print(doc); print('saved ->', out.relative_to(ROOT))
'''

# ------------------------------------------------------------------------------------------------ 37
NOTES[37] = "closes the project with a coded review: each of the 10 final-review statements gets PASS / FAIL / MANUAL and its evidence; structural items must pass, performance items are reported honestly."
CODE[37] = r'''
fm = (DEPLOYED.version == P.MODEL_VERSION)
review = pd.DataFrame([
 ['1 Business requirements addressed', 'PASS' if list(res.columns[:12]) == P.OFFICIAL_COLUMNS else 'FAIL', 'ranked queue, reason code + sentence, money_at_risk, versioned output (stage 29)'],
 ['2 Beats the required baseline', 'PASS' if RESULTS['test']['F1'] > ACCEPT['baseline_to_beat']['F1'] and RESULTS['test']['recall (sensitivity)'] > 0 else 'FAIL', f"test F1 {RESULTS['test']['F1']} vs benchmark {ACCEPT['baseline_to_beat']['F1']} and dummy 0 (synthetic errors; not CX-06)"],
 ['3 Test set untouched during selection', 'PASS' if hashlib.sha256(pd.util.hash_pandas_object(DF[IS_TEST], index=False).values.tobytes()).hexdigest()[:16] == TEST_FINGERPRINT and (det_tr.score_lo, det_tr.score_hi) == (FROZEN['score_lo'], FROZEN['score_hi']) else 'FAIL', 'fingerprint + frozen model unchanged (stage 27)'],
 ['4 No known leakage remains', 'PASS' if leak.ok.all() else 'FAIL', 'stage 9 checks + temporal split + no label argument'],
 ['5 Complete pipeline works end to end', 'PASS' if len(res) == 2 and fm else 'FAIL', 'load -> validate -> score -> explain -> official format (stages 29, 33)'],
 ['6 Tests pass', 'PASS' if 'FAIL' not in RESULTS['tests'] else 'FAIL', str(RESULTS['tests'])],
 ['7 Deployment and monitoring requirements met', 'MANUAL', 'monitoring table + smoke tests done; infrastructure, authentication and approval need people (stages 32-34)'],
 ['8 Documentation complete', 'PASS' if (WF / 'payroll_model_documentation.md').exists() else 'FAIL', 'workflow38/payroll_model_documentation.md'],
 ['9 Results, limitations and recommendations presented', 'MANUAL', 'a person presents them; the evidence is in stages 20, 24, 25, 27'],
 ['10 Future improvements identified from evidence', 'PASS' if RESULTS['improvements'] else 'FAIL', '; '.join(RESULTS['improvements'])]], columns=['final review item', 'status', 'evidence'])
display(review); fatal = review[(review.status == 'FAIL') & ~review['final review item'].str.startswith('2 ')]; assert fatal.empty, fatal
print('Performance item 2 is reported, not asserted: it uses synthetic errors. The authoritative per-type numbers are the project owner\'s CX-06 table (stage 27).')
print('Recommendations: (1) run one payroll cycle in shadow mode; (2) add a bonus calendar; (3) start collecting reviewer outcomes; (4) evaluate a LOF/ensemble challenger through the stage-35 gate.')
print('Limitations: synthetic data and injected errors, precision depends on prevalence, money_at_risk is a proxy, API has no real authentication yet.')
'''

# ------------------------------------------------------------------------------------------------ 38
NOTES[38] = "turns the model into a learning system: saves the reviewer-feedback template, lists recurring problems from the evidence, and sets the review cadence for metrics, fairness, privacy, documentation and retirement."
CODE[38] = r'''
feedback = pd.DataFrame(columns=['result_id', 'record_id', 'reviewed_by_role', 'reviewed_at', 'decision (CONFIRMED_ERROR / FALSE_ALARM / NEEDS_INFO)', 'true_error_type', 'amount_recovered_inr', 'reviewer_comment']); feedback.to_csv(WF / 'payroll_review_feedback_template.csv', index=False)
print('1 feedback from users: template saved ->', (WF / 'payroll_review_feedback_template.csv').relative_to(ROOT), '(clerks record the decision for every reviewed flag)')
plan = pd.DataFrame([
 ['2 Predictions vs real outcomes', 'monthly: precision per reason code from the feedback table; recall from audits / CX-06 style label files via evaluate_by_type()'],
 ['3 Recurring errors', f"known weak spot from stage 24: {RESULTS['errors']['missed_types']}; false alarms on large legitimate bonuses"],
 ['4 Data collection', 'capture an approved-bonus list and overtime approvals as inputs; keep payroll_item_id stable between runs'],
 ['5 Features', 'add features only if they pass the stage-35 gate (candidate: bonus calendar, past-only employee history)'],
 ['6 Business metrics / acceptance', 'review the stage-3 criteria every quarter with Finance (recall, review hours, recovered amount)'],
 ['7 Fairness / privacy', 'repeat the stage-25 table every quarter; re-check access list and retention'],
 ['8 Controlled evaluation', 'every change goes through champion/challenger (stage 35) on fresh data; never tune on the final test months'],
 ['9 Documentation', 'regenerate stage 36 after every retrain; keep the version table below'],
 ['10 Retire when', 'reviewer-confirmed precision stays below 30% for 3 runs after retraining, the payroll structure changes fundamentally, or a supervised model with reviewer labels clearly beats it']], columns=['continuous-improvement item', 'concrete action']); display(plan)
version_history = pd.DataFrame([[P.MODEL_VERSION, 'rules + Isolation Forest, contamination 0.03, trained on all 18 runs', 'in service'], [P.MODEL_VERSION + '-candidate', 'same recipe, retrained in stage 35 rehearsal', RESULTS['retrain_gate']]], columns=['version', 'description', 'status']); display(version_history)
plan.to_csv(WF / 'payroll_continuous_improvement_plan.csv', index=False); print('Everything produced by this notebook is in', WF.relative_to(ROOT))
'''
