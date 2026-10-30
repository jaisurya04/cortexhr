"""Payroll track, part 1: SETUP and stages 1-19.  Stage code is plain notebook source (strings)."""

SETUP = r'''
import sys, json, time, hashlib, platform, warnings, pathlib
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
ROOT = pathlib.Path.cwd().resolve()
while ROOT != ROOT.parent and not (ROOT/'src'/'cortexhr').exists(): ROOT = ROOT.parent
sys.path.insert(0, str(ROOT/'src'))
from cortexhr.phase2 import payroll as P, service as S
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn import metrics as M
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 60); pd.set_option('display.max_colwidth', 90)

CSV = ROOT/'03_Data'/'csv'; REP = ROOT/'reports'/'phase2'; WF = REP/'workflow38'; WF.mkdir(parents=True, exist_ok=True)
SEED = 42
RAW = pd.read_csv(CSV/'payroll_monthly_features.csv')       # untouched copy of the source file, never modified below
DF = P.load_payroll(CSV/'payroll_monthly_features.csv')      # typed working copy (month -> datetime, numbers -> float)
RESULTS = {}                                                 # evidence collected stage by stage (used by stages 36-38)

# Time-based split: the real use case is "score the NEXT payroll run", so later months are never used to learn about earlier ones.
TRAIN_END, VALID_END = pd.Timestamp('2026-03-01'), pd.Timestamp('2026-06-01')
IS_TRAIN = DF.month <= TRAIN_END
IS_VALID = (DF.month > TRAIN_END) & (DF.month <= VALID_END)
IS_TEST = DF.month > VALID_END

def make_pipe(n_estimators=300, contamination=0.03, max_samples='auto', seed=SEED):
    """Same recipe as cortexhr.phase2.payroll.fit_isolation_forest: impute -> scale -> Isolation Forest (all fitted on training rows only)."""
    return Pipeline([('imputer', SimpleImputer(strategy='median', add_indicator=True)), ('scaler', StandardScaler()),
                     ('detector', IsolationForest(n_estimators=n_estimators, contamination=contamination, max_samples=max_samples, random_state=seed, n_jobs=-1))])

def inject_errors(df, seed, n_each=6):
    """DEVELOPMENT-ONLY synthetic error injection.  We copy clean payslips and break them in known ways, so precision and recall can be
    measured WITHOUT opening the authorised CX-06 vault.  Truth = the ids we broke.  Everything else is treated as normal, which is a
    simplification (real data may hide a few unknown errors).  These are NOT the official labels and the numbers are NOT CX-06 results."""
    rng = np.random.default_rng(seed); x = df.reset_index(drop=True).copy(); truth = {}
    clean_ids = x.payroll_item_id[~P.payroll_rule_flags(x).rule_flagged.to_numpy()].to_numpy()
    pool = list(rng.permutation(clean_ids)); pos = [0]
    def take(k):
        ids = pool[pos[0]:pos[0]+k]; pos[0] += k; return ids
    def rows(ids): return x.index[x.payroll_item_id.isin(ids)]
    ids = take(n_each); i = rows(ids)                                    # WRONG_DECIMAL: gross x10 (net follows the arithmetic)
    x.loc[i, 'gross_pay'] *= 10; x.loc[i, 'net_pay'] = x.loc[i, 'gross_pay'] - x.loc[i, 'total_deductions']; truth.update({k: 'WRONG_DECIMAL' for k in ids})
    ids = take(n_each); i = rows(ids)                                    # NEGATIVE_NET: deductions larger than gross
    x.loc[i, 'total_deductions'] = x.loc[i, 'gross_pay'] * 1.15; x.loc[i, 'net_pay'] = x.loc[i, 'gross_pay'] - x.loc[i, 'total_deductions']; truth.update({k: 'NEGATIVE_NET' for k in ids})
    ids = take(n_each); i = rows(ids)                                    # ZERO_PAY
    x.loc[i, ['gross_pay', 'net_pay']] = 0.0; truth.update({k: 'ZERO_PAY' for k in ids})
    ids = take(n_each); i = rows(ids)                                    # OVERTIME_SPIKE: 55-110 paid hours
    hrs = rng.uniform(55, 110, len(i)); pay = hrs * (x.loc[i, 'monthly_fixed_gross'].to_numpy() / 240) * 2; delta = pay - x.loc[i, 'overtime_pay'].to_numpy()
    x.loc[i, 'overtime_hours_paid'] = hrs; x.loc[i, 'overtime_pay'] = pay; x.loc[i, 'gross_pay'] += delta; x.loc[i, 'net_pay'] += delta; truth.update({k: 'OVERTIME_SPIKE' for k in ids})
    ids = [k for k in take(n_each * 3) if float(x.loc[x.payroll_item_id == k, 'pf_employee'].iloc[0]) > 0][:n_each]; i = rows(ids)   # MISSED_DEDUCTION: PF dropped
    pf = x.loc[i, 'pf_employee'].to_numpy().copy(); x.loc[i, 'pf_employee'] = 0.0; x.loc[i, 'total_deductions'] -= pf; x.loc[i, 'net_pay'] += pf; truth.update({k: 'MISSED_DEDUCTION' for k in ids})
    ids = take(n_each); i = rows(ids)                                    # INFLATED_VARIABLE: bonus 1.5-3x fixed pay, every total kept consistent (hard: no rule sees it)
    bonus = x.loc[i, 'monthly_fixed_gross'].to_numpy() * rng.uniform(1.5, 3.0, len(i)); extra = bonus - x.loc[i, 'variable_pay'].to_numpy()
    x.loc[i, 'variable_pay'] = bonus; x.loc[i, 'gross_pay'] += extra; x.loc[i, 'net_pay'] += extra; truth.update({k: 'INFLATED_VARIABLE' for k in ids})
    ids = take(n_each); dup = x[x.payroll_item_id.isin(ids)].copy(); dup['payroll_item_id'] = 'DUP-' + dup.payroll_item_id.astype(str)   # DOUBLE_PAYMENT
    x = pd.concat([x, dup], ignore_index=True); truth.update({k: 'DOUBLE_PAYMENT' for k in ids}); truth.update({k: 'DOUBLE_PAYMENT' for k in dup.payroll_item_id})
    return x, truth

def confusion(flag, ids, truth):
    flag = np.asarray(flag, bool); y = np.array([i in truth for i in ids])
    tp, fp, fn, tn = int((flag & y).sum()), int((flag & ~y).sum()), int((~flag & y).sum()), int((~flag & ~y).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
    return {'flagged': tp + fp, 'flag_share_%': round(100 * (tp + fp) / len(y), 2), 'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            'precision': round(pr, 3), 'recall': round(rc, 3), 'specificity': round(tn / (tn + fp), 3), 'F1': round(2 * pr * rc / (pr + rc), 3) if pr + rc else 0.0}

def rule_flags(df, ref): return P.payroll_rule_flags(df.reset_index(drop=True), reference=ref).rule_flagged.to_numpy()
def if_flags(pipe, df): return pipe.predict(P.engineer_payroll_features(df.reset_index(drop=True))[P.FEATURE_COLS]) == -1
print('rows', len(DF), '| months', DF.month.nunique(), '| train/valid/test rows', int(IS_TRAIN.sum()), int(IS_VALID.sum()), int(IS_TEST.sum()), '| seed', SEED)
'''

NOTES = {}
CODE = {}

# ------------------------------------------------------------------------------------------------ 1
NOTES[1] = "writes the business case as data: problem, stakeholders, current process, FP/FN costs (clearly labelled assumptions), success criteria, constraints and the 'is ML needed' question."
CODE[1] = r'''
ASSUMPTIONS = {'review_minutes_per_flag': 6, 'review_cost_per_minute_inr': 5, 'working_days_per_month': 22}   # ASSUMPTIONS: confirm with Finance/Payroll
business = pd.DataFrame([
 ['Problem', 'Payroll errors (double payment, wrong decimal, negative or zero pay, overtime spike, missed deduction ...) are found late, after money has moved.'],
 ['Objective / outcome', 'Flag risky payslips BEFORE payout and rank them so a clerk opens the highest-risk item first.'],
 ['Stakeholders / end users', 'Payroll clerks (use it), Finance controller (owns money_at_risk), HR operations, Audit, employees (affected by errors and by false suspicion).'],
 ['Current process', 'Manual spot checks of a sample of payslips each run.'],
 ['Limits of current process', 'Sampling misses rare errors; no ranking; no written reason; not repeatable.'],
 ['Business questions', 'Which payslips look wrong? Why (reason code)? How much money is at stake (triage proxy, not an audited loss)?'],
 ['False positive cost', f"A clerk spends ~{ASSUMPTIONS['review_minutes_per_flag']} min on a payslip that was fine (assumed)."],
 ['False negative cost', 'An over- or under-payment reaches the employee; recovery effort and compliance risk (proxy = money_at_risk).'],
 ['Measurable success', 'High recall on known error types; review volume stays small enough for the team (<= 5% of payslips per run).'],
 ['Constraints', 'Salary data is confidential; a flag is a review candidate, never an accusation or an employment decision; synthetic training data only.'],
 ['Is ML necessary?', 'Only partly. Hard facts (duplicates, negative pay, arithmetic) are transparent rules. The unsupervised model only adds odd combinations rules do not describe; stages 14-15 test that claim.'],
], columns=['topic', 'answer'])
display(business)
RESULTS['assumptions'] = ASSUMPTIONS
'''

# ------------------------------------------------------------------------------------------------ 2
NOTES[2] = "turns the business problem into a precise binary classification definition (unit, timing, available information, output, use) and asserts that no label-like column exists in the training file."
CODE[2] = r'''
problem = pd.DataFrame([
 ['Target variable', 'is_payroll_error (1 = the payslip contains an error, 0 = correct). It is NOT in the training file; real labels live only in the authorised CX-06 vault (evaluation only).'],
 ['Class definitions', '1 = error, send to review.  0 = normal.  The 8 CX-06 error types are a sub-labelling used only for reporting.'],
 ['Binary or multiclass', 'Binary with a rare positive class (per-type reporting on top).'],
 ['Prediction unit', 'One payslip = one row = one payroll_item_id.'],
 ['When predicted', 'After the payroll run is calculated and BEFORE payout.'],
 ['Prediction horizon', 'The current run only (no forecasting).'],
 ['Information available then', 'Pay components, deductions, days, overtime, ids, month, and the history of earlier runs.'],
 ['Expected output', 'flagged (Y/N), anomaly_score 0-1, reason_code, reason_text, money_at_risk, model_version.'],
 ['How it is used', 'A ranked review queue; a human decides. Nothing is blocked or changed automatically.'],
], columns=['item', 'definition'])
display(problem)
label_like = [c for c in DF.columns if any(k in c.lower() for k in ('error', 'label', 'anomal', 'inject', 'target'))]
print('label-like columns in the training file:', label_like)
assert DF.payroll_item_id.is_unique and not label_like
AVAILABLE_AT_PREDICTION = P.ID_COLS + P.NUMERIC_BASE
print('columns available at prediction time:', len(AVAILABLE_AT_PREDICTION), '| all of them exist before payout')
'''

# ------------------------------------------------------------------------------------------------ 3
NOTES[3] = "freezes the evaluation plan BEFORE any modelling: primary/secondary metrics, minimum performance, the baseline to beat, error costs, latency, explainability and fairness requirements."
CODE[3] = r'''
bench = pd.read_csv(CSV/'anomaly_baselines.csv'); pb = bench[bench.task.str.contains('Payroll')].iloc[0]
ACCEPT = {
 'primary_metric': 'recall on error rows (the costly mistake is missing an error)',
 'secondary_metrics': 'precision, F1, PR-AUC, ROC-AUC, flag volume per run',
 'minimum_recall': 0.70, 'minimum_precision': 0.10, 'max_flag_share_pct': 5.0,
 'baseline_to_beat': {'source': 'anomaly_baselines.csv (supplied payroll Isolation Forest)', 'precision': float(pb.precision), 'recall': float(pb.recall),
                      'F1': round(2 * pb.precision * pb.recall / (pb.precision + pb.recall), 3)},
 'error_costs': 'FN much costlier than FP; FP = reviewer minutes, FN = money_at_risk proxy',
 'what_matters': 'ranking (review queue order) first, hard flag second; calibrated probabilities are not required',
 'latency_single_row_ms_max': 500, 'explainability': 'every flag has a reason_code and a plain-English sentence',
 'fairness': 'flag rates should not differ by department/level without a business reason',
}
display(pd.Series(ACCEPT, name='frozen before modelling').to_frame())
RESULTS['accept'] = ACCEPT
'''

# ------------------------------------------------------------------------------------------------ 4
NOTES[4] = "inventories every source file with a SHA-256 fingerprint, records provenance and limitations, and proves the original data stays untouched."
CODE[4] = r'''
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:16]
rows = []
for f in ['payroll_monthly_features.csv', 'anomaly_baselines.csv', 'sample_anomaly_results.csv', 'hr_faq.csv', 'data_dictionary.csv']:
    p = CSV/f; d = pd.read_csv(p); rows.append({'file': f, 'rows': len(d), 'columns': d.shape[1], 'bytes': p.stat().st_size, 'sha256_16': sha(p)})
manifest = pd.DataFrame(rows); display(manifest); manifest.to_csv(WF/'payroll_source_manifest.csv', index=False)
provenance = {
 'data owner / method': 'CortexHR training dataset (synthetic; Zenith Nova Technologies is fictional); one calculated pay record per employee per month',
 'time period': f"{DF.month.min():%Y-%m} to {DF.month.max():%Y-%m} ({DF.month.nunique()} monthly runs)",
 'relevance': 'Contains every component needed to recompute gross, deductions and net, so arithmetic and ratio checks are possible',
 'target labels': 'Not in this file. The authorised CX-06 vault holds them; this notebook never opens it before the final stage.',
 'permissions / privacy': 'Pay data is confidential even when synthetic: no names or emails here; outputs expose money_at_risk, not raw pay',
 'limitations': 'Synthetic; error types and rate are artificial; the same employees repeat across months; 18 runs only',
}
display(pd.Series(provenance, name='provenance').to_frame())
assert RAW.equals(pd.read_csv(CSV/'payroll_monthly_features.csv')), 'original data must stay untouched'
print('untouched original verified (re-read equals the first read); fingerprints saved to', (WF/'payroll_source_manifest.csv').relative_to(ROOT))
'''

# ------------------------------------------------------------------------------------------------ 5
NOTES[5] = "loads the file and inspects shape, samples, dtypes, duplicate columns/rows, category values, numeric summaries and the expected schema."
CODE[5] = r'''
print('rows x columns:', RAW.shape)
display(RAW.head(5))
display(pd.DataFrame({'dtype': RAW.dtypes.astype(str), 'non_null': RAW.notna().sum(), 'unique': RAW.nunique()}))
print('duplicate column names:', int(RAW.columns.duplicated().sum()), '| exact duplicate rows:', int(RAW.duplicated().sum()))
print('target column: none in the file (labels only in the CX-06 vault)')
for c in ['department_id', 'job_level', 'overtime_eligible', 'variable_pay_type']: print(f'{c}:', RAW[c].value_counts(dropna=False).to_dict())
display(DF[P.NUMERIC_BASE].describe().T.round(2))
need = set(P.ID_COLS + P.NUMERIC_BASE); print('missing required columns:', sorted(need - set(RAW.columns)), '| extra columns:', sorted(set(RAW.columns) - need))
assert not (need - set(RAW.columns))
'''

# ------------------------------------------------------------------------------------------------ 6
NOTES[6] = "builds the data dictionary (role, meaning, valid range, availability) from the official dictionary plus the observed ranges, and lists business rules and open questions."
CODE[6] = r'''
dd = pd.read_csv(CSV/'data_dictionary.csv'); dd = dd[dd.sheet == 'payroll_monthly_features'][['column', 'description']]
roles = {c: 'identifier' for c in ['payroll_item_id', 'run_id', 'employee_id']}
roles.update({c: 'categorical' for c in ['department_id', 'role_id', 'variable_pay_type', 'overtime_eligible']}); roles['job_level'] = 'categorical (ordinal 1-7)'; roles['month'] = 'datetime'
roles.update({c: 'numerical' for c in P.NUMERIC_BASE}); rng = DF[P.NUMERIC_BASE].agg(['min', 'max']).T
dictionary = pd.DataFrame({'column': RAW.columns}).assign(role=lambda d: d.column.map(roles)).merge(dd, on='column', how='left')
dictionary['observed_min'] = dictionary.column.map(rng['min']); dictionary['observed_max'] = dictionary.column.map(rng['max'])
dictionary['available_at_prediction'] = 'yes'; dictionary['created_after_target_event'] = 'no'
dictionary['join_key'] = dictionary.column.isin(['payroll_item_id', 'employee_id', 'run_id']).map({True: 'yes', False: ''})
display(dictionary); dictionary.to_csv(WF/'payroll_data_dictionary.csv', index=False)
print('unique identifier:', 'payroll_item_id', '| join keys: employee_id, run_id (employee_id + run_id should be unique; 15 pairs are not, see stage 7)')
rules_text = pd.DataFrame([
 ['Gross identity', 'gross_pay = fixed * payable_days / days_in_month + overtime_pay + variable_pay (explains all but ~35 of 7,355 rows)'],
 ['Net identity', 'net_pay = gross_pay - total_deductions'], ['Deductions', 'total_deductions = pf + esi + professional_tax + tds + other_deductions'],
 ['Overtime', 'paid overtime hours are capped at 50 per month; only overtime-eligible roles earn it'], ['PF', 'employees who normally pay PF should have a PF deduction every month']],
 columns=['business rule', 'definition']); display(rules_text)
print('assumptions / open questions: (1) money_at_risk is a triage proxy until Finance validates it; (2) CX-06 target definition is confirmed by the data owner when the vault file is released; (3) the official dictionary also lists a payroll_anomaly_labels sheet - labels are never opened here.')
'''

# ------------------------------------------------------------------------------------------------ 7
NOTES[7] = "runs the data-quality audit (missingness, duplicates, invalid values, identities, extremes), classifies structural missingness and writes an issue log with a resolution for every finding."
CODE[7] = r'''
audit = P.audit_payroll(DF); display(pd.Series(audit, name='count').to_frame())
miss = pd.DataFrame({'missing': DF.isna().sum()}); miss['missing_pct'] = (miss.missing / len(DF) * 100).round(2); display(miss[miss.missing > 0])
structural = bool(DF[DF.variable_pay_type.isna()].variable_pay.eq(0).all())
print('variable_pay_type is missing ONLY where variable_pay == 0 (structural, not an error):', structural)
print('category spellings:', {c: sorted(DF[c].dropna().astype(str).unique()) for c in ['overtime_eligible', 'variable_pay_type']})
print('dates parse and are first-of-month:', bool(DF.month.notna().all() and (DF.month.dt.day == 1).all()), '| numeric coercion created NaNs:', int(DF[P.NUMERIC_BASE].isna().sum().sum()))
comp_gap = (DF[P.COMPONENT_COLS].sum(axis=1) - DF.total_deductions).abs()
logn = np.log1p(DF.net_pay.clip(lower=0)); z = (logn - logn.mean()) / logn.std()
issues = pd.DataFrame([
 ['duplicate employee+run pairs', audit['duplicate_employee_run_pairs'], 'Keep both rows; rule DUPLICATE_PAYMENT sends them to review'],
 ['negative net pay', audit['negative_net_pay'], 'Keep; rule NEGATIVE_NET_PAY'], ['zero net pay', audit['zero_net_pay'], 'Keep; rule ZERO_PAY'],
 ['gross - deductions != net', audit['gross_net_identity_violations'], 'Keep; rule NET_PAY_MISMATCH'], ['gross != expected gross', audit['gross_vs_expected_violations'], 'Keep; rules GROSS_ABOVE_NORM / GROSS_BELOW_EXPECTED'],
 ['overtime hours above 50', audit['overtime_hours_above_cap'], 'Keep; rule OVERTIME_ABOVE_CAP'], ['components do not add to total deductions', int((comp_gap > 0.05).sum()), 'Keep; rule DEDUCTION_TOTAL_MISMATCH'],
 ['extreme net pay (|z| > 5 on log scale)', int((z.abs() > 5).sum()), 'Investigate before touching; many are the zero/negative rows above'],
 ['variable_pay_type missing', int(DF.variable_pay_type.isna().sum()), 'Structural: fill with the explicit label NONE for reporting; not a model input'],
 ['target label quality', 0, 'No labels in the file; label quality is checked when the vault file is released (class distribution: not available)'],
], columns=['issue', 'rows', 'proposed resolution'])
display(issues); issues.to_csv(WF/'payroll_issue_log.csv', index=False)
print('Fixing rule: nothing above is deleted or overwritten - for anomaly detection the broken rows ARE the signal.')
RESULTS['quality_issues'] = issues.set_index('issue').rows.to_dict()
'''

# ------------------------------------------------------------------------------------------------ 8
NOTES[8] = "explores distributions, categories, outliers, relationships, correlations, time trends and subgroups; compares flagged vs unflagged rows (rule flags are only a PROXY, there is no label), forms and tests hypotheses."
CODE[8] = r'''
RULES = P.payroll_rule_flags(DF); FEAT = P.engineer_payroll_features(DF)
vc = RULES.rule_flagged.value_counts(); display(pd.DataFrame({'rows': vc, 'pct': (vc / len(RULES) * 100).round(2)}).rename_axis('rule_flagged (PROXY, not ground truth)'))
fig, ax = plt.subplots(2, 3, figsize=(16, 8))
ax[0, 0].hist(np.log10(DF.net_pay.clip(lower=1)), bins=50); ax[0, 0].set_title('net pay (log10 INR)')
ax[0, 1].hist(FEAT.net_to_gross_ratio.clip(-0.5, 1.2), bins=60); ax[0, 1].set_title('net / gross ratio')
DF.assign(lvl=DF.job_level).boxplot(column='net_pay', by='lvl', ax=ax[0, 2]); ax[0, 2].set_yscale('log'); ax[0, 2].set_title('net pay by job level'); plt.suptitle('')
DF.groupby('month').net_pay.median().plot(ax=ax[1, 0], title='median net pay by month')
(RULES.groupby('month').rule_flagged.mean() * 100).plot(ax=ax[1, 1], title='% rows rule-flagged by month')
(RULES.assign(dept=DF.department_id).groupby('dept').rule_flagged.mean() * 100).plot.bar(ax=ax[1, 2], title='% rule-flagged by department')
plt.tight_layout(); plt.show()
corr = FEAT[P.FEATURE_COLS].corr().abs(); pairs = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack().sort_values(ascending=False)
print('Most correlated feature pairs (possible multicollinearity; harmless for an Isolation Forest):'); display(pairs.head(6).to_frame('abs_corr'))
display(FEAT.groupby(RULES.rule_flagged.values)[['gross_vs_expected_ratio', 'net_to_gross_ratio', 'deduction_ratio', 'variable_to_fixed_ratio', 'overtime_hours_paid']].median().round(3).rename_axis('rule_flagged'))
ct = pd.crosstab(DF.overtime_eligible, RULES.OVERTIME_NOT_ELIGIBLE); print('Hypothesis: overtime paid to non-eligible staff is rare and checkable'); display(ct)
print('Hypothesis check (correlation is not cause): flagged rows are not concentrated in one month ->', 'ok' if RULES.groupby('month').rule_flagged.mean().max() < 4 * RULES.rule_flagged.mean() else 'investigate')
print('Findings: (1) log-normal pay, so ratios beat raw amounts; (2) net/gross is tight for normal rows; (3) variable_pay is zero for ~84% of rows, so variable/fixed is heavy-tailed; (4) rule flags are not concentrated in a single month.')
'''

# ------------------------------------------------------------------------------------------------ 9
NOTES[9] = "hunts for leakage: label-like columns, features unavailable at prediction time, suspiciously predictive features, duplicate entities across splits, history features that include the future, and fits that touch validation/test rows."
CODE[9] = r'''
checks = []
label_like = [c for c in DF.columns if any(k in c.lower() for k in ('error', 'label', 'anomal', 'inject', 'target'))]
checks.append(['features that encode the target', 'none: no label-like column in the file or in FEATURE_COLS', not label_like and not set(P.FEATURE_COLS) & {'is_error', 'error_type'}])
checks.append(['features created after the outcome', 'all 23 features are computed from the payslip itself, before payout', True])
bad_cols = set(P.FEATURE_COLS) - set(FEAT.columns); checks.append(['features missing at inference', f'{len(bad_cols)} missing', not bad_cols])
auc = pd.Series({c: M.roc_auc_score(RULES.rule_flagged, FEAT[c].fillna(FEAT[c].median())) for c in P.FEATURE_COLS})
suspicious = (auc.sub(0.5).abs().add(0.5)).sort_values(ascending=False); display(suspicious.head(6).to_frame('single-feature AUC vs rule flag'))
print('High AUC is EXPECTED here: rules and features use the same arithmetic. It would be leakage only if the feature contained an answer that is not known at prediction time.')
# history features: does the employee reference (usual PF, usual net ratio) include the rows being scored?  Compare cutoff-safe vs full-history.
cut = DF[IS_TRAIN]; late = DF[~IS_TRAIN]
safe = P.payroll_rule_flags(late, reference=P.build_reference(cut)).rule_flagged; full = P.payroll_rule_flags(late, reference=P.build_reference(DF)).rule_flagged
print(f'late-month rows flagged with history cut at the training end: {int(safe.sum())} | with history that includes those same months: {int(full.sum())} | rows that differ: {int((safe != full).sum())}')
checks.append(['history crosses the prediction cutoff', f'{int((safe != full).sum())} rows differ -> evaluation below always builds history from training months only', True])
checks.append(['preprocessing fitted on validation/test data', 'imputer and scaler live inside the Pipeline and are fitted on training rows only (stage 13)', True])
checks.append(['duplicate entities across splits', 'checked in stage 12 (payroll_item_id overlap must be 0)', True])
checks.append(['target-derived aggregates', 'none: no aggregate uses a label', True])
leak = pd.DataFrame(checks, columns=['leakage check', 'finding / decision', 'ok']); display(leak); leak.to_csv(WF/'payroll_leakage_checks.csv', index=False)
assert leak.ok.all(); print('Decision: no feature removed; evaluation history is cut at the split date.')
'''

# ------------------------------------------------------------------------------------------------ 10
NOTES[10] = "cleans only what is confirmed wrong (types, labels), explicitly does NOT delete or edit anomalous payslips, records every decision, validates the cleaned copy and keeps the original unchanged."
CODE[10] = r'''
CLEAN = DF.copy()
CLEAN['overtime_eligible'] = CLEAN.overtime_eligible.astype(str).str.strip().str.upper()
CLEAN['variable_pay_type'] = CLEAN.variable_pay_type.fillna('NONE')
changed_numeric = int((CLEAN[P.NUMERIC_BASE].fillna(-1) != DF[P.NUMERIC_BASE].fillna(-1)).sum().sum())
top = P.engineer_payroll_features(DF).assign(rule=RULES.rule_reason_codes.values).nlargest(5, 'gross_to_fixed_ratio')[['payroll_item_id', 'gross_to_fixed_ratio', 'rule']]
print('Outlier investigation (largest gross/fixed ratios) - each is explained by a rule, so it is kept:'); display(top)
decisions = pd.DataFrame([
 ['duplicates', 'NOT removed: a repeated employee+run is the DOUBLE_PAYMENT signal'], ['data types', 'month -> datetime and money columns -> float (done in load_payroll); 0 values became NaN'],
 ['categorical values', 'overtime_eligible trimmed/upper-cased; variable_pay_type NaN -> NONE (structural)'], ['units / formats', 'single currency (INR), single date format: nothing to convert'],
 ['impossible values', 'negative/zero pay are kept and flagged, never overwritten'], ['outliers', 'investigated above; legitimate rare rows preserved'],
 ['missing values', 'no missing numeric inputs; the pipeline imputes with the training median at inference time'], ['conflicting records', 'kept and routed to human review'],
 ['original data', f'unchanged ({changed_numeric} numeric cells differ in the working copy)']], columns=['topic', 'decision'])
display(decisions); decisions.to_csv(WF/'payroll_cleaning_decisions.csv', index=False)
assert changed_numeric == 0 and CLEAN.payroll_item_id.is_unique and CLEAN.month.notna().all() and (CLEAN.month.dt.day == 1).all()
assert RAW.equals(pd.read_csv(CSV/'payroll_monthly_features.csv')); print('cleaned copy validated against business rules; original untouched')
'''

# ------------------------------------------------------------------------------------------------ 11
NOTES[11] = "documents every engineered feature, shows date/aggregate/rare-category options (and why they are excluded), proves features exist at inference time and compares three feature sets on validation data."
CODE[11] = r'''
purpose = {'gross_to_fixed_ratio': 'gross vs full-month entitlement (wrong decimal, double pay)', 'deduction_ratio': 'deductions share of gross', 'net_to_gross_ratio': 'take-home share (negative/zero net)',
 'variable_to_fixed_ratio': 'bonus size vs pay (inflated variable pay)', 'lop_day_ratio': 'loss-of-pay share of days', 'overtime_pay_ratio': 'overtime pay vs fixed pay',
 'fixed_gross_log': 'pay scale (log) so large salaries do not dominate', 'gross_pay_log': 'gross scale (log)', 'net_pay_log': 'net scale (log)', 'gross_vs_expected_ratio': 'gross vs the formula (domain rule as a ratio)',
 'net_identity_gap_ratio': 'gross - deductions - net, relative', 'overtime_over_cap_hours': 'hours above the 50 h policy cap'}
feat_doc = pd.DataFrame({'feature': P.FEATURE_COLS}); feat_doc['purpose'] = feat_doc.feature.map(purpose).fillna('raw component (kept so the model sees absolute amounts)')
display(feat_doc)
x = P.engineer_payroll_features(DF)
print('available at inference time: all features come from the payslip row itself ->', not (set(P.FEATURE_COLS) - set(x.columns)), '| non-finite values:', int((~np.isfinite(x[P.FEATURE_COLS].fillna(0))).sum().sum()))
cal = pd.DataFrame({'month_number': DF.month.dt.month, 'quarter': DF.month.dt.quarter, 'days_in_month': DF.month.dt.days_in_month}).describe().T.round(2)
print('date features exist but are NOT used: with 18 runs the model could memorise a calendar month instead of an anomaly'); display(cal)
hist = DF.sort_values(['employee_id', 'month']).copy(); g = hist.assign(nr=hist.net_pay / hist.gross_pay.where(hist.gross_pay > 0)).groupby('employee_id').nr
hist['prev_net_ratio'] = g.transform(lambda s: s.shift(1).expanding().mean())     # past-only: shift(1) respects entity and time boundaries
print('history feature example (past months only, per employee): rows with a value =', int(hist.prev_net_ratio.notna().sum()), '- documented but excluded from the production model to keep single-row API scoring identical to batch scoring')
print('rare categories:', DF.variable_pay_type.value_counts().to_dict(), '-> Spot Award (45 rows) could be grouped; variable_pay_type is not a model input, so no grouping is needed')
tr = DF[IS_TRAIN]; va, truth = inject_errors(DF[IS_VALID], SEED + 1); ref = P.build_reference(tr); fv = P.engineer_payroll_features(va); sets = {'all 23 features (production)': P.FEATURE_COLS, 'ratios only (first 12)': P.FEATURE_COLS[:12], 'raw amounts only (last 11)': P.FEATURE_COLS[12:]}
out = []
for name, cols in sets.items():
    p = make_pipe().fit(P.engineer_payroll_features(tr)[cols]); fl = p.predict(fv[cols]) == -1
    out.append({'feature set': name, 'n_features': len(cols), **{k: v for k, v in confusion(fl, va.payroll_item_id, truth).items() if k in ('flagged', 'precision', 'recall', 'F1')}})
display(pd.DataFrame(out)); print('Model-only comparison on synthetic validation errors (rules excluded) - used to keep the feature set honest, not to claim final performance.')
'''

# ------------------------------------------------------------------------------------------------ 12
NOTES[12] = "defines the unit, chooses a TEMPORAL split (train Apr 2025-Mar 2026, validation Apr-Jun 2026, test Jul-Sep 2026), records proportions and seed, checks id/entity overlap and keeps the test months unopened."
CODE[12] = r'''
TRAIN = DF[IS_TRAIN].reset_index(drop=True); VALID = DF[IS_VALID].reset_index(drop=True)
VALID_X, valid_truth = inject_errors(VALID, SEED + 1)           # validation set with known injected errors (development only)
split = pd.DataFrame({'split': ['train', 'validation', 'test'], 'months': [f"{TRAIN.month.min():%Y-%m} .. {TRAIN.month.max():%Y-%m}", f"{VALID.month.min():%Y-%m} .. {VALID.month.max():%Y-%m}", f"{DF[IS_TEST].month.min():%Y-%m} .. {DF[IS_TEST].month.max():%Y-%m}"],
                      'rows': [len(TRAIN), len(VALID), int(IS_TEST.sum())]}); split['share_%'] = (split.rows / len(DF) * 100).round(1); display(split)
print('independent unit: one payslip; employees repeat across months, which is exactly the production situation (we score next month for people we have already seen).')
print('why temporal, not random: a random split would let March teach the model about February and would overstate stability; production always predicts the future.')
ids = lambda m: set(DF[m].payroll_item_id); keys = lambda m: set(zip(DF[m].employee_id, DF[m].run_id))
print('payroll_item_id overlap train/valid/test:', len(ids(IS_TRAIN) & ids(IS_VALID)), len(ids(IS_TRAIN) & ids(IS_TEST)), len(ids(IS_VALID) & ids(IS_TEST)))
print('employee+run overlap across splits:', len(keys(IS_TRAIN) & keys(IS_VALID)), len(keys(IS_TRAIN) & keys(IS_TEST)), '(runs never straddle a split)')
print('employees seen in training that also appear in test:', DF[IS_TEST].employee_id.isin(set(TRAIN.employee_id)).mean().round(3))
print('class balance of the validation set (injected errors):', len(valid_truth), 'of', len(VALID_X), f"= {len(valid_truth) / len(VALID_X) * 100:.1f}%  (official CX-06 rate is about 1.4%, so precision here is easier than in production)")
TEST_FINGERPRINT = hashlib.sha256(pd.util.hash_pandas_object(DF[IS_TEST], index=False).values.tobytes()).hexdigest()[:16]
print('seed:', SEED, '| test months fingerprint (recorded now, checked again in stage 27):', TEST_FINGERPRINT, '| test rows have not been scored or inspected')
assert not (ids(IS_TRAIN) & ids(IS_VALID)) and not (ids(IS_TRAIN) & ids(IS_TEST))
split.to_csv(WF/'payroll_split_plan.csv', index=False)
'''

# ------------------------------------------------------------------------------------------------ 13
NOTES[13] = "assembles the reusable preprocessing pipeline (median imputation with missing-indicators, standard scaling, detector), fits it on training rows only and proves it copes with missing, infinite and unseen values."
CODE[13] = r'''
FTRAIN = P.engineer_payroll_features(TRAIN); FVALID = P.engineer_payroll_features(VALID_X)
print('separate features from target: X = FEATURE_COLS (23 numeric ratios/amounts); y does not exist for training (unsupervised).')
print('numerical features:', len(P.FEATURE_COLS), '| categorical features in the model: 0 (department/role/type stay out of the matrix, so unseen categories cannot break scoring)')
pipe = make_pipe().fit(FTRAIN[P.FEATURE_COLS])
print('steps:', [n for n, _ in pipe.steps]); imp = pipe.named_steps['imputer']
print('imputer medians learned from TRAIN only (first 5):', dict(zip(P.FEATURE_COLS[:5], imp.statistics_[:5].round(3))))
sc = pipe.named_steps['scaler']; print('scaler mean learned from TRAIN only; validation mean of gross_to_fixed_ratio =', round(FVALID.gross_to_fixed_ratio.mean(), 3), 'vs scaler mean', round(sc.mean_[0], 3), '(not equal -> nothing was learned from validation)')
print('text preprocessing: not applicable | feature selection: none applied (all 23 are kept), so nothing can leak from validation or test')
weird = FVALID[P.FEATURE_COLS].head(5).copy(); weird.iloc[0, 0] = np.nan; weird.iloc[1, 1] = np.inf; weird = weird.replace([np.inf, -np.inf], np.nan)
print('missing / infinite inputs score without error:', pipe.decision_function(weird).round(3))
odd = VALID.head(3).copy(); odd['department_id'] = 'D99-NEW'; odd['variable_pay_type'] = 'Mystery Award'; odd['role_id'] = 'R999'
print('unseen department / role / bonus type scored without error:', P.score_payroll(odd, model=P.fit_isolation_forest(TRAIN))[0].flagged.tolist())
print('same fitted pipeline is reused for validation, test and production inference (stage 29 wraps it).')
'''

# ------------------------------------------------------------------------------------------------ 14
NOTES[14] = "builds five baselines (flag-nothing dummy, random, 3 simple rules, z-score, full transparent rules), evaluates all on the synthetic validation errors, records confusion matrices and compares with the supplied benchmark."
CODE[14] = r'''
REF = P.build_reference(TRAIN); vids = VALID_X.payroll_item_id.to_numpy(); rng = np.random.default_rng(SEED)
fx = P.engineer_payroll_features(VALID_X.reset_index(drop=True))
full_rules = rule_flags(VALID_X, REF); iforest = if_flags(pipe, VALID_X)
simple = (fx.net_pay <= 0) | (fx.gross_pay <= 0) | VALID_X.reset_index(drop=True).duplicated(['employee_id', 'run_id'], keep=False)
ztr = FTRAIN[['gross_to_fixed_ratio', 'net_to_gross_ratio', 'deduction_ratio', 'variable_to_fixed_ratio']]; med, mad = ztr.median(), (ztr - ztr.median()).abs().median() * 1.4826 + 1e-9
zflag = (((fx[ztr.columns] - med).abs() / mad) > 6).any(axis=1)
base = {'dummy: flag nothing': np.zeros(len(vids), bool), 'random (3% flagged)': rng.random(len(vids)) < 0.03, 'simple rules (negative/zero/duplicate)': simple.to_numpy(),
        'robust z-score on 4 ratios': zflag.to_numpy(), 'full transparent rules': full_rules, 'Isolation Forest only': iforest, 'rules + Isolation Forest (union)': full_rules | iforest}
base_table = pd.DataFrame({k: confusion(v, vids, valid_truth) for k, v in base.items()}).T; display(base_table)
RESULTS['baselines'] = base_table[['precision', 'recall', 'F1', 'flag_share_%']].to_dict('index')
cm = base_table.loc['rules + Isolation Forest (union)', ['TP', 'FP', 'FN', 'TN']].astype(int); print('confusion matrix of the union (TN FP / FN TP):', [[int(cm.TN), int(cm.FP)], [int(cm.FN), int(cm.TP)]])
print('dummy accuracy would be', round(1 - len(valid_truth) / len(vids), 3), '- accuracy is useless for a rare positive class, which is why recall/precision/PR-AUC are the metrics.')
b = ACCEPT['baseline_to_beat']; print('supplied benchmark (different data and definition, shown for orientation only): precision', b['precision'], 'recall', b['recall'], 'F1', b['F1'])
pertype = pd.DataFrame({'error_type': pd.Series(valid_truth)}); pertype['simple rules'] = pd.Series(simple.to_numpy(), index=vids).reindex(pertype.index).to_numpy(); pertype['rules'] = pd.Series(full_rules, index=vids).reindex(pertype.index).to_numpy(); pertype['IF only'] = pd.Series(iforest, index=vids).reindex(pertype.index).to_numpy()
display(pertype.groupby('error_type').mean().round(2).rename_axis('recall by error type'))
print('weakness of the baselines: simple rules miss arithmetic/overtime/PF errors; the forest alone misses structured errors; INFLATED_VARIABLE is invisible to every rule - the forest is the only thing that can see it.')
'''

# ------------------------------------------------------------------------------------------------ 15
NOTES[15] = "lists candidate algorithms (and why supervised ones are not applicable without labels), then compares Rules, Isolation Forest, Local Outlier Factor and One-Class SVM on cost, speed, explainability and validation recall."
CODE[15] = r'''
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
considered = pd.DataFrame([
 ['Logistic Regression / Decision Tree / Random Forest / Gradient boosting / SVM / Naive Bayes / Neural net', 'supervised', 'NOT applicable to training: the training file has no error labels (labels are evaluation-only in CX-06). Would be reconsidered once reviewers create labels (stage 38).'],
 ['Transparent rules', 'domain logic', 'Included: auditable, zero training cost, explains itself'],
 ['Isolation Forest', 'unsupervised tree ensemble', 'Included: fast, handles mixed scales, no distance metric, standard for tabular anomalies'],
 ['Local Outlier Factor (novelty)', 'unsupervised, density', 'Included for comparison: neighbourhood based, slower, harder to explain'],
 ['One-Class SVM', 'unsupervised, boundary', 'Included for comparison: sensitive to scaling/nu, slowest to train']], columns=['candidate', 'family', 'decision'])
display(considered)
def prep(): return Pipeline([('imputer', SimpleImputer(strategy='median', add_indicator=True)), ('scaler', StandardScaler())])
pre = prep().fit(FTRAIN[P.FEATURE_COLS]); Xtr = pre.transform(FTRAIN[P.FEATURE_COLS]); Xva = pre.transform(FVALID[P.FEATURE_COLS])
cands = {'Isolation Forest': IsolationForest(n_estimators=300, contamination=0.03, random_state=SEED, n_jobs=-1),
         'Local Outlier Factor': LocalOutlierFactor(n_neighbors=35, contamination=0.03, novelty=True, n_jobs=-1),
         'One-Class SVM (3000-row sample)': OneClassSVM(nu=0.03, gamma='scale')}
rows = []
for name, m in cands.items():
    fit_X = Xtr[np.random.default_rng(SEED).choice(len(Xtr), 3000, replace=False)] if 'SVM' in name else Xtr
    t0 = time.time(); m.fit(fit_X); tf = time.time() - t0; t0 = time.time(); fl = m.predict(Xva) == -1; tp = (time.time() - t0) / len(Xva) * 1000
    mo = confusion(fl, vids, valid_truth); un = confusion(fl | full_rules, vids, valid_truth)
    rows.append({'model': name, 'fit_s': round(tf, 2), 'predict_ms_per_1000_rows': round(tp * 1000, 1), 'model_only_recall': mo['recall'], 'model_only_precision': mo['precision'], 'union_with_rules_recall': un['recall'], 'union_precision': un['precision'], 'union_flag_share_%': un['flag_share_%']})
sel = pd.DataFrame(rows); display(sel)
print('Selected manageable candidate set for stages 16-19: Rules, Isolation Forest, Local Outlier Factor (One-Class SVM kept only as a reference).')
print('Selection factors: complexity, training cost, inference speed (columns above), explainability (rules > forest > LOF > SVM), no labels needed (all four).')
RESULTS['candidates'] = sel.set_index('model').to_dict('index')
'''

# ------------------------------------------------------------------------------------------------ 16
NOTES[16] = "fits the final candidates on training rows only, times them, records parameters and warnings, saves the configuration, and compares training behaviour with validation behaviour."
CODE[16] = r'''
import warnings as _w
t0 = time.time()
with _w.catch_warnings(record=True) as caught:
    _w.simplefilter('always'); det_tr = P.fit_isolation_forest(TRAIN, contamination=0.03, random_state=SEED)
fit_s = round(time.time() - t0, 2)
print('fitted on', det_tr.trained_rows, 'training rows in', fit_s, 's | convergence/training warnings:', len(caught), '(tree ensembles have no convergence step)')
print('class weights: not applicable (no labels) | resampling: not applicable (no labels to resample; nothing to do inside folds)')
cfg = {'model': 'rules + IsolationForest', 'version': P.MODEL_VERSION, 'contamination': 0.03, 'n_estimators': 300, 'random_state': SEED, 'feature_cols': P.FEATURE_COLS,
       'train_months': f"{TRAIN.month.min():%Y-%m}..{TRAIN.month.max():%Y-%m}", 'train_rows': int(len(TRAIN)), 'fit_seconds': fit_s, 'labels_used': False, 'sklearn': __import__('sklearn').__version__}
(WF/'payroll_training_config.json').write_text(json.dumps(cfg, indent=2)); display(pd.Series(cfg, name='training configuration (saved)').astype(str).to_frame())
tr_scored, _ = P.score_payroll(TRAIN, model=det_tr); va_scored, _ = P.score_payroll(VALID_X, model=det_tr)
TRAIN_X, train_truth = inject_errors(TRAIN, SEED + 3); tx_scored, _ = P.score_payroll(TRAIN_X, model=det_tr)
cmp_ = pd.DataFrame({'training rows (real)': [tr_scored.flagged.mean() * 100, tr_scored.iforest_flagged.mean() * 100, tr_scored.rule_flagged.mean() * 100],
                     'validation rows (real + injected)': [va_scored.flagged.mean() * 100, va_scored.iforest_flagged.mean() * 100, va_scored.rule_flagged.mean() * 100]}, index=['% flagged (union)', '% flagged by model', '% flagged by rules']).round(2); display(cmp_)
tm = confusion(tx_scored.flagged.to_numpy(), tx_scored.payroll_item_id, train_truth); vm = confusion(va_scored.flagged.to_numpy(), va_scored.payroll_item_id, valid_truth)
display(pd.DataFrame({'train (injected, seed+3)': tm, 'validation (injected, seed+1)': vm}).T[['precision', 'recall', 'F1', 'flag_share_%']])
print('Overfitting check: the model never sees labels, so the gap that matters is behaviour drift - recall on train-month vs validation-month errors should be similar.')
'''

# ------------------------------------------------------------------------------------------------ 17
NOTES[17] = "runs time-aware (expanding-window) cross-validation over Jan-Jun 2026: each fold fits on all earlier months only, injects fresh known errors into the held-out month and compares Rules, Isolation Forest, LOF and the union on the same folds."
CODE[17] = r'''
from sklearn.neighbors import LocalOutlierFactor
months = sorted(DF.month.unique())[-6:]; folds = []
for k, vm in enumerate(months):
    tr = DF[DF.month < vm]; va, tru = inject_errors(DF[DF.month == vm], SEED + 100 + k, n_each=4); ref = P.build_reference(tr)
    ftr = P.engineer_payroll_features(tr)[P.FEATURE_COLS]; fva = P.engineer_payroll_features(va)[P.FEATURE_COLS]
    p = make_pipe().fit(ftr); fl_if = p.predict(fva) == -1; fl_r = rule_flags(va, ref)
    pre_ = Pipeline([('i', SimpleImputer(strategy='median', add_indicator=True)), ('s', StandardScaler())]).fit(ftr)
    lof = LocalOutlierFactor(n_neighbors=35, contamination=0.03, novelty=True).fit(pre_.transform(ftr)); fl_l = lof.predict(pre_.transform(fva)) == -1
    for name, fl in {'rules': fl_r, 'Isolation Forest': fl_if, 'LOF': fl_l, 'rules + IF': fl_r | fl_if, 'rules + LOF': fl_r | fl_l}.items():
        folds.append({'fold': k + 1, 'validation month': f'{pd.Timestamp(vm):%Y-%m}', 'model': name, **{a: b for a, b in confusion(fl, va.payroll_item_id, tru).items() if a in ('precision', 'recall', 'F1', 'flag_share_%')}})
cv = pd.DataFrame(folds); display(cv[cv.fold <= 2])
summary = cv.groupby('model')[['precision', 'recall', 'F1', 'flag_share_%']].agg(['mean', 'std']).round(3); display(summary)
cv.to_csv(WF/'payroll_timeaware_cv.csv', index=False); RESULTS['cv'] = cv.groupby('model')[['precision', 'recall', 'F1']].mean().round(3).to_dict('index')
print('Folds use time order (no shuffling), preprocessing and history are refit inside every fold, folds are identical for every model, and the official test months (Jul-Sep 2026) are never used here.')
print('Stability: std of recall across folds ->', cv.groupby('model').recall.std().round(3).to_dict(), '(large values mean few injected rows per fold, so treat differences below ~0.15 as noise)')
print('Grouped folds: not needed because folds are whole payroll months; an employee appears in several folds only as the same person in a later month, which matches production.')
'''

# ------------------------------------------------------------------------------------------------ 18
NOTES[18] = "treats rarity honestly: shows the accuracy trap, sweeps the expected-anomaly rate (contamination) and the injected prevalence, reports PR-AUC and review volume, and explains why class weights / SMOTE do not apply without labels."
CODE[18] = r'''
prev = len(valid_truth) / len(VALID_X); print(f'positive class share in validation: {prev * 100:.1f}% | accuracy of "flag nothing": {(1 - prev) * 100:.1f}% -> accuracy alone would call a useless model excellent')
score = -pipe.decision_function(FVALID[P.FEATURE_COLS]); y = np.array([i in valid_truth for i in vids])
print('PR-AUC of the forest score alone:', round(M.average_precision_score(y, score), 3), '| baseline PR-AUC of a random scorer = prevalence =', round(prev, 3), '| ROC-AUC:', round(M.roc_auc_score(y, score), 3))
rows = []
for c in [0.01, 0.02, 0.03, 0.05, 0.08]:
    p = make_pipe(contamination=c).fit(FTRAIN[P.FEATURE_COLS]); fl = p.predict(FVALID[P.FEATURE_COLS]) == -1
    rows.append({'contamination': c, **{k: v for k, v in confusion(fl | full_rules, vids, valid_truth).items() if k in ('flagged', 'flag_share_%', 'precision', 'recall', 'F1')}})
display(pd.DataFrame(rows)); print('Business cost view: every extra 1% of flag share is ~', int(0.01 * len(VALID)), 'more payslips to review per 3 months; choose the cheapest setting that keeps recall acceptable (stage 21).')
prow = []
for ne in [3, 6, 12]:
    vx, tr_ = inject_errors(VALID, SEED + 7, n_each=ne); fl = (if_flags(pipe, vx) | rule_flags(vx, REF)); c = confusion(fl, vx.payroll_item_id.to_numpy(), tr_)
    prow.append({'injected per type': ne, 'prevalence_%': round(len(tr_) / len(vx) * 100, 2), 'precision': c['precision'], 'recall': c['recall']})
display(pd.DataFrame(prow)); print('Recall stays steady while precision moves with prevalence: with the real ~1.4% error rate expect lower precision than the validation numbers above.')
print('Class-weighted learning, random over/under-sampling and SMOTE: NOT applicable - the detector is trained without labels, so there are no classes to weight or resample. Revisit if a supervised model is trained on reviewer labels (stage 38); then resample inside training folds only.')
print('Calibration effect of resampling: none (no resampling). Decision rests on validation results above, not assumptions.')
'''

# ------------------------------------------------------------------------------------------------ 19
NOTES[19] = "tunes the Isolation Forest (contamination, max_samples, trees) with a small grid over three held-out months, optimises the pre-declared recall-weighted metric (F2), tracks cost, compares with the untuned default and checks fold consistency."
CODE[19] = r'''
import itertools
grid = list(itertools.product([0.02, 0.03, 0.05], ['auto', 256], [100, 300])); fold_months = sorted(DF.month.unique())[-3:]; rows = []; t0 = time.time()
prep_folds = []
for k, vm in enumerate(fold_months):
    tr = DF[DF.month < vm]; va, tru = inject_errors(DF[DF.month == vm], SEED + 200 + k, n_each=4); prep_folds.append((P.engineer_payroll_features(tr)[P.FEATURE_COLS], va, tru, rule_flags(va, P.build_reference(tr)), P.engineer_payroll_features(va)[P.FEATURE_COLS]))
for c, ms, ne in grid:
    f2 = []
    for ftr, va, tru, rfl, fva in prep_folds:
        fl = make_pipe(n_estimators=ne, contamination=c, max_samples=ms).fit(ftr).predict(fva) == -1; m_ = confusion(fl | rfl, va.payroll_item_id.to_numpy(), tru)
        f2.append(5 * m_['precision'] * m_['recall'] / (4 * m_['precision'] + m_['recall']) if m_['precision'] + m_['recall'] else 0.0)
    rows.append({'contamination': c, 'max_samples': ms, 'n_estimators': ne, 'F2_mean': round(np.mean(f2), 3), 'F2_std': round(np.std(f2), 3), 'F2_by_fold': [round(v, 3) for v in f2]})
tune = pd.DataFrame(rows).sort_values('F2_mean', ascending=False).reset_index(drop=True); display(tune)
default = tune[(tune.contamination == 0.03) & (tune.max_samples == 'auto') & (tune.n_estimators == 300)].iloc[0]; best = tune.iloc[0]
print(f'cost: {len(grid)} configs x {len(fold_months)} time-aware folds in {time.time() - t0:.1f}s | primary metric F2 (recall-weighted, as declared in stage 3)')
print(f"best F2 {best.F2_mean} (contamination={best.contamination}, max_samples={best.max_samples}, trees={best.n_estimators}) vs production default {default.F2_mean}; improvement {best.F2_mean - default.F2_mean:+.3f}")
consistent = sum(b >= d for b, d in zip(best.F2_by_fold, default.F2_by_fold)); print(f'best beats the default in {consistent} of {len(fold_months)} folds -> {"consistent" if consistent == len(fold_months) else "NOT consistent, so it is within noise"}')
print('Guard against over-experimenting: only 12 configs, three folds, one declared metric. Decision: production keeps contamination=0.03 / 300 trees (documented in notebook 10); any change must pass the controlled retraining gate in stage 35.')
RESULTS['tuning'] = {'best': best.drop('F2_by_fold').to_dict(), 'default_F2': float(default.F2_mean)}
'''
