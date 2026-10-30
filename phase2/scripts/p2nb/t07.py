from .common import NB, SETUP, checks_cell

def build():
    n = NB()
    n.md("""# Task 7 (Day 7): Explain every flag with reason codes and money at risk
**Objective.** Give each flag a reason code, a plain sentence and an estimate of the money at risk, in the official output format.
**Why it matters.** Unexplained alerts get ignored. A clear reason and a rupee amount makes people act.
**You will learn.** Designing a useful output, mapping model signals to reasons, estimating money at risk, matching an output format, prioritising alerts.
**Rule.** Same 12 columns as `sample_anomaly_results`, in the same order. Reasons are written first, models second: "the model said so" is not a reason.""")
    n.code(SETUP)

    n.stage(1, "Data Collection", "The official sample sheet and our frozen payroll predictions from Task 6.")
    n.code("""sample = H.load('sample_anomaly_results'); frozen = pd.read_csv(OUT / 'payroll_predictions_frozen.csv', parse_dates=['month'])
print('sample sheet:', sample.shape, '| ours (frozen, all rows scored):', frozen.shape, '| flagged:', int(frozen.flagged.sum()))
sample.head(3)""")

    n.stage(2, "Data Understanding", "Study the sample sheet's columns before writing any code.")
    n.code("""pd.DataFrame({'dtype': sample.dtypes.astype(str), 'example': sample.iloc[0], 'non-null': sample.notna().sum()})""")
    n.code("""sample[sample.source_dataset == 'payroll_monthly_features'].sample(5, random_state=1)[['record_id', 'period', 'anomaly_score', 'flagged', 'reason_code', 'reason_text', 'money_at_risk']]""")

    n.stage(3, "Data Quality Audit", "Is every flag explained, and is every amount a real number?")
    n.code("""fl = frozen[frozen.flagged]
audit = {'flagged rows': len(fl), 'flags without a reason code': int(fl.reason_code.isin(['', 'NOT_FLAGGED']).sum() + fl.reason_code.isna().sum()),
         'flags without reason text': int(fl.reason_text.isna().sum() + (fl.reason_text.str.len() < 20).sum()),
         'money_at_risk missing/inf/negative': int((~np.isfinite(fl.money_at_risk) | (fl.money_at_risk < 0)).sum()),
         'flags with zero money at risk': int((fl.money_at_risk == 0).sum())}
pd.Series(audit, name='count').to_frame()""")

    n.stage(4, "Exploratory Data Analysis (EDA)", "Which reasons dominate, and how big are the amounts?")
    n.code("""primary = fl.reason_code.str.split('|').str[0]
fig, ax = plt.subplots(1, 2, figsize=(14, 3.8))
primary.value_counts().plot.barh(ax=ax[0], title='primary reason code (flagged rows)'); ax[0].invert_yaxis()
ax[1].hist(np.log10(fl.money_at_risk.clip(lower=1)), bins=40); ax[1].set_title('log10 money at risk (Rs)')
fig.tight_layout();""")

    n.stage(5, "Data Cleaning", "Output hygiene: stable types, rounding, no NaN money, ids and periods in the sample's style.")
    n.code("""official = P.to_official_format(frozen.assign(month=pd.to_datetime(frozen.month)))
official['money_at_risk'] = official.money_at_risk.round(2)
print('columns equal to sample:', official.columns.tolist() == sample.columns.tolist())
official.head(3)""")

    n.stage(6, "Feature Engineering", "The mapping from signals to reasons, and the money-at-risk definition.")
    n.code("""pd.DataFrame({'reason_code': list(P.REASON_GLOSSARY), 'plain meaning': list(P.REASON_GLOSSARY.values())})""")
    n.md("""**money_at_risk (a ranking estimate, not an audited loss).** Overpayment = paid minus what fixed pay, payable days, overtime and variable pay explain (`expected_gross`); a duplicate payslip counts the net pay paid twice; missing PF counts the PF owed; overtime above the cap counts the extra overtime pay; negative or zero pay counts the amount that should have been paid. Finance validates the final figure.""")
    n.code("""x = frozen[frozen.flagged].copy(); x['paid_minus_expected_gross'] = (x.gross_pay - x.expected_gross).round(0)
x[x.GROSS_ABOVE_NORM][['payroll_item_id', 'gross_pay', 'expected_gross', 'paid_minus_expected_gross', 'money_at_risk']].head(5)""")

    n.stage(7, "Train-Test Split", "There is no model training in this task, so the held-out piece is the 20-row review sample chosen BEFORE reading any reason text: one pick per reason code first, then the largest remaining amounts.")
    n.code("""rs = np.random.RandomState(C.SEED)
fl2 = frozen[frozen.flagged].assign(primary=lambda d: d.reason_code.str.split('|').str[0])
first = fl2.groupby('primary', group_keys=False).apply(lambda g: g.sample(1, random_state=rs)).head(20)
rest = fl2[~fl2.payroll_item_id.isin(first.payroll_item_id)].sort_values('money_at_risk', ascending=False).head(20 - len(first))
review_ids = pd.concat([first, rest]).payroll_item_id.tolist()
print(len(review_ids), 'review rows chosen;', fl2[fl2.payroll_item_id.isin(review_ids)].primary.nunique(), 'different primary reason codes')""")

    n.stage(8, "Model Building", "Put the results into the official format and rank by money at risk.")
    n.code("""flagged_out = P.to_official_format(frozen.assign(month=pd.to_datetime(frozen.month)), flagged_only=True)
flagged_out[['result_id', 'record_id', 'employee_id', 'period', 'flagged', 'reason_code', 'money_at_risk']].head(8)""")
    n.think("two flags have the same reason. How do you decide which comes first?",
            """same = flagged_out[flagged_out.reason_code.str.startswith('NET_PAY_MISMATCH')].head(6)
same[['result_id', 'reason_code', 'anomaly_score', 'money_at_risk']]""",
            "By the rupee amount first, because a clerk who can only open ten items should open the ten that could cost most. When amounts tie, use the higher `anomaly_score`, then the id so the order never changes between runs. A rule-confirmed flag also beats a model-only flag with the same amount because it is quicker to verify.")

    n.stage(9, "Model Evaluation", "The 4 acceptance checks of the task, as code.")
    n.code("""cols_ok = official.columns.tolist() == sample.columns.tolist()
ranked = flagged_out.money_at_risk.is_monotonic_decreasing
reasons_ok = bool((flagged_out.reason_text.str.len() > 40).all() and (~flagged_out.reason_text.str.contains('model said so', case=False)).all())
numbers_in_text = flagged_out.reason_text.str.contains(r'Rs [\\d,]+').mean()
pd.Series({'columns match the sample (same names, same order)': cols_ok, 'flags ranked by money at risk (largest first)': ranked,
           'every flag has a plain reason sentence': reasons_ok, 'share of reason texts quoting the row\\'s own rupee figures': round(float(numbers_in_text), 3)}).to_frame('result')""")

    n.stage(10, "Hyperparameter Tuning", "No model settings to tune here. The one design choice is how many sentences a reason may contain; check it stays readable.")
    n.code("""words = flagged_out.reason_text.str.split().str.len(); sentences = flagged_out.reason_text.str.count(r'\\. ') + 1
pd.DataFrame({'words': words.describe().round(1), 'sentences': sentences.describe().round(1)})""")
    n.md("Reasons are capped at three sentences (`_compose_reason(max_sentences=3)`), so a clerk reads one short paragraph, not a report.")

    n.stage(11, "Model Interpretation", "Can a clerk understand each of 20 flags without help? Read them like a clerk would.")
    n.code("""review = flagged_out[flagged_out.record_id.isin(review_ids)][['result_id', 'record_id', 'reason_code', 'reason_text', 'money_at_risk']].reset_index(drop=True)
review.to_csv(OUT / 'payroll_review_sample_20_notebook.csv', index=False)
review.head(5)""")
    n.md("**Friend test (a human step, cannot be automated).** Show `reports/phase2/payroll_review_sample_20.csv` to a friend who knows nothing about the project. Ask: *what is wrong with this payslip and what would you do?* Write their feedback in your report; fix any sentence they could not follow.")

    n.stage(12, "Final Conclusions and Recommendations")
    n.code("""print(f'''OUTPUT: {len(flagged_out)} flagged payslips in the 12 official columns, ranked by money at risk (largest first).
TOP ITEM: {flagged_out.iloc[0].record_id} - {flagged_out.iloc[0].reason_code} - Rs {flagged_out.iloc[0].money_at_risk:,.0f}
TOTAL MONEY AT RISK (ranking estimate, not an audited loss): Rs {flagged_out.money_at_risk.sum():,.0f}
RECOMMENDATION: let finance confirm the money formula per reason code, and keep tie-breaking deterministic so reviewers always see the same order.''')""")
    n.checklist(checks_cell("""checks = {
    "'Columns match the sample sheet'": cols_ok,
    "'Every flag has a reason a clerk understands'": reasons_ok,
    "'Flags are ranked by money at risk'": ranked,
    "'20-row test passes (needs a real person)'": 'MANUAL',
}"""))
    n.footer("Your output table (20 rows) = `reports/phase2/payroll_review_sample_20.csv`, plus your friend's feedback.",
             "[pandas](https://github.com/pandas-dev/pandas) | [scikit-learn](https://github.com/scikit-learn/scikit-learn) | [pytest](https://github.com/pytest-dev/pytest)",
             "a reason that says 'the model said so' is not accepted in audits. Teams write reasons first, models second.",
             "you have detectors with explanations. Task 8 packages them as a service other systems can call.",
             "I turned model signals into explanations people can act on.")
    return n.save("P2_T07_reason_codes_money_at_risk.ipynb")
