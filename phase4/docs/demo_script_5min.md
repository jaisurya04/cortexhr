# Five-minute demo script (record it yourself; about 650 words)

> Numbers quoted below come from the saved run in this folder (baseline forecast). Re-read them after you re-run `scripts/run_forecast.py` with statsmodels installed.

**0:00 - 0:30 Problem.** "HR at Growmar has data but no single place to use it. In four phases I built models; in Phase 4 I put them behind one live API, made them watch themselves, and gave HR a console."

**0:30 - 1:30 Dashboard.** Open the console. Let the welcome popup play, close it. "Every number here comes from our data: 441 employees, 12.7% average 12-month risk, 29 openings, the pay-band model's MAE of 53,925 rupees against a role-average baseline of 1,15,094." Change the period to 12 months. Search `EMP-0002`.

**1:30 - 2:30 Forecast.** Forecast page. "Headcount moves from 441 to about 455 in six months; the shaded band is an 80% range. We tested the method by hiding the last six months." Show the accuracy table and say the coverage honestly. Answer the question: "I trust the sixth month least because uncertainty accumulates."

**2:30 - 3:30 Use cases.** Insights > Attrition Risk: filter High, open one person, read the three reasons, point at the "AI-generated, for human review" label. Resume Screener: rank J01; "it never sees name, gender, age, college tier, e-mail or city". Pay-Band Advisor: show a range, not one number.

**3:30 - 4:30 Trust.** Fairness & Audit: status ALERT. "The monitor found that `months_since_last_raise` collapsed: 77% of employees have the same value. That is a data question for HR before anyone trusts 2026 scores. The fairness checks passed their thresholds, but a passed check is not proof of fairness, and the gender pay gap was not checked because the restricted file is not in the repository." Show the 4-step plan.

**4:30 - 5:00 Handover.** API & Deploy page, model cards, release notes. "Which model would I not release? See my written answer." Close with limits you chose to disclose.

Checklist before recording: API and console running, smoke test green, popup not yet dismissed in this browser session (open a new tab), notifications bell showing the drift alert.
