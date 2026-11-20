# Insights Console - usage guide

| Sidebar item | What it holds | API it calls |
|---|---|---|
| Dashboard | Welcome popup, KPI cards, Workforce Overview (period selector 3 / 6 / 12 months), Attrition Risk Analysis, Key Insights, Model Performance, Recent Activity; search (Ctrl K) by employee id or department; bell = current alerts | `/v1/overview`, `/v1/attrition/employees` |
| Insights | Tabs: **Attrition Risk** (filters, explain one employee), **Payroll Anomalies** (score a payslip), **HR Assistant** (chat), **Pay-Band Advisor** | `/v1/attrition/*`, `/v1/anomaly/payroll`, `/v1/chat`, `/v1/payband/*` |
| Forecast | Six-month forecast with 80% range, events, accuracy table, plain-language explanation, what-if event | `/v1/forecast`, `/v1/forecast/scenario` |
| Fairness & Audit | Status OK / WATCH / ALERT, PSI chart, fairness re-checks, history, "Run monitoring now" | `/v1/monitoring` |
| Models | Availability of every model and the evaluation tables | `/v1/models`, `/v1/performance` |
| Resume Screener | Pick a job, rank resumes (name, gender, age, college tier, e-mail, city are never seen) | `/v1/resume/rank`, `/v1/resume/jobs` |
| API & Deploy | Live health, uptime, round-trip time, endpoint list | `/v1/health` |
| Settings | Connection info (the API key is shown only as set / not set), greeting options | - |

* Every AI output carries the label **"AI-generated, for human review"**.
* Errors (API down, wrong key, model missing, bad input) appear as plain sentences; the console never shows a stack trace unless `CORTEXHR_ENV=development`.
* The welcome popup appears once per browser session, closes with the x button, Esc or a click outside, and can be re-opened with the sun button. Its greeting follows the time of day.
* Day 19 think-for-yourself ("Which tab would a real HR manager use first? Does your console make that obvious?"): my answer is **Insights > Attrition Risk** ("who should I talk to?"), and the dashboard opens with the high-risk count in Key Insights. Whether this is obvious enough is something only your stranger test can show; write down what you observe.

## Update 2026-11-18 - five tabs, label, navigation
* Insights has exactly five tabs: **Risks, Anomalies (Payroll / Attendance), Assistant, Resume Screener (pay-band advisor inside), Forecast**. Start with **Risks** (a hint says so on first visit).
* Every AI result is drawn through `console/ai_render.py`, which prints "AI-generated, for human review" first; `tests/test_ai_label.py` fails if a page bypasses it.
* Sidebar entries are real buttons, so the chosen page and tab survive reruns. The API address comes from `CORTEXHR_API_URL`.
* These Streamlit changes have NOT been run in real Streamlit (not installable in the build sandbox): verify on your machine.
