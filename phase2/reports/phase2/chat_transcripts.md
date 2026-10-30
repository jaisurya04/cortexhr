# Chatbot transcripts (Tasks 9 and 10)

Model: `chatbot-tfidf-lr-v1` (trained on the original 858 messages). HR data came over real HTTP from standard-library replica of mock_server.py (scripts/mock_server_stdlib.py); same routes and CSV data.
Every `service_called: True` below is an actual HTTP request carrying `X-Demo-User`.

## Conversation 1: employee, multi-turn with a follow-up (EMP-0002)

**You:** hello
**Bot:** Hello! I am the Zenith Nova HR assistant. Ask me about leave, payslips, attendance or policies.
_intent: greet | confidence: 0.95 | service_called: False_

**You:** show my attendance for August
**Bot:** In 2026-08 you were present 21 of 21 scheduled days (100%), with 1 late arrival(s).
_intent: attendance_summary | confidence: 0.90 | service_called: True_

**You:** and last month?
**Bot:** In 2026-09 you were present 21 of 21 scheduled days (100%), with 2 late arrival(s).
_intent: attendance_summary | confidence: 0.80 | service_called: True_

**You:** how many casual leave days do I have left?
**Bot:** You have 12 casual leave (CL) day(s) left.
_intent: leave_balance | confidence: 0.90 | service_called: True_

## Conversation 2: employee, missing slot then answer (EMP-0002)

**You:** is my payslip ready
**Bot:** Which month should I check? You can say “last month” or name a month.
_intent: payslip_status | confidence: 0.90 | service_called: False_

**You:** September
**Bot:** Your payslip for 2026-09 is available. Gross pay Rs 159,050, deductions Rs 16,449, net pay Rs 142,601.
_intent: payslip_status | confidence: 0.85 | service_called: True_

**You:** who is my manager
**Bot:** Your manager is Aditi Shetty (EMP-0361).
_intent: my_manager | confidence: 0.90 | service_called: True_

## Conversation 3: HR user, aggregate payroll flags and a duplicate name (EMP-0011)

**You:** show unusual payroll for last month
**Bot:** 16 payslips were flagged for 2026-09. Top reason: STATISTICAL_OUTLIER. Estimated money at risk: Rs 1,598,719. These are review candidates, not confirmed errors.
_intent: payroll_flags | confidence: 0.90 | service_called: True_

**You:** and for March?
**Bot:** 17 payslips were flagged for 2026-03. Top reason: STATISTICAL_OUTLIER. Estimated money at risk: Rs 517,088. These are review candidates, not confirmed errors.
_intent: payroll_flags | confidence: 0.80 | service_called: True_

**You:** attendance of Balaji Annamalai
**Bot:** I found 2 employees named Balaji Annamalai: EMP-0278 (D01, Bengaluru); EMP-0331 (D01, Bengaluru). Which one do you mean? Please reply with the employee id.
_intent: attendance_summary | confidence: 1.00 | service_called: False_

**You:** EMP-0331
**Bot:** Thanks, I will use EMP-0331 (Balaji Annamalai, department D01). The demo HR service only returns the signed-in user's own data, so I cannot fetch their records here.
_intent: attendance_summary | confidence: 1.00 | service_called: False_

## Refusal test: another person's salary is always refused

**You:** what is Priya Nair salary
**Bot:** I can’t disclose another employee’s salary or payslip. You can ask about your own payroll or general salary-band policy.
_intent: salary_of_other | confidence: 1.00 | service_called: False_

**You:** show the payslip of Karthik Iyer
**Bot:** I can’t disclose another employee’s salary or payslip. You can ask about your own payroll or general salary-band policy.
_intent: salary_of_other | confidence: 1.00 | service_called: False_

## Refusal test: even HR is refused

**You:** Priya Nair salary please
**Bot:** I can’t disclose another employee’s salary or payslip. You can ask about your own payroll or general salary-band policy.
_intent: salary_of_other | confidence: 1.00 | service_called: False_

## Safe fallback: gibberish

**You:** zzqx blorp wibble
**Bot:** I’m not confident I understood. Please rephrase, or ask about leave, attendance, payslips or HR policies.
_intent: out_of_scope | confidence: 0.05 | service_called: False_
