# Phase 2 status review (reviewer's notes on cortexhr_phase2_v6)

Everything below was **run or observed** in a review session; nothing here is copied from the earlier README claims.

## What I ran
| Check | Result |
|---|---|
| Phase 1 tests `python tests/run_all.py` | 32 passed |
| Phase 2 tests `python tests_phase2/run_all.py` | 56 passed, 1 module skipped (`test_api.py` needs FastAPI + httpx, not installed in the review sandbox: **run `pytest -q tests_phase2` on your laptop**) |
| `app/main.py` | compiles; routes: GET /v1/health, GET /v1/models, POST /v1/anomaly/payroll (+batch), POST /v1/anomaly/attendance (+batch), POST /v1/chat. Not started (no uvicorn here) |
| All 12 official dialogues over real HTTP (stdlib replica of `mock_server.py`, header X-Demo-User) | 28/28 turns got the expected intent; 9 turns made a service call; salary_of_other (DLG-05) refused without any call; follow-ups reused the slot |
| Chatbot held-out accuracy | 81.06% (107/132), as reported by the repo; not improved to 90%+ |
| CX-06 per-error-type evaluation (the item the repo says is blocked) | done with the authorised label file, see below |

## CX-06: payroll results per error type (labels used for measurement only; frozen predictions from `reports/phase2/payroll_predictions_frozen.csv`)
| error_type | support | tp | fn | precision | recall | rules_recall | model_recall |
|---|---|---|---|---|---|---|---|
| DOUBLE_PAYMENT | 15 | 15 | 0 | 0.068 | 1.000 | 1.000 | 0.000 |
| MISSED_DEDUCTION | 15 | 14 | 1 | 0.064 | 0.933 | 0.933 | 0.067 |
| NEGATIVE_NET | 8 | 8 | 0 | 0.037 | 1.000 | 1.000 | 0.625 |
| OVERTIME_SPIKE | 15 | 11 | 4 | 0.051 | 0.733 | 0.533 | 0.733 |
| ROLE_MISMATCH | 15 | 15 | 0 | 0.068 | 1.000 | 1.000 | 0.000 |
| STALE_SALARY | 12 | 12 | 0 | 0.055 | 1.000 | 1.000 | 0.000 |
| WRONG_DECIMAL | 15 | 15 | 0 | 0.068 | 1.000 | 1.000 | 0.667 |
| ZERO_PAY | 5 | 5 | 0 | 0.024 | 1.000 | 1.000 | 0.600 |
| overall | 100 | 95 | 5 | 0.316 | 0.950 | 0.920 | 0.300 |

Overall: rules + model flagged 301 payslips, caught 95 of 100 injected errors: **precision 0.316, recall 0.95**; rules alone: 107 flagged, precision 0.860, recall 0.92; model alone: 221 flagged, precision 0.136, recall 0.30. Provided baseline (`anomaly_baselines`): precision 0.217, recall 0.40.
- Rules caught what the model missed: DOUBLE_PAYMENT, ROLE_MISMATCH, STALE_SALARY (model recall 0 on all three).
- Hardest overall: OVERTIME_SPIKE (combined recall 0.733, 4 of 15 missed; spikes below the 50-hour cap look normal).
- Honest caveat: I cannot verify from the zip that the rules were designed before the labels were seen; the repo states the predictions were frozen first. The precision of the combined detector is limited by 206 pooled false positives.
Saved as `reports/phase2/payroll_cx06_per_type.csv`. To make notebook 11 show it, copy the label file to `04_Validation_Vault/unlock_CX-06.csv` and re-run notebook 11 (the vault is restricted: do not commit it).

## Problems found
1. `reports/phase2/chatbot_dialogue_results.csv` records `service_called = False` for 26 of 28 turns. A run over real HTTP made 9 calls (leave balance, payslip, manager, attendance, unusual attendance, payroll flags): the old file looks like an offline run and understates the behaviour. Corrected evidence: `reports/phase2/chatbot_dialogue_results_over_http.csv`. Regenerate it from notebook 15 with the mock service running.
2. `PHASE2_IMPLEMENTATION_README.md`, `docs/phase2_quality_gate.md` and notebook 11 still say CX-06 evaluation is blocked: update after step above.
3. Four official intents (attrition risk, salary band, headcount forecast, resume score) answer "that feature is not available yet": correct for Phase 2 (they belong to later phases), but say so in your report.
4. DLG-02 "is my March payslip ready" answers that no record exists for 2026-03 (the demo data covers recent months only); DLG-08 asks "this month" and the scored payroll has no 2026-10 run: both honest, but explain them if asked.
5. Nothing here has a Git tag. Run `git tag phase-2` and `git push --tags` after committing.

## Not verified here
API tests and the live FastAPI app, the Streamlit app, the official uvicorn mock server, a clean-kernel re-run of the 10 Phase 2 notebooks, and any PDF report.


## Follow-up after this review (v8)
- Problem 1 fixed at the source: `scripts/regenerate_phase2_reports.py` now runs the 12 dialogues over real HTTP, so `chatbot_dialogue_results.csv` shows 9 service calls (the separate `_over_http.csv` is no longer needed).
- Problem 2 fixed: README, `PHASE2_IMPLEMENTATION_README.md`, `docs/phase2_quality_gate.md`, the validation summary and notebook `P2_T06` now point to the measured CX-06 table. The reviewed table is internally consistent (tp + fn = support; 95 tp of 301 flagged) but the label file was never inspected by the package author.
