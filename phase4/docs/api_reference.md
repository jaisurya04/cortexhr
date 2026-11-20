# API reference (`/v1`, version 1.0.0)

Interactive docs: `<api address>/docs`. JSON endpoints use JSON bodies. If `CORTEXHR_API_KEY` is set, send `X-API-Key: <key>` on protected endpoints. When `CORTEXHR_AUTH_SECRET` is configured, sign in with `/v1/auth/login` and send `Authorization: Bearer <token>` to protected endpoints; roles are enforced server-side.
Every response that contains an AI output carries a `notice` that starts with or contains **"AI-generated, for human review"**.

## Errors
```json
{"error": "unknown_employee", "details": ["'EMP-NOPE' is not in the 2026 scoring population (430 active employees)"]}
```
| Status | `error` | When |
|---|---|---|
| 401 | `unauthorized` | key or bearer token required and missing / wrong / expired |
| 403 | `forbidden` | authenticated role is not allowed to access this endpoint |
| 413 | `file_too_large` | resume upload exceeds 8 MB |
| 415 | `unsupported_file` | resume upload is not PDF or DOCX |
| 404 | `unknown_employee`, `unknown_series`, `not_found` | id or path does not exist |
| 422 | `validation` | body or query invalid (Pydantic or domain rule); `details` lists each field |
| 503 | `model_unavailable` | the model or library a call needs is not available (message says which) |
| 500 | `internal_error` | unexpected; the message carries a request id, logs carry the stack trace |

## Examples
```bash
curl $API/v1/health
curl -X POST $API/v1/attrition/predict -H "Content-Type: application/json" -d '{"employee_id": "EMP-0002"}'
curl -X POST $API/v1/resume/rank -H "Content-Type: application/json" -d '{"job_id": "J01", "top_k": 3}'
curl "$API/v1/forecast?metric=headcount_end&department_id=ALL"
curl "$API/v1/monitoring?refresh=true"
curl -X POST $API/v1/forecast/scenario -H "Content-Type: application/json" \
     -d '{"scenarios": [{"effect": "hiring_freeze", "start": "2026-11", "end": "2027-01"}]}'
```

## Endpoint details
| Endpoint | Request | Response (main fields) |
|---|---|---|
| `GET /v1/health` | - | `status` ok / degraded / unavailable, `version`, `uptime_s`, `models{name: state}` |
| `POST /v1/auth/login` | `{username, password}` | 8-hour signed bearer token and account role |
| `GET /v1/auth/me` | Bearer token | Validates the session and returns the authenticated identity |
| `POST /v1/auth/logout` | Bearer token | Revokes the token for the lifetime of the current API process |
| `GET /v1/models` | - | per model `status`, `version`, `source`, `detail` |
| `POST /v1/attrition/predict` | `{employee_id}` | `risk_score_12m`, `risk_band`, `top_reasons[3]`, `model_version`, `source` (`live_model` or `saved_phase3_scores`) |
| `GET /v1/attrition/employees` | `band`, `department_id`, `limit<=500` | `count`, `population`, `bands`, `mean_risk`, `employees[]` |
| `POST /v1/anomaly/payroll` | one payslip (see `schemas.PayrollIn`) | `count`, `flagged_count`, `results[]` with score and reasons |
| `POST /v1/anomaly/attendance` | one attendance month | same shape |
| `POST /v1/chat` | `{message, session_id, demo_user?}` | `reply`, `intent`, `confidence` |
| `POST /v1/resume/rank` | `{job_id, top_k}` or `{job{...}, resume_ids?, top_k}` | `candidates[]` ranked with scores; protected fields (name, gender, age, ...) are refused |
| `POST /v1/resume/upload` | multipart `file`, `job_id`, `top_k` | In-memory PDF/DOCX text extraction and temporary candidate ranking; 8 MB cap, 20-page PDF cap; uploaded data is cleared on API restart |
| `POST /v1/payband/recommend` | 9 job-related fields | `recommended_low/mid/high`, `inputs_used`, `never_used` |
| `GET /v1/forecast` | `metric` = headcount_end / exits / attrition_rate_monthly_pct, `department_id` | `history`, `forecast[{forecast_month, forecast_value, lower_80, upper_80}]`, `events`, `accuracy`, `explanation` |
| `POST /v1/forecast/scenario` | `scenarios[{effect, start, end, department_id}]` | what-if forecast (503 if statsmodels missing) |
| `GET /v1/monitoring` | `refresh` | `status`, `coverage`, `not_checked`, `explanation`, `report`, `history` |

Thresholds used by monitoring: PSI < 0.10 OK, 0.10 to 0.25 WATCH, > 0.25 ALERT; pay gap WATCH at 0.02, ALERT at 0.05; twin-resume tolerance 0.001 (see `monitoring_policy.md`).
