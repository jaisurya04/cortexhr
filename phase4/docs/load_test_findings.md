# Day 17 think-for-yourself: "What breaks first if 100 people call it at once? Guess, then test."

**Guess (written before the test):** the CPU-heavy calls break first: monitoring refresh (it re-fits models) and the anomaly detectors (Isolation Forest with 300 trees); memory is a second risk on a 512 MB instance.

**Test:** `python scripts/load_test.py --base-url <address> --users 100` (100 users start at the same moment and each calls ten endpoints). Run here against the Flask stand-in that calls the same handlers, in the build sandbox. **This is not a measurement of FastAPI/uvicorn or of a Render instance**; repeat it against your deployed API and compare.

| Run | Total time | Errors | Slowest endpoint at p95 |
|---|---|---|---|
| Before the fix | 87.1 s | 2 (monitoring) | monitoring refresh, 53 s |
| After the fix | 23.4 s | 0 | payroll anomaly, 6.0 s |

**What broke first (confirmed):**
1. `GET /v1/monitoring?refresh=true`: every request re-ran the whole monitor (about 0.4 s of CPU each, 100 in parallel) and two readers received an error because the report file was read while it was being rewritten. **Fixed:** the report is written to a temporary file and renamed (atomic); refreshes are single-flight and reuse a result younger than 60 seconds. Regression test: `tests/test_services_models.py::test_concurrent_monitoring_refresh_runs_once_and_never_returns_a_broken_file`.
2. After the fix the slowest calls are the scoring endpoints for payroll and attendance (CPU-bound Isolation Forest scoring). Their p95 was 5-6 s with 100 simultaneous callers on this machine.

**What to expect on a free Render instance:** 0.1 CPU is far less than this sandbox, so latencies will be much higher and the first call after 15 idle minutes takes about a minute. Mitigations if needed: batch endpoints (`/anomaly/*/batch`), a paid instance, or scoring in a background queue. Not done: they were not required for a demo.
