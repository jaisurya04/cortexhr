"""Smoke test of a running CortexHR API: /v1/health plus ONE real prediction for every model.

    python phase4/scripts/smoke_test.py --base-url http://127.0.0.1:8000
    python phase4/scripts/smoke_test.py --base-url https://<your-live-address> --api-key "$CORTEXHR_API_KEY"

Standard library only, so it also runs from a phone (Termux / a-Shell) or any CI runner. Exit code 0 = every check passed.
The payloads are real rows copied from the project's own data.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

PAYROLL = {"payroll_item_id": "PI-000001", "run_id": "PR-2025-04", "employee_id": "EMP-0002", "month": "2025-04", "department_id": "D08", "role_id": "R047",
           "job_level": 4, "overtime_eligible": "N", "monthly_fixed_gross": 144460.0, "payable_days": 30.0, "lop_days": 0.0, "overtime_hours_paid": 0.0, "overtime_pay": 0.0,
           "variable_pay": 34670.4, "variable_pay_type": "Performance Bonus", "gross_pay": 179130.4, "pf_employee": 1800.0, "esi_employee": 0.0, "professional_tax": 200.0,
           "tds": 18625.79, "other_deductions": 0.0, "total_deductions": 20625.79, "net_pay": 158504.61}
ATTENDANCE = {"employee_id": "EMP-0002", "month": "2025-04", "scheduled_days": 20, "present_days": 20, "wfh_days": 18, "late_arrivals": 1, "avg_hours_worked": 8.5, "overtime_hours": 0.0}
PAYBAND = {"department_id": "D01", "role_id": "R002", "job_level": 1, "band_mid_2026": 500000, "market_index": 0.4676, "prior_experience_years": 0.0, "skills_count": 5,
           "work_location": "Pune", "highest_education": "B.E. Computer Science"}


def request_headers(key: str | None = None, token: str | None = None) -> dict[str, str]:
    """Build headers for both optional API-key and password-login bearer auth."""
    headers = {"Content-Type": "application/json"}
    if key:
        headers["X-API-Key"] = key
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def call(base: str, method: str, path: str, key: str | None, body: dict | None = None,
         timeout: float = 60, token: str | None = None) -> tuple[int, dict]:
    req = urllib.request.Request(base.rstrip("/") + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers=request_headers(key, token))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode())
        except Exception:
            return e.code, {"error": "non-json error body"}


def checks(base: str, key: str | None, token: str | None = None) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []

    def run(name, method, path, body, verify):
        t = time.time()
        try:
            status, js = call(base, method, path, key, body, token=token)
            ok, note = (status == 200 and bool(verify(js))), f"HTTP {status}"
            if status != 200:
                note += f" {js.get('error', '')} {js.get('details', '')}"[:120]
        except Exception as exc:                                  # network failure is a failed check, not a crash
            ok, note = False, f"{type(exc).__name__}: {exc}"[:120]
        out.append((name, ok, f"{note} ({time.time() - t:.1f}s)"))

    run("health", "GET", "/v1/health", None, lambda j: j["status"] in ("ok", "degraded") and j["version"])
    run("models", "GET", "/v1/models", None, lambda j: all(m["status"] != "unavailable" for m in j["models"].values()))
    run("attrition", "POST", "/v1/attrition/predict", {"employee_id": "EMP-0002"}, lambda j: 0 <= j["risk_score_12m"] <= 1 and j["risk_band"] in ("Low", "Medium", "High"))
    run("payroll anomaly", "POST", "/v1/anomaly/payroll", PAYROLL, lambda j: j["count"] == 1)
    run("attendance anomaly", "POST", "/v1/anomaly/attendance", ATTENDANCE, lambda j: j["count"] == 1)
    run("chatbot", "POST", "/v1/chat", {"message": "What is the notice period?", "session_id": "smoke-test"}, lambda j: j["reply"])
    run("resume ranking", "POST", "/v1/resume/rank", {"job_id": "J01", "top_k": 3}, lambda j: len(j["candidates"]) == 3)
    run("pay band", "POST", "/v1/payband/recommend", PAYBAND, lambda j: j["recommended_low"] < j["recommended_mid"] < j["recommended_high"])
    run("forecast", "GET", "/v1/forecast?metric=headcount_end&department_id=ALL", None, lambda j: len(j["forecast"]) == 6)
    run("monitoring", "GET", "/v1/monitoring", None, lambda j: j["status"] in ("OK", "WATCH", "ALERT"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default=os.environ.get("CORTEXHR_API_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--api-key", default=os.environ.get("CORTEXHR_API_KEY"))
    ap.add_argument("--token", default=os.environ.get("CORTEXHR_SMOKE_TOKEN"), help="Existing Bearer token; alternatively set CORTEXHR_SMOKE_TOKEN")
    ap.add_argument("--username", default=os.environ.get("CORTEXHR_SMOKE_USERNAME"), help="Login username; alternatively set CORTEXHR_SMOKE_USERNAME")
    a = ap.parse_args()
    password = os.environ.get("CORTEXHR_SMOKE_PASSWORD")
    if a.token and a.username:
        ap.error("Use either --token or --username, not both.")
    if bool(a.username) != bool(password):
        ap.error("For password login, set both CORTEXHR_SMOKE_USERNAME and CORTEXHR_SMOKE_PASSWORD.")
    token = a.token
    if a.username:
        status, payload = call(a.base_url, "POST", "/v1/auth/login", a.api_key,
                               {"username": a.username, "password": password})
        if status != 200 or not payload.get("access_token"):
            print(f"Login failed: HTTP {status} {payload}")
            return 1
        token = payload["access_token"]
    print(f"CortexHR smoke test against {a.base_url}\n")
    res = checks(a.base_url, a.api_key, token)
    for name, ok, note in res:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:20s} {note}")
    failed = [n for n, ok, _ in res if not ok]
    print(f"\n{len(res) - len(failed)} of {len(res)} checks passed" + (f"; failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
