"""Standard-library replica of 05_Starter/mock_services/mock_server.py (same routes, same CSV data, same X-Demo-User
rule).  Use it only on a machine where FastAPI/uvicorn cannot be installed; otherwise run the official one:

    cd 05_Starter/mock_services && uvicorn mock_server:app --port 9000
    python scripts/mock_server_stdlib.py --port 9000        # replica
"""
import argparse
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

D = Path(__file__).resolve().parents[1] / "05_Starter" / "mock_services" / "mock_data"
emp, bal, lv, pay, hol, att, usr = (pd.read_csv(D / f"{n}.csv") for n in
                                    ["employees", "leave_balances", "leave_requests", "payslips", "holidays", "attendance_monthly", "demo_users"])


def _jsonable(obj):
    return json.loads(pd.io.json.ujson_dumps(obj)) if False else json.loads(json.dumps(obj, default=lambda o: None if pd.isna(o) else str(o)))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _send(self, code, payload):
        body = json.dumps(_jsonable(payload)).encode()
        self.send_response(code); self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path); q = {k: v[0] for k, v in parse_qs(url.query).items()}; path = url.path
        if path == "/api/v1/holidays":
            return self._send(200, hol.to_dict("records"))
        e = self.headers.get("X-Demo-User")
        if not e or usr[usr.employee_id == e].empty:
            return self._send(401, {"detail": "Send header X-Demo-User with a valid employee_id"})
        if path == "/api/v1/employees/me":
            return self._send(200, emp[emp.employee_id == e].iloc[0].to_dict())
        if path == "/api/v1/employees/me/team":
            return self._send(200, emp[emp.manager_id == e][["employee_id", "full_name", "role_id"]].to_dict("records"))
        if path == "/api/v1/leave/balance":
            r = bal[bal.employee_id == e]
            return self._send(200, r.iloc[0].to_dict()) if len(r) else self._send(404, {"detail": "No leave balance"})
        if path == "/api/v1/leave/requests":
            r = lv[lv.employee_id == e].sort_values("applied_on", ascending=False).head(int(q.get("limit", 5)))
            return self._send(200, r.to_dict("records"))
        m = re.match(r"^/api/v1/payslips/me/(.+)$", path)
        if m:
            r = pay[(pay.employee_id == e) & (pay.run_id == f"PR-{m.group(1)}")]
            return self._send(200, r.iloc[0].to_dict()) if len(r) else self._send(404, {"detail": "No payslip for that month (use YYYY-MM within the last 6 months)"})
        if path == "/api/v1/attendance/summary":
            r = att[(att.employee_id == e) & att.period_end_date.str.startswith(q.get("month", "?"))]
            return self._send(200, r.iloc[0].to_dict()) if len(r) else self._send(404, {"detail": "No attendance for that month (use YYYY-MM)"})
        return self._send(404, {"detail": "Not found"})


def start(port=9000):
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=9000); a = ap.parse_args()
    s = ThreadingHTTPServer(("127.0.0.1", a.port), Handler); print("replica mock HR service on", a.port); s.serve_forever()
