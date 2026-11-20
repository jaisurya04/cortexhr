"""Optional Flask stand-in for the HTTP contract tests.

The module stays importable when Flask is absent; the individual test reports a
real SKIP rather than aborting the complete Phase 4 test discovery process.
"""
from __future__ import annotations

import threading
from _common import need


def make_app(reg, api_key: str | None = None):
    need("flask", "werkzeug")
    from flask import Flask, jsonify, request
    from cortexhr_phase4.services import handlers as h
    from cortexhr_phase4.services.errors import ApiError

    app = Flask(__name__)

    @app.before_request
    def auth():
        if api_key and request.path != "/v1/health" and request.headers.get("X-API-Key") != api_key:
            return jsonify(error="unauthorized", details=["missing or wrong X-API-Key header"]), 401

    @app.errorhandler(ApiError)
    def api_error(e):
        return jsonify(e.body()), e.status

    @app.errorhandler(Exception)
    def other(e):
        return jsonify(error="internal_error", details=[type(e).__name__]), 500

    j = lambda: request.get_json()
    routes = [("GET", "/v1/health", lambda: h.health(reg)), ("GET", "/v1/models", lambda: h.models(reg)), ("GET", "/v1/overview", lambda: h.overview(reg)),
              ("POST", "/v1/attrition/predict", lambda: h.attrition_predict(reg, j()["employee_id"])),
              ("GET", "/v1/attrition/employees", lambda: h.attrition_list(reg, request.args.get("band"), request.args.get("department_id"), int(request.args.get("limit", 25)))),
              ("POST", "/v1/anomaly/payroll", lambda: h.payroll(reg, [j()])), ("POST", "/v1/anomaly/attendance", lambda: h.attendance(reg, [j()])),
              ("POST", "/v1/chat", lambda: h.chat(reg, j()["message"], j()["session_id"], j().get("demo_user", "EMP-0002"))),
              ("POST", "/v1/resume/rank", lambda: h.resume_rank(reg, j())), ("GET", "/v1/resume/jobs", lambda: h.resume_jobs(reg)),
              ("POST", "/v1/payband/recommend", lambda: h.payband(reg, j())), ("GET", "/v1/payband/options", lambda: h.payband_options(reg)),
              ("GET", "/v1/forecast", lambda: h.forecast(reg, request.args.get("metric", "headcount_end"), request.args.get("department_id", "ALL"))),
              ("POST", "/v1/forecast/scenario", lambda: h.forecast_scenario(reg, j()["scenarios"], j().get("departments"))),
              ("GET", "/v1/monitoring", lambda: h.monitoring(reg, request.args.get("refresh") == "true")), ("GET", "/v1/performance", lambda: h.performance(reg))]
    for i, (method, path, fn) in enumerate(routes):
        app.add_url_rule(path, endpoint=f"r{i}", view_func=lambda fn=fn: jsonify(fn()), methods=[method])
    return app


class Running:
    """Context manager that serves the optional stand-in on a free local port."""
    def __init__(self, app):
        need("flask", "werkzeug")
        from werkzeug.serving import make_server
        self.server = make_server("127.0.0.1", 0, app, threaded=True)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"

    def __enter__(self) -> "Running":
        self.thread.start()
        return self

    def __exit__(self, *a) -> None:
        self.server.shutdown()
