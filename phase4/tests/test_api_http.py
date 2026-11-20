"""Day 17 over real HTTP: smoke_test.py, the console API client, error shapes, API key.

Uses the Flask stand-in (tests/support/flask_double.py) ONLY when FastAPI is not installed; with FastAPI installed see test_api_fastapi.py."""
import subprocess
import sys

from _common import ROOT, registry

from cortexhr_phase4.console.api_client import ApiClient
from support.flask_double import Running, make_app


def test_smoke_test_script_passes_every_model():
    reg = registry()
    from _common import skip
    if any(reg.info.get(k) and reg.info[k].status == "unavailable" for k in ("payroll", "attendance", "resume", "payband", "chatbot")):
        skip("model_unavailable: one or more required model artifacts are unavailable in this runtime")
    with Running(make_app(reg)) as srv:
        out = subprocess.run([sys.executable, str(ROOT / "scripts" / "smoke_test.py"), "--base-url", srv.url], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "10 of 10 checks passed" in out.stdout


def test_smoke_test_script_fails_loudly_when_api_is_down():
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "smoke_test.py"), "--base-url", "http://127.0.0.1:9"], capture_output=True, text=True, timeout=120)
    assert out.returncode == 1 and "FAIL" in out.stdout and "0 of 10" in out.stdout


def test_client_friendly_messages_for_each_failure_mode():
    reg = registry()
    from _common import skip
    if reg.info.get("payroll") and reg.info["payroll"].status == "unavailable":
        skip("model_unavailable: payroll model unavailable in this runtime")
    with Running(make_app(reg, api_key="s3cret")) as srv:
        assert ApiClient(srv.url, "s3cret").get("/v1/health").ok
        no_key = ApiClient(srv.url).get("/v1/models")
        assert no_key.status == 401 and "CORTEXHR_API_KEY" in no_key.message()
        nf = ApiClient(srv.url, "s3cret").post("/v1/attrition/predict", {"employee_id": "EMP-NOPE"})
        assert nf.status == 404 and nf.error == "unknown_employee" and "not accepted" in nf.message()
        bad = ApiClient(srv.url, "s3cret").post("/v1/payband/recommend", {"department_id": "D01"})
        assert not bad.ok and bad.status in (422, 500)
    down = ApiClient("http://127.0.0.1:9", timeout=2).get("/v1/health")
    assert down.status == 0 and "Cannot reach the CortexHR API" in down.message("http://127.0.0.1:9")


def test_every_documented_get_endpoint_returns_json():
    reg = registry()
    from _common import skip
    if reg.info.get("payroll") and reg.info["payroll"].status == "unavailable":
        skip("model_unavailable: payroll model unavailable in this runtime")
    with Running(make_app(reg)) as srv:
        c = ApiClient(srv.url)
        for path in ("/v1/health", "/v1/models", "/v1/overview", "/v1/attrition/employees", "/v1/resume/jobs", "/v1/payband/options", "/v1/forecast", "/v1/monitoring", "/v1/performance"):
            r = c.get(path)
            assert r.ok and isinstance(r.data, dict) and r.data, path
        assert c.get("/v1/forecast", metric="exits").data["forecast"][0]["lower_80"] >= 0
        assert c.get("/v1/forecast", metric="headcount_end", department_id="D99").status == 404
        sc = c.post("/v1/forecast/scenario", {"scenarios": [{"effect": "hiring_freeze", "start": "2026-11", "end": "2027-01"}]})
        assert sc.status in (200, 503), "scenario needs statsmodels: 503 with a clear message when it is missing"
        if sc.status == 503:
            assert "statsmodels" in " ".join(sc.details)
