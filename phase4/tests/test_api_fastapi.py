"""Day 17: the real FastAPI + Pydantic layer. Needs: pip install -r requirements-dev.txt (skipped, and reported as skipped, otherwise)."""
from _common import need, registry, skip


def _client(api_key=None):
    need("fastapi", "pydantic", "httpx")
    from dataclasses import replace

    from fastapi.testclient import TestClient

    from cortexhr_phase4.api.app import create_app
    from cortexhr_phase4.core.settings import load_settings
    return TestClient(create_app(replace(load_settings(), api_key=api_key, auth_secret=None, auth_users=""), registry()))


def test_openapi_lists_every_v1_route():
    c = _client()
    paths = set(c.get("/openapi.json").json()["paths"])
    for p in ("/v1/health", "/v1/models", "/v1/attrition/predict", "/v1/anomaly/payroll", "/v1/anomaly/attendance", "/v1/chat", "/v1/resume/rank", "/v1/payband/recommend", "/v1/forecast", "/v1/monitoring"):
        assert p in paths, p
    assert all(p.startswith("/v1/") or p in ("/docs", "/openapi.json", "/redoc") for p in paths)


def test_health_open_and_models_protected_when_key_set():
    c = _client("k")
    assert c.get("/v1/health").status_code == 200
    r = c.get("/v1/models")
    assert r.status_code == 401 and r.json()["error"] == "unauthorized"
    assert c.get("/v1/models", headers={"X-API-Key": "k"}).status_code == 200


def test_validation_errors_use_the_common_json_shape():
    c = _client()
    r = c.post("/v1/attrition/predict", json={"wrong": 1})
    assert r.status_code == 422 and r.json()["error"] == "validation" and r.json()["details"]
    r = c.post("/v1/anomaly/payroll", json={"month": "bad"})
    assert r.status_code == 422
    r = c.get("/v1/does-not-exist")
    assert r.status_code == 404 and r.json()["error"] == "not_found"


def test_each_model_endpoint_returns_a_real_prediction():
    from cortexhr_phase4.services import samples
    from _common import registry, skip
    if registry().info.get("payroll") and registry().info["payroll"].status == "unavailable":
        skip("model_unavailable: model_unavailable: payroll model unavailable in this runtime; install the exact pinned pandas/model environment")
    c = _client()
    assert c.post("/v1/attrition/predict", json={"employee_id": samples.employee_id()}).json()["risk_band"] in ("Low", "Medium", "High")
    assert c.post("/v1/anomaly/payroll", json=samples.payroll_row()).json()["count"] == 1
    assert c.post("/v1/payband/recommend", json=samples.payband_row()).json()["recommended_mid"] > 0
    assert len(c.post("/v1/resume/rank", json={"job_id": "J01", "top_k": 3}).json()["candidates"]) == 3
    assert len(c.get("/v1/forecast").json()["forecast"]) == 6


def test_pydantic_schemas_reject_impossible_values():
    need("pydantic")
    from pydantic import ValidationError

    from cortexhr_phase4.api import schemas
    from cortexhr_phase4.services import samples
    ok = samples.payroll_row()
    schemas.PayrollIn(**ok)
    for bad in ({"payable_days": 40}, {"month": "2025-13-45x"}, {"unknown_field": 1}, {"net_pay": float("nan")}):
        try:
            schemas.PayrollIn(**{**ok, **bad})
            raise AssertionError(f"accepted {bad}")
        except ValidationError:
            pass
    try:
        schemas.AttendanceIn(employee_id="E", scheduled_days=10, present_days=12, avg_hours_worked=8)
        raise AssertionError("present_days > scheduled_days accepted")
    except ValidationError:
        pass


def test_hr_assistant_analytics_intents_through_real_api_route():
    c = _client()
    questions = {
        "How many employees are at high attrition risk?": "attrition_high_risk_count",
        "What is the attendance rate?": "attendance_rate",
        "What is the workforce forecast?": "workforce_forecast",
        "How many candidates were screened?": "recruitment_screening_summary",
    }
    for question, intent in questions.items():
        response = c.post("/v1/chat", json={"message": question, "session_id": "api-mind-test"})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["intent"] == intent
        assert body["reply"] and "human review" in body["notice"].lower()
