"""Task 8: the HTTP layer.  Needs pytest + fastapi + httpx (pip install -r 05_Starter/requirements.txt).
These tests were written without a web server available, so run them on your machine: pytest -q tests_phase2/test_api.py"""
import sys

import pytest

pytest.importorskip("fastapi"); pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from _helpers import ROOT, api_payload  # noqa: E402

sys.path.insert(0, str(ROOT / "05_Starter" / "api_skeleton"))
from app.main import app  # noqa: E402

OFFICIAL = ["result_id", "source_dataset", "record_id", "employee_id", "period", "anomaly_score", "flagged",
            "reason_code", "reason_text", "money_at_risk", "model_version", "scored_at"]
ATT = dict(employee_id="E1", month="2026-09", scheduled_days=20, present_days=20, avg_hours_worked=8.5,
           late_arrivals=1, overtime_hours=0.0)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:          # the 'with' runs start-up, so models are loaded once here
        yield c


def test_health_reports_loaded_models(client):
    j = client.get("/v1/health").json()
    assert j["status"] == "ok" and all(j["models_loaded"].values()) and "payroll" in j["model_versions"]


def test_payroll_clean_row_returns_official_format_and_no_issue(client):
    r = client.post("/v1/anomaly/payroll", json=api_payload())
    assert r.status_code == 200
    j = r.json()
    assert list(j) == OFFICIAL and j["flagged"] == "N" and j["reason_text"] == "No issue found." and j["anomaly_score"] < 0.6


def test_payroll_negative_net_is_flagged_with_a_real_score_and_money(client):
    r = client.post("/v1/anomaly/payroll", json=api_payload(net_pay=-500, total_deductions=50500))
    j = r.json()
    assert r.status_code == 200 and j["flagged"] == "Y" and j["anomaly_score"] >= 0.9 and j["money_at_risk"] > 0
    assert "NEGATIVE_NET_PAY" in j["reason_code"] and "below zero" in j["reason_text"]


def test_payroll_duplicate_of_a_known_payslip_is_caught(client):
    row = api_payload(payroll_item_id="API-DUP", run_id="PR-2025-04", employee_id="EMP-0002", month="2025-04-01")
    j = client.post("/v1/anomaly/payroll", json=row).json()
    assert "DUPLICATE_PAYMENT" in j["reason_code"]


def test_payroll_bad_input_returns_422(client):
    assert client.post("/v1/anomaly/payroll", json={"employee_id": "E1"}).status_code == 422                 # missing fields
    assert client.post("/v1/anomaly/payroll", json=api_payload(month="September")).status_code == 422
    assert client.post("/v1/anomaly/payroll", json=api_payload(payable_days=28, lop_days=9)).status_code == 422
    assert client.post("/v1/anomaly/payroll", json=api_payload(tds=-5)).status_code == 422
    assert client.post("/v1/anomaly/payroll", json=api_payload(surprise="x")).status_code == 422           # unknown field


def test_attendance_clean_and_zero_attendance(client):
    ok = client.post("/v1/anomaly/attendance", json=ATT).json()
    assert list(ok) == OFFICIAL and ok["flagged"] == "N"
    bad = client.post("/v1/anomaly/attendance", json=dict(ATT, present_days=0, avg_hours_worked=0)).json()
    assert bad["flagged"] == "Y" and "NO_ATTENDANCE" in bad["reason_code"]


def test_attendance_bad_input_returns_422(client):
    assert client.post("/v1/anomaly/attendance", json=dict(ATT, present_days=25)).status_code == 422   # present > scheduled
    assert client.post("/v1/anomaly/attendance", json=dict(ATT, avg_hours_worked=30)).status_code == 422
    assert client.post("/v1/anomaly/attendance", json=dict(ATT, month="2026-99")).status_code == 422
    assert client.post("/v1/anomaly/attendance", json={"scheduled_days": 20}).status_code == 422


def test_batch_ranks_by_money_and_enforces_the_limit(client):
    rows = [api_payload(payroll_item_id="B-OK", employee_id="EMP-0003"),
            api_payload(payroll_item_id="B-BAD", employee_id="EMP-0004", gross_pay=500000, net_pay=480000, total_deductions=20000)]
    j = client.post("/v1/anomaly/payroll/batch", json={"rows": rows}).json()
    assert j["count"] == 2 and j["flagged_count"] == 1 and j["results"][0]["record_id"] == "B-BAD"
    too_many = {"rows": [api_payload(payroll_item_id=f"X{i}", employee_id=f"EMP-9{i:03d}") for i in range(501)]}
    assert client.post("/v1/anomaly/payroll/batch", json=too_many).status_code == 422
    assert client.post("/v1/anomaly/payroll/batch", json={"rows": []}).status_code == 422


def test_chat_refuses_salary_of_other_and_rejects_blank(client):
    j = client.post("/v1/chat", json={"message": "what is Priya Nair salary", "session_id": "t1", "demo_user": "EMP-0002"}).json()
    assert j["intent"] == "salary_of_other" and j["service_called"] is False and "model_version" in j
    assert client.post("/v1/chat", json={"message": "   ", "session_id": "t1"}).status_code == 422
    assert client.post("/v1/chat", json={"message": "hi"}).status_code == 422


def test_chat_ignores_a_client_supplied_role(client):
    j = client.post("/v1/chat", json={"message": "suggest a salary band for Data Analyst", "session_id": "t2",
                                       "user_role": "HR", "demo_user": "EMP-0002"}).json()
    assert "permission" in j["reply"].lower()
