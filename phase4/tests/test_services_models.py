"""Day 17 (model layer): every endpoint's logic against the real saved models, with numbers cross-checked against source data."""
import json

import numpy as np
import pandas as pd
from _common import csv_dir, registry

from cortexhr_phase4.core.settings import PHASE3
from cortexhr_phase4.services import handlers as h
from cortexhr_phase4.services import samples
from cortexhr_phase4.services.errors import ApiError
from cortexhr_phase4.services.util import jsonable


def _err(fn, *a, **k):
    try:
        fn(*a, **k)
    except ApiError as e:
        return e
    raise AssertionError("expected ApiError")


def test_registry_loads_every_model_and_is_honest_about_fallbacks():
    r = registry()
    from _common import skip
    if any(i.status == "unavailable" for i in r.info.values()):
        skip("model_unavailable: at least one original model artifact requires the pinned Python/pandas environment")
    assert set(r.info) == {"attrition", "payroll", "attendance", "chatbot", "resume", "payband", "forecast"} and r.ready()
    for name, i in r.info.items():
        assert i.status in ("loaded", "fallback") and i.version, name
        if i.status == "fallback":
            assert i.detail, f"{name}: a fallback must say why"


def test_health_and_models_shape():
    r = registry()
    hd = h.health(r)
    assert hd["status"] in ("ok", "degraded", "unavailable") and hd["version"] == "1.0.0" and set(hd["models"]) == set(r.info)
    payroll_status = r.info["payroll"].status
    assert h.models(r)["models"]["payroll"]["status"] == payroll_status
    assert payroll_status in ("loaded", "unavailable", "fallback")


def test_attrition_predict_matches_phase3_report():
    r = registry()
    saved = pd.read_csv(PHASE3 / "reports" / "task14" / "attrition_risk_2026.csv")
    row = saved.iloc[7]
    out = h.attrition_predict(r, row.employee_id)
    assert abs(out["risk_score_12m"] - row.risk_score_12m) < 1e-3 and out["risk_band"] == row.risk_band and len(out["top_reasons"]) == 3
    assert out["source"] in ("live_model", "saved_phase3_scores") and "human review" in out["notice"].lower()
    e = _err(h.attrition_predict, r, "EMP-NOPE")
    assert e.status == 404 and e.error == "unknown_employee"


def test_attrition_list_bands_equal_phase3_counts():
    r = registry()
    d = h.attrition_list(r, limit=500)
    assert d["bands"] == {"Low": 301, "Medium": 86, "High": 43} and d["population"] == 430 and len(d["employees"]) == 430
    hi = h.attrition_list(r, band="High", limit=500)
    assert hi["count"] == 43 and all(e["risk_band"] == "High" for e in hi["employees"])
    assert [e["risk_score_12m"] for e in hi["employees"]] == sorted((e["risk_score_12m"] for e in hi["employees"]), reverse=True)
    assert _err(h.attrition_list, r, band="Purple").status == 422


def test_payroll_and_attendance_use_the_real_detectors():
    r = registry()
    if r.info["payroll"].status == "unavailable":
        err = _err(h.payroll, r, [samples.payroll_row()])
        assert err.status == 503 and err.error == "model_unavailable"
    else:
        p = h.payroll(r, [samples.payroll_row()])
        assert p["count"] == 1 and p["results"][0]["employee_id"] == "EMP-0002" and "never proof" in p["notice"]
        bad = samples.payroll_row()
        bad["net_pay"] = bad["gross_pay"] * 3                        # net pay above gross is an impossible payslip
        flagged = h.payroll(r, [bad])
        assert flagged["flagged_count"] == 1, "an impossible payslip must be flagged"
        batch = h.payroll(r, [samples.payroll_row(i=i) for i in range(5)])
        assert batch["count"] == 5
    a = h.attendance(r, [samples.attendance_row()])
    assert a["count"] == 1 and a["model_version"]


def test_payroll_invalid_row_is_a_clean_error_not_a_crash():
    r = registry()
    from _common import skip
    if r.info["payroll"].status == "unavailable":
        skip("model_unavailable: payroll validation endpoint requires the pinned model runtime")
    row = samples.payroll_row()
    row["month"] = "not-a-month"
    e = _err(h.payroll, r, [row])
    assert e.status == 422 and e.error == "validation"


def test_chatbot_answers_policy_question_and_rejects_blank():
    r = registry()
    c = h.chat(r, "What is the notice period?", "t1")
    assert c["intent"] == "notice_period_policy" and "30 days" in c["reply"] and c["model_version"]
    assert _err(h.chat, r, "   ", "t1").status == 422


def test_hr_assistant_routes_analytics_questions_to_real_phase_data():
    r = registry()
    cases = [
        ("How many employees are at high attrition risk?", "attrition_high_risk_count", "43"),
        ("Which department has the highest attrition risk?", "attrition_department_risk", "Department"),
        ("What is the attendance rate?", "attendance_rate", "%"),
        ("How many absences were recorded?", "attendance_absences", "unpaid absence days"),
        ("How many candidates were screened?", "recruitment_screening_summary", "unique resumes"),
        ("What is the workforce forecast?", "workforce_forecast", "prediction interval"),
    ]
    for question, intent, expected in cases:
        out = h.chat(r, question, "analytics-test-session")
        assert out["intent"] == intent, (question, out)
        assert expected.casefold() in out["reply"].casefold(), (question, out["reply"])
        assert "human review" in out["notice"].casefold()
        assert "employee_id" not in out and "candidate_name" not in out


def test_hr_assistant_attendance_and_payroll_anomaly_counts_use_saved_evidence():
    r = registry()
    att = h.chat(r, "Show attendance anomalies", "analytics-anomaly-session")
    assert att["intent"] == "attendance_anomalies" and "flagged attendance records" in att["reply"]
    pay = h.chat(r, "Are there any payroll anomalies?", "analytics-anomaly-session")
    assert pay["intent"] == "payroll_anomalies" and "flagged payroll records" in pay["reply"]


def test_hr_assistant_payroll_analytics_is_data_backed_when_source_exists():
    r = registry()
    out = h.chat(r, "What is the total payroll?", "analytics-payroll-session")
    assert out["intent"] == "payroll_total"
    assert "INR" in out["reply"] and "latest supplied payroll month" in out["reply"]


def test_resume_ranking_sorted_and_protected_fields_refused():
    r = registry()
    out = h.resume_rank(r, {"job_id": "J01", "top_k": 5})
    sc = [c["score"] for c in out["candidates"]]
    assert len(sc) == 5 and sc == sorted(sc, reverse=True) and [c["rank"] for c in out["candidates"]] == [1, 2, 3, 4, 5]
    e = _err(h.resume_rank, r, {"job_id": "J01", "top_k": 3, "gender": "F"})
    assert e.status in (400, 422), "a protected attribute in the request must be refused"
    assert _err(h.resume_rank, r, {"job_id": "J99", "top_k": 3}).status in (404, 422)


def test_payband_low_mid_high_and_validation():
    r = registry()
    b = h.payband(r, samples.payband_row())
    assert b["recommended_low"] < b["recommended_mid"] < b["recommended_high"] and "never a final offer" in b["notice"]
    assert "gender" in b["never_used"] and "career_gap_months" not in b["inputs_used"]
    e = _err(h.payband, r, {**samples.payband_row(), "role_id": "XX"})
    assert e.status == 422 and "role_id" in e.details[0]
    e2 = _err(h.payband, r, {**samples.payband_row(), "band_mid_2026": -5})
    assert e2.status == 422


def test_forecast_endpoint_matches_saved_forecast_file():
    r = registry()
    f = h.forecast(r)
    saved = r.obj["forecast"]["forecast"]
    exp = saved[(saved.metric == "headcount_end") & (saved.department_id == "ALL")]
    assert len(f["forecast"]) == 6 and np.allclose([x["forecast_value"] for x in f["forecast"]], exp.forecast_value)
    assert all(x["lower_80"] <= x["forecast_value"] <= x["upper_80"] for x in f["forecast"])
    assert f["history"][-1]["month"] == "2026-09-01" and f["trained_through"] == "2026-09" and f["accuracy"]
    assert _err(h.forecast, r, "headcount_end", "D99").status == 404


def test_overview_numbers_come_from_the_data_not_the_screenshot():
    r = registry()
    o = h.overview(r)
    hist = pd.read_csv(csv_dir() / "monthly_headcount_history.csv", parse_dates=["month"])
    allr = hist[hist.department_id == "ALL"].sort_values("month")
    k = o["kpis"]
    assert k["total_employees"]["value"] == allr.headcount_end.iloc[-1]
    assert abs(k["attrition_risk"]["value"] - pd.read_csv(PHASE3 / "reports" / "task14" / "attrition_risk_2026.csv").risk_score_12m.mean() * 100) < 1e-9
    assert k["open_positions"]["value"] == int(pd.read_csv(csv_dir() / "job_descriptions.csv").openings.sum())
    assert abs(k["avg_salary"]["value"] - pd.read_csv(csv_dir() / "attrition_scoring_2026.csv").annual_fixed_pay.mean()) < 1e-6
    meta = json.loads((PHASE3 / "artifacts" / "metadata" / "salary_band_v1_metadata.json").read_text())
    mp = o["model_performance"]
    assert abs(mp["mae"] - meta["test_metrics"][meta["model"]]["MAE_rupees"]) < 1e-6 and abs(mp["coverage"] - meta["band_coverage_test"]) < 1e-12
    shown = json.dumps(o)
    for demo in ("12480", "12,480", "8.92", "6.1%"):
        assert demo not in shown, f"screenshot demo value {demo!r} leaked into the data"
    assert o["workforce"]["series"][-1]["employees"] == allr.headcount_end.iloc[-1] and len(o["workforce"]["series"]) == 13
    assert o["attrition"]["high"] + o["attrition"]["medium"] + o["attrition"]["low"] == 430


def test_monitoring_handler_returns_status_and_history():
    r = registry()
    m = h.monitoring(r)
    assert m["status"] in ("OK", "WATCH", "ALERT") and "not that the system is fair" in m["notice"] and isinstance(m["history"], list)


def test_performance_endpoint_has_every_evaluation_table():
    p = h.performance(registry())
    assert {"attrition", "resume", "fairness_phase3", "payband"} <= set(p)
    assert "not proven" in p["attrition"]["verdict"].lower()
    ranking = pd.DataFrame(p["resume"]["ranking"])
    assert abs(ranking.iloc[0]["precision@5"] - 0.94) < 1e-9


def test_jsonable_handles_numpy_nan_and_timestamps():
    out = jsonable({"a": np.float64("nan"), "b": np.int64(3), "c": pd.Timestamp("2026-01-01"), "d": [np.bool_(True)], "e": float("inf")})
    assert out == {"a": None, "b": 3, "c": "2026-01-01T00:00:00", "d": [True], "e": None}
    json.dumps(out)


def test_concurrent_monitoring_refresh_runs_once_and_never_returns_a_broken_file():
    """Found by scripts/load_test.py: 100 simultaneous refreshes each re-ran the monitor, and a reader could see a half-written JSON file."""
    import threading
    from dataclasses import replace
    import tempfile
    from pathlib import Path

    from cortexhr_phase4.monitoring import monitor as mon
    r = registry()
    reg2 = type(r)(replace(r.settings, output_dir=Path(tempfile.mkdtemp())))
    calls, errors, results = [], [], []
    orig = mon.run_monitor

    def counted(s, now=None):
        calls.append(1)
        return orig(s, now)

    mon.run_monitor = counted
    h._MON_LAST_RUN["at"] = 0.0
    try:
        def worker(refresh):
            try:
                results.append(h.monitoring(reg2, refresh)["status"])
            except Exception as exc:  # noqa: BLE001
                errors.append(repr(exc))
        ts = [threading.Thread(target=worker, args=(i % 2 == 0,)) for i in range(24)]
        [t.start() for t in ts]
        [t.join() for t in ts]
    finally:
        mon.run_monitor = orig
    assert not errors, errors
    assert len(calls) == 1, f"the monitor ran {len(calls)} times for 24 simultaneous requests"
    assert len(results) == 24 and set(results) <= {"OK", "WATCH", "ALERT"}
    again = h.monitoring(reg2, True)                         # inside the 60 s window: reused, not re-run
    assert len(calls) == 1 and again["refreshed"] is False
