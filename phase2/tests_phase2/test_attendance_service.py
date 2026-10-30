"""Task 8: attendance detector, saved models, registry, load-once and retrain fallback."""
import json
import tempfile
from pathlib import Path

import pandas as pd

from _helpers import CSV
from cortexhr.phase2 import attendance as A
from cortexhr.phase2 import payroll as P
from cortexhr.phase2 import service as S

_C = {}


def _det():
    if "d" not in _C:
        _C["d"] = A.fit_attendance_detector(pd.read_csv(CSV / "attendance_monthly_features.csv"))
    return _C["d"]


def _row(**o):
    base = dict(employee_id="E1", scheduled_days=20, present_days=20, avg_hours_worked=8.5, late_arrivals=1, overtime_hours=0.0)
    base.update(o)
    return pd.DataFrame([base])


def test_normal_month_is_not_flagged_and_zero_month_is():
    ok = A.score_attendance(_row(), _det())
    assert not bool(ok.flagged.iloc[0]) and ok.reason_code.iloc[0] == A.NOT_FLAGGED
    bad = A.score_attendance(_row(present_days=0, avg_hours_worked=0.0), _det())
    assert bool(bad.flagged.iloc[0]) and "NO_ATTENDANCE" in bad.reason_code.iloc[0] and bad.anomaly_score.iloc[0] >= 0.9


def test_paid_leave_explains_a_zero_attendance_month():
    out = A.score_attendance(_row(present_days=0, avg_hours_worked=0.0, paid_leave_days=20), _det())
    assert "NO_ATTENDANCE" not in out.reason_code.iloc[0]


def test_other_rules():
    d = _det()
    assert "IMPLAUSIBLE_HOURS" in A.score_attendance(_row(avg_hours_worked=15.0), d).reason_code.iloc[0]
    assert "EXTREME_OVERTIME" in A.score_attendance(_row(overtime_hours=95.0), d).reason_code.iloc[0]
    assert "LONG_UNPAID_ABSENCE" in A.score_attendance(_row(longest_unpaid_streak_days=6.0, unpaid_absence_days=6.0), d).reason_code.iloc[0]
    assert "LATE_ARRIVALS_EXCEED_SCHEDULE" in A.score_attendance(_row(late_arrivals=25), d).reason_code.iloc[0]


def test_attendance_official_format_and_money_is_blank():
    sample = pd.read_csv(CSV / "sample_anomaly_results.csv")
    out = A.to_official_format(A.score_attendance(pd.concat([_row(), _row(present_days=0, avg_hours_worked=0.0)]), _det()))
    assert list(out.columns) == list(sample.columns)
    assert out.flagged.iloc[0] == "Y" and out.money_at_risk.isna().all()


def test_train_all_saves_models_and_registry_then_loads_them():
    with tempfile.TemporaryDirectory() as d:
        reg = S.train_all(models_dir=Path(d))
        assert {"payroll", "attendance", "chatbot"} <= set(reg)
        for name, info in reg.items():
            assert (Path(d) / info["file"]).exists() and len(info["sha256"]) == 64 and info["version"]
        assert reg["payroll"]["labels_used"] is False and reg["attendance"]["labels_used"] is False
        assert json.loads((Path(d) / S.REGISTRY).read_text()) == reg
        assert isinstance(S.load_payroll_detector(models_dir=Path(d)), P.PayrollDetector)
        assert isinstance(S.load_attendance_detector(models_dir=Path(d)), A.AttendanceDetector)


def test_corrupt_model_file_triggers_retrain_instead_of_crash():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / S.PAYROLL_FILE).write_bytes(b"not a joblib file")
        det = S.load_payroll_detector(models_dir=Path(d))
        assert isinstance(det, P.PayrollDetector) and (Path(d) / S.REGISTRY).exists()


def test_the_models_shipped_in_the_repo_load_and_match_the_registry():
    import hashlib
    reg = S.read_registry()
    assert {"payroll", "attendance", "chatbot"} <= set(reg), "run python scripts/train_phase2_models.py"
    for name, info in reg.items():
        data = (S.MODELS_DIR / info["file"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == info["sha256"], f"{name} file changed after it was registered"
    assert isinstance(S.load_payroll_detector(), P.PayrollDetector)


def test_service_scores_rows_in_the_official_shape_and_ranks_batches():
    det = S.load_payroll_detector()
    good = dict(payroll_item_id="G1", run_id="PR-2026-10", employee_id="EMP-0002", month="2026-10", monthly_fixed_gross=50000,
                payable_days=31, lop_days=0, overtime_hours_paid=0, overtime_pay=0, variable_pay=0, gross_pay=50000,
                pf_employee=1800, esi_employee=0, professional_tax=200, tds=1000, other_deductions=0, total_deductions=3000,
                net_pay=47000)
    bad = dict(good, payroll_item_id="B1", employee_id="EMP-0003", gross_pay=500000, net_pay=480000, total_deductions=20000)
    out = S.score_payroll_rows([good, bad], det)
    assert [r["record_id"] for r in out][0] == "B1" and out[0]["flagged"] == "Y" and out[1]["flagged"] == "N"
    assert out[1]["reason_text"] == "No issue found." and out[0]["money_at_risk"] > 400000
    assert list(out[0]) == list(pd.read_csv(CSV / "sample_anomaly_results.csv").columns)
    try:
        S.score_payroll_rows([dict(good, month="September")], det)
    except ValueError:
        pass
    else:
        raise AssertionError("bad month must raise ValueError (the API turns it into 422)")
