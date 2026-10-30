"""Shared, cached fixtures for the Phase 2 tests (plain functions so they also run without pytest)."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
CSV = ROOT / "03_Data" / "csv"
_CACHE = {}


def payroll_df():
    from cortexhr.phase2.payroll import load_payroll
    if "df" not in _CACHE:
        _CACHE["df"] = load_payroll(CSV / "payroll_monthly_features.csv")
    return _CACHE["df"].copy()


def payroll_scored():
    """Detector fitted on the full file and the scored frame (computed once)."""
    from cortexhr.phase2.payroll import score_payroll
    if "scored" not in _CACHE:
        scored, det = score_payroll(payroll_df())
        _CACHE["scored"], _CACHE["det"] = scored, det
    return _CACHE["scored"].copy(), _CACHE["det"]


def clean_row(offset=0):
    """A real payslip that no rule flags, with a fresh item id and run so it is not a duplicate."""
    from cortexhr.phase2.payroll import payroll_rule_flags
    scored, det = payroll_scored()
    ok = scored[~scored.flagged].payroll_item_id
    df = payroll_df()
    row = df[df.payroll_item_id == ok.iloc[offset]].copy()
    row["payroll_item_id"] = f"TEST-{offset}"
    row["run_id"] = "PR-2026-10"
    return row


def api_payload(**over):
    base = dict(payroll_item_id="API-T1", run_id="PR-2026-10", employee_id="EMP-0002", month="2026-10-01",
                monthly_fixed_gross=50000, payable_days=31, lop_days=0, overtime_hours_paid=0, overtime_pay=0,
                variable_pay=0, gross_pay=50000, pf_employee=1800, esi_employee=0, professional_tax=200, tds=1000,
                other_deductions=0, total_deductions=3000, net_pay=47000, department_id="D01", role_id="R01",
                job_level=2, overtime_eligible="N", variable_pay_type="None")
    base.update(over)
    return base
