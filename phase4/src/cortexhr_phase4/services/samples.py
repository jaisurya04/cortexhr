"""Real example inputs taken from the project's own data (used by the smoke test, the console defaults and the tests)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..core.settings import PHASE2, PHASE3


def payroll_row(csv_dir: Path | None = None, i: int = 0) -> dict:
    d = pd.read_csv(Path(csv_dir or PHASE2 / "03_Data" / "csv") / "payroll_monthly_features.csv").iloc[i]
    keys = ["payroll_item_id", "run_id", "employee_id", "month", "department_id", "role_id", "job_level", "overtime_eligible", "monthly_fixed_gross", "payable_days",
            "lop_days", "overtime_hours_paid", "overtime_pay", "variable_pay", "variable_pay_type", "gross_pay", "pf_employee", "esi_employee", "professional_tax", "tds",
            "other_deductions", "total_deductions", "net_pay"]
    row = {k: (d[k].item() if hasattr(d[k], "item") else d[k]) for k in keys if pd.notna(d[k])}
    row["month"] = str(row["month"])[:7]
    return row


def attendance_row(csv_dir: Path | None = None, i: int = 0) -> dict:
    d = pd.read_csv(Path(csv_dir or PHASE2 / "03_Data" / "csv") / "attendance_monthly_features.csv").iloc[i]
    keys = ["employee_id", "month", "scheduled_days", "present_days", "wfh_days", "late_arrivals", "avg_hours_worked", "overtime_hours"]
    row = {k: (d[k].item() if hasattr(d[k], "item") else d[k]) for k in keys}
    row["month"] = str(row["month"])[:7]
    return row


def payband_row(csv_dir: Path | None = None, i: int = 0) -> dict:
    d = pd.read_csv(Path(csv_dir or PHASE3 / "03_Data" / "csv") / "salary_training.csv").iloc[i]
    keys = ["department_id", "role_id", "job_level", "band_mid_2026", "market_index", "prior_experience_years", "skills_count", "work_location", "highest_education"]
    return {k: (d[k].item() if hasattr(d[k], "item") else d[k]) for k in keys}


def employee_id(csv_dir: Path | None = None, i: int = 0) -> str:
    return str(pd.read_csv(Path(csv_dir or PHASE3 / "03_Data" / "csv") / "attrition_scoring_2026.csv").employee_id.iloc[i])
