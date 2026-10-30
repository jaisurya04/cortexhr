"""Single source of truth for seeds, column roles and paths."""
from pathlib import Path
import os

SEED = 42
TARGET = "left_within_12m"
N_SPLITS = 5
TEST_SIZE = 0.20

ID_COLS = ["row_id", "employee_id"]
DATE_COLS = ["observation_date"]
# Post-outcome columns: only known AFTER the employee leaves (confirmed in 01 audit).
LEAKAGE_CONFIRMED = [
    "exit_date", "resignation_date", "exit_reason",
    "exit_interview_done", "fnf_settlement_amount",
    "voluntary_left_within_12m",          # derived from the same exit event as the target
]
# Dataset-origin metadata: not an employee attribute, differs by label rate -> excluded.
METADATA_EXCLUDED = ["source"]
FORBIDDEN_AS_INPUT = [TARGET] + ID_COLS + DATE_COLS + LEAKAGE_CONFIRMED + METADATA_EXCLUDED

CATEGORICAL_CORE = ["department_id", "role_id", "work_location", "work_mode",
                    "employment_type", "highest_education"]
NUMERIC_CORE = [
    "job_level", "tenure_months", "total_experience_years", "previous_employers_count",
    "commute_distance_km", "annual_fixed_pay", "comp_ratio", "raises_received",
    "last_raise_pct", "months_since_last_raise", "promotions_received",
    "months_since_last_promotion", "internal_transfers", "last_performance_rating",
    "last_engagement_score", "manager_changes_12m", "training_hours_12m",
    "attendance_rate_3m", "late_arrivals_3m", "avg_overtime_hours_per_month_3m",
    "avg_hours_worked_3m", "wfh_share_3m", "unpaid_leave_days_6m",
    "paid_leave_days_6m", "dept_attrition_rate_prior_12m",
]
# Engineered inside the pipeline (names produced by features.Derive)
DERIVED = ["pay_vs_dept_median", "raise_stagnation_ratio", "never_promoted"]
# Engineered from attendance_monthly_features / payroll_monthly_features (cutoff-safe)
MONTHLY_ATT = ["att_attendance_rate_mean_3m", "att_unpaid_absence_sum_3m", "att_late_arrivals_sum_3m",
               "att_overtime_mean_3m", "att_avg_hours_mean_3m", "att_wfh_share_3m", "att_half_days_sum_3m"]
MONTHLY_PAY = ["pay_variable_pay_mean_3m", "pay_lop_days_sum_3m", "pay_overtime_pay_mean_3m",
               "pay_variable_share_3m", "pay_net_to_gross_3m"]
MONTHLY_ALL = MONTHLY_ATT + MONTHLY_PAY

SHEET_ATTRITION = "attrition_dataset"
SHEET_ATT = "attendance_monthly_features"
SHEET_PAY = "payroll_monthly_features"
SHEET_BENCH = "baseline_benchmarks"
SHEET_ANOM_BENCH = "anomaly_baselines"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_path() -> Path:
    env = os.environ.get("CORTEXHR_DATA")
    return Path(env) if env else project_root() / "03_Data" / "CortexHR_Dataset.xlsx"


def reports_dir() -> Path:
    d = project_root() / "reports"; (d / "figures").mkdir(parents=True, exist_ok=True)
    (d / "tables").mkdir(parents=True, exist_ok=True); return d
