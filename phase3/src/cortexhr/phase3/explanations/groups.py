"""Explanation 'players': the 43 raw model inputs folded into 24 HR-readable groups.
Related columns (e.g. attendance_rate_3m and att_attendance_rate_mean_3m) share ONE player, so credit is not split between near-duplicates."""
from __future__ import annotations

GROUPS: dict[str, list[str]] = {
    "tenure": ["tenure_months"],
    "career_history": ["total_experience_years", "previous_employers_count"],
    "job_level": ["job_level"],
    "commute": ["commute_distance_km"],
    "pay_level": ["annual_fixed_pay", "comp_ratio"],
    "raises": ["raises_received", "last_raise_pct", "months_since_last_raise"],
    "promotion": ["promotions_received", "months_since_last_promotion", "internal_transfers"],
    "performance": ["last_performance_rating"],
    "engagement": ["last_engagement_score"],
    "manager_changes": ["manager_changes_12m"],
    "training": ["training_hours_12m"],
    "attendance": ["attendance_rate_3m", "att_attendance_rate_mean_3m", "att_unpaid_absence_sum_3m", "unpaid_leave_days_6m", "att_half_days_sum_3m"],
    "punctuality": ["late_arrivals_3m", "att_late_arrivals_sum_3m"],
    "hours_and_overtime": ["avg_overtime_hours_per_month_3m", "att_overtime_mean_3m", "avg_hours_worked_3m", "att_avg_hours_mean_3m", "pay_overtime_pay_mean_3m"],
    "remote_work": ["wfh_share_3m", "att_wfh_share_3m", "work_mode"],
    "paid_leave": ["paid_leave_days_6m"],
    "team_context": ["dept_attrition_rate_prior_12m", "department_id"],
    "role": ["role_id"],
    "work_location": ["work_location"],
    "employment_type": ["employment_type"],
    "education": ["highest_education"],
    "variable_pay": ["pay_variable_pay_mean_3m", "pay_variable_share_3m"],
    "pay_deductions": ["pay_lop_days_sum_3m", "pay_net_to_gross_3m"],
}
# Groups whose wording must never reveal or imply a protected characteristic (age, background, place of origin).
SENSITIVE_PROXY = {"career_history", "education", "work_location"}


def check_groups(columns) -> None:
    flat = [c for cols in GROUPS.values() for c in cols]
    dup = {c for c in flat if flat.count(c) > 1}
    missing, extra = sorted(set(columns) - set(flat)), sorted(set(flat) - set(columns))
    if dup or missing or extra:
        raise ValueError(f"group definition mismatch: duplicated={sorted(dup)} missing={missing} unknown={extra}")
