"""Driver -> plain-English sentence lookup. Wording rules: say 'predicted risk', never 'will leave'; never blame the person;
no protected-characteristic wording; sensitive-proxy groups never show a value."""
from __future__ import annotations
import math
import pandas as pd
from .groups import SENSITIVE_PROXY

UP, DOWN = "raised", "lowered"


def _n(v):  # not available?
    return v is None or (isinstance(v, float) and math.isnan(v)) or (hasattr(pd, "isna") and pd.isna(v))


def _fmt_money(v): return f"Rs {v:,.0f}"


def describe(group: str, r: pd.Series) -> str:
    """The employee's own value(s) for a group, in words (empty-safe)."""
    g = group
    if g == "tenure":
        m = int(r["tenure_months"]); return f"A short time at the company so far ({m} months)" if m < 12 else f"The length of service ({m / 12:.1f} years)"
    if g == "job_level": return f"The job level (level {int(r['job_level'])})"
    if g == "commute": return f"The commute distance ({r['commute_distance_km']:.0f} km)"
    if g == "pay_level": return f"Pay compared with the role's mid-point ({r['comp_ratio']:.2f} times the mid-point)"
    if g == "raises":
        lr = "" if _n(r["last_raise_pct"]) else f", last raise {r['last_raise_pct']:.1f}%"
        return f"The time since the last raise or joining ({int(r['months_since_last_raise'])} months{lr})"
    if g == "promotion":
        return "Having no promotion on record" if _n(r["months_since_last_promotion"]) else f"The time since the last promotion ({int(r['months_since_last_promotion'])} months)"
    if g == "performance":
        return "No recent performance rating is on record (a data gap, not a judgement)" if _n(r["last_performance_rating"]) else f"The latest performance rating ({r['last_performance_rating']:.1f} out of 5)"
    if g == "engagement":
        return "No recent engagement score is on record (a data gap, not a judgement)" if _n(r["last_engagement_score"]) else f"The latest engagement score ({r['last_engagement_score']:.1f} out of 5)"
    if g == "manager_changes": return f"The number of manager changes in the last 12 months ({int(r['manager_changes_12m'])})"
    if g == "training": return f"The training hours in the last 12 months ({r['training_hours_12m']:.0f} hours)"
    if g == "attendance": return f"The attendance pattern over the last 3 months (attendance {r['attendance_rate_3m']:.0%}, {r['att_unpaid_absence_sum_3m']:.0f} unpaid absence days)"
    if g == "punctuality": return f"The late arrivals over the last 3 months ({int(r['late_arrivals_3m'])} late arrivals)"
    if g == "hours_and_overtime": return f"The working hours and overtime pattern (about {r['avg_hours_worked_3m']:.1f} hours per day, {r['avg_overtime_hours_per_month_3m']:.1f} overtime hours per month)"
    if g == "remote_work": return f"The remote-work pattern ({r['work_mode']} mode, {r['wfh_share_3m']:.0%} of days from home)"
    if g == "paid_leave": return f"The paid leave taken in the last 6 months ({r['paid_leave_days_6m']:.0f} days)"
    if g == "team_context": return f"The department's attrition rate over the previous 12 months ({r['dept_attrition_rate_prior_12m']:.0%})"
    if g == "role": return "The role the person holds"
    if g == "employment_type": return f"The type of employment ({r['employment_type']})"
    if g == "variable_pay": return f"The variable pay over the last 3 months (average {_fmt_money(r['pay_variable_pay_mean_3m'])} per month)"
    if g == "pay_deductions": return f"The loss-of-pay days and take-home ratio over the last 3 months ({r['pay_lop_days_sum_3m']:.0f} loss-of-pay days, take-home {r['pay_net_to_gross_3m']:.0%} of gross)"
    if g in SENSITIVE_PROXY:
        return {"career_history": "Factors in the employee's career history", "education": "Factors in the employee's background", "work_location": "Factors linked to the work location"}[g]
    raise KeyError(g)


_SUFFIX = {
    "tenure": "; this reflects a common pattern at this stage of service, not the person's performance",
    "team_context": "; this describes the team, not the individual",
    "role": "; this describes common patterns in the role, not the individual",
}


def sentence(group: str, r: pd.Series, phi: float) -> str:
    verb = UP if phi > 0 else DOWN
    tail = " (a general pattern in the data, so treat it as context only)" if group in SENSITIVE_PROXY else _SUFFIX.get(group, "")
    return f"{describe(group, r)} {verb} the predicted risk{tail}."


BANNED_PHRASES = ("will leave", "going to leave", "is going to resign", "unhappy", "disloyal", "lazy", "gender", "age", "old", "young", "caste", "religion", "married", "pregnan")


def wording_problems(text: str) -> list[str]:
    import re
    t = text.lower(); return [b for b in BANNED_PHRASES if re.search(r"\b%s\b" % re.escape(b), t)]
