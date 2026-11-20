"""DESIGN-REFERENCE VALUES - NOT REAL DATA.

These are the demonstration numbers printed in the approved dashboard screenshot (12,480 employees, 8.7 % attrition, ...). They exist so
the visual design can be reproduced and compared with the reference at 1365x768. They are used ONLY by
``scripts/render_reference_preview.py`` or when ``CORTEXHR_DESIGN_REFERENCE=1`` is set explicitly. The live console and the API never
use them: real numbers always come from ``GET /v1/overview`` (see handlers.overview).

Values hidden behind the welcome popup in the screenshot (months Apr-Jun of the chart) are not visible in the reference; they continue
the visible trend and are an assumption.
"""
from __future__ import annotations

REFERENCE_NOTICE = "Design reference values - sample numbers from the approved screenshot, not real company data."

REFERENCE_OVERVIEW = {
    "reference": True,
    "kpis": {
        "total_employees": {"value": 12480, "change_pct": 2.5, "spark": [8.8, 9.0, 9.4, 9.3, 9.9, 10.3, 10.6, 10.5, 11.2, 11.0, 11.6, 12.4]},
        "attrition_risk": {"value": 8.7, "change_pct": -1.2, "spark": [5.2, 5.7, 5.4, 6.0, 5.8, 6.3, 6.2, 6.8, 6.5, 7.2, 7.6, 8.2]},
        "open_positions": {"value": 24, "change_pct": 4.3, "spark": [8.6, 8.2, 8.7, 8.1, 8.6, 8.9, 8.4, 8.8, 8.7, 8.4, 8.9, 9.8]},
        "avg_salary": {"value": 892000, "change_pct": 3.1, "spark": [8.2, 8.5, 8.4, 8.6, 8.9, 8.8, 9.0, 8.9, 9.1, 9.0, 9.2, 9.6]},
    },
    "workforce": {"series": [
        {"month": "2026-01", "label": "Jan", "employees": 6000, "exits": 3600},
        {"month": "2026-02", "label": "Feb", "employees": 6700, "exits": 3700},
        {"month": "2026-03", "label": "Mar", "employees": 7800, "exits": 2900},
        {"month": "2026-04", "label": "Apr", "employees": 7600, "exits": 3300},
        {"month": "2026-05", "label": "May", "employees": 8300, "exits": 3600},
        {"month": "2026-06", "label": "Jun", "employees": 8500, "exits": 3300},
        {"month": "2026-07", "label": "Jul", "employees": 8800, "exits": 3500},
        {"month": "2026-08", "label": "Aug", "employees": 9100, "exits": 3400},
        {"month": "2026-09", "label": "Sep", "employees": 9400, "exits": 3300}],
        "axis": {"min": 0, "max": 10000, "step": 2000}},
    "hiring": {"hires_last_6m": 1240, "change_pct": 8.2, "top_locations": [{"name": "Chennai"}, {"name": "Bengaluru"}, {"name": "Hyderabad"}], "locations": 3},
    "attrition": {"high": 12.4, "medium": 24.6, "low": 62.9, "population": 100, "mean_risk_pct": 8.7, "change_pct": -1.2,
                  "note": "Attrition risk is 1.2% lower than last period."},
    "insights": {"headcount_growth_pct_6m": 6.1, "high_risk_count": 2,
                 "lines": ["Headcount is expected to grow by 6.1% in the next 6 months.", "2 employees are at high attrition risk. Consider retention actions.",
                           "No significant bias detected in latest fairness audit."]},
    "model_performance": {"model": "Ridge Regression", "mae": 53925, "r2": 0.983, "coverage": 0.802, "baseline_mae": 115094, "mae_change_pct": -3.2, "r2_change": 0.01},
    "activity": [{"text": "Forecast report generated", "ago": "2 min ago"}, {"text": "Fairness check completed", "ago": "5 min ago"}, {"text": "New model deployed (v1.0.0)", "ago": "10 min ago"}],
    "departments": [], "notifications": [], "version": "1.0.0", "notice": REFERENCE_NOTICE,
}
REFERENCE_USER = {"name": "Jaisurya", "role": "Admin"}
