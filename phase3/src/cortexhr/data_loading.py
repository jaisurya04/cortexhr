"""Loading, cleaning and cutoff-safe monthly aggregation."""
from __future__ import annotations
import numpy as np, pandas as pd
from . import config as C

_CITY = {"bengaluru": "Bengaluru", "bangalore": "Bengaluru",
         "madras": "Chennai", "chennai": "Chennai",
         "delhi ncr": "Delhi NCR", "delhi-ncr": "Delhi NCR"}


def load_sheet(name: str, path=None) -> pd.DataFrame:
    path = path or C.data_path()
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Put CortexHR_Dataset.xlsx in 03_Data/ "
                                f"or set CORTEXHR_DATA.")
    return pd.read_excel(path, sheet_name=name)


def load_attrition(path=None) -> pd.DataFrame:
    return load_sheet(C.SHEET_ATTRITION, path)


def canonical_city(s: pd.Series) -> pd.Series:
    key = s.astype("string").str.strip().str.lower()
    return key.map(_CITY).fillna(s.astype("string").str.strip()).astype(object).where(s.notna(), np.nan)


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Stateless, label-free cleaning shared by training and scoring (no learned statistics)."""
    d = df.copy()
    if "work_location" in d:
        d["work_location"] = canonical_city(d["work_location"])
    for c in ["tenure_months", "total_experience_years"]:
        if c in d: d.loc[d[c] < 0, c] = np.nan
    if {"total_experience_years", "tenure_months"} <= set(d.columns):
        bad = d["total_experience_years"] < (d["tenure_months"] / 12 - 0.05)
        d.loc[bad, "total_experience_years"] = np.nan   # impossible: experience < tenure
    for c in ["attendance_rate_3m", "wfh_share_3m"]:
        if c in d: d[c] = d[c].clip(0, 1)
    return d


def prepare_attrition(df: pd.DataFrame):
    """Drop exact duplicate snapshots (ignoring row_id). Returns (frame, log)."""
    n0 = len(df)
    out = df.loc[~df.drop(columns=["row_id"], errors="ignore").duplicated(keep="first")].reset_index(drop=True)
    return out, {"rows_before": n0, "exact_duplicates_dropped": n0 - len(out), "rows_after": len(out)}


# ---------------- monthly aggregation ----------------
def dedupe_payroll(pay: pd.DataFrame):
    keys = [c for c in pay.columns if c != "payroll_item_id"]
    m = pay.duplicated(subset=keys, keep="first")
    return pay.loc[~m].reset_index(drop=True), int(m.sum())


def _window(monthly: pd.DataFrame, obs: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rows of `monthly` inside the `window` complete months ending on/before each observation_date."""
    m = monthly.merge(obs[["employee_id", "observation_date"]], on="employee_id", how="inner")
    month_end = m["month"] + pd.offsets.MonthEnd(0)
    last_ok = m["observation_date"].dt.to_period("M").dt.to_timestamp()
    last_ok = last_ok.where(m["observation_date"] >= m["observation_date"] + pd.offsets.MonthEnd(0),
                            last_ok - pd.offsets.MonthBegin(1))
    first_ok = last_ok - pd.DateOffset(months=window - 1)
    keep = (m["month"] <= last_ok) & (m["month"] >= first_ok) & (month_end <= m["observation_date"])
    return m.loc[keep]


def aggregate_attendance(att, obs, window=3):
    w = _window(att, obs, window)
    g = w.groupby(["employee_id", "observation_date"])
    out = pd.DataFrame({
        "att_attendance_rate_mean_3m": g["attendance_rate"].mean(),
        "att_unpaid_absence_sum_3m": g["unpaid_absence_days"].sum(),
        "att_late_arrivals_sum_3m": g["late_arrivals"].sum(),
        "att_overtime_mean_3m": g["overtime_hours"].mean(),
        "att_avg_hours_mean_3m": g["avg_hours_worked"].mean(),
        "att_half_days_sum_3m": g["half_days_unpaid"].sum(),
        "att_wfh_share_3m": g["wfh_days"].sum() / g["present_days"].sum().replace(0, np.nan),
        "att_months_covered": g.size(),
    })
    return out.reset_index()


def aggregate_payroll(pay, obs, window=3):
    pay, _ = dedupe_payroll(pay)
    w = _window(pay, obs, window)
    g = w.groupby(["employee_id", "observation_date"])
    out = pd.DataFrame({
        "pay_variable_pay_mean_3m": g["variable_pay"].mean(),
        "pay_lop_days_sum_3m": g["lop_days"].sum(),
        "pay_overtime_pay_mean_3m": g["overtime_pay"].mean(),
        "pay_variable_share_3m": g["variable_pay"].sum() / g["gross_pay"].sum().replace(0, np.nan),
        "pay_net_to_gross_3m": g["net_pay"].sum() / g["gross_pay"].sum().replace(0, np.nan),
        "pay_months_covered": g.size(),
    })
    return out.reset_index()


def join_monthly_features(attr, att, pay, window=3):
    """Left-join cutoff-safe monthly aggregates to attrition rows (validated 1:1).
    Returns (joined, report)."""
    a = aggregate_attendance(att, attr, window); p = aggregate_payroll(pay, attr, window)
    keys = ["employee_id", "observation_date"]
    for t, nm in ((a, "attendance"), (p, "payroll")):
        assert not t.duplicated(keys).any(), f"{nm} aggregate not unique per {keys}"
    n0 = len(attr)
    j = attr.merge(a, on=keys, how="left", validate="one_to_one").merge(p, on=keys, how="left", validate="one_to_one")
    assert len(j) == n0, "join changed row count"
    rep = {
        "rows_before_join": n0, "rows_after_join": len(j), "cardinality": "attrition(1) -> aggregates(1), validated one_to_one",
        "attendance_matched_rows": int(j["att_months_covered"].notna().sum()),
        "payroll_matched_rows": int(j["pay_months_covered"].notna().sum()),
        "unmatched_rows_attendance": int(j["att_months_covered"].isna().sum()),
        "unmatched_rows_payroll": int(j["pay_months_covered"].isna().sum()),
        "rows_dropped": 0, "window_months": window,
        "temporal_cutoff": "only complete months with month_end <= observation_date; last `window` such months",
    }
    return j, rep
