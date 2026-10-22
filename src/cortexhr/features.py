"""Leakage-aware, reusable feature pipeline. The same code trains and scores.

    pipe = build_pipeline(); pipe.fit(X_train, y_train); pipe.transform(X_new)

Learned statistics (imputer medians, scaler mean/std, one-hot categories, department pay medians, role pay
mid-points) are fitted on TRAINING rows only because Pipeline.fit() sees only what it is given.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from . import config as C
from .data_loading import clean_raw


def assert_safe_input_columns(cols):
    bad = sorted(set(cols) & set(C.FORBIDDEN_AS_INPUT))
    if bad: raise ValueError(f"Forbidden columns (target/id/date/leakage/metadata) in model input: {bad}")


class RawCleaner(BaseEstimator, TransformerMixin):
    """Stateless fixes: canonical city, impossible values -> NaN, rates clipped to [0, 1]. No learned statistics."""
    def fit(self, X, y=None): return self
    def transform(self, X): return clean_raw(pd.DataFrame(X))
    def get_feature_names_out(self, input_features=None): return np.asarray(input_features)


class Derive(BaseEstimator, TransformerMixin):
    """Derived features. Reference statistics are learned in fit() on training rows only.

    pay_vs_dept_median : annual pay / median pay of the same department+job level (fallback: department, then all)
    raise_stagnation_ratio : months since last raise / (tenure + 1)  - stuck pay relative to time in company
    never_promoted : 1 if promotions_received == 0
    raise_recency_bucket : months since last raise in buckets 0-6 / 6-12 / 12-18 / 18+ / no raise on record (level-up from the playbook)
    overtime_flag : 1 if average monthly overtime is above the TRAINING 75th percentile (level-up; threshold learned in fit)
    log_annual_pay : log1p(pay) after blanking pay-typos (pay > typo_factor x its role mid-point)
    """
    def __init__(self, typo_factor: float = 3.0):
        self.typo_factor = typo_factor

    def _implied_mid(self, X): return X["annual_fixed_pay"] / X["comp_ratio"]

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.role_mid_ = self._implied_mid(X).groupby(X["role_id"]).median().to_dict()
        pay = self._fix_pay(X)
        self.grp_median_ = pay.groupby([X["department_id"], X["job_level"]]).median().to_dict()
        self.dept_median_ = pay.groupby(X["department_id"]).median().to_dict()
        self.global_median_ = float(pay.median())
        ot = X["avg_overtime_hours_per_month_3m"] if "avg_overtime_hours_per_month_3m" in X else pd.Series(dtype=float)
        self.overtime_p75_ = float(ot.quantile(0.75)) if len(ot.dropna()) else np.nan
        return self

    def _fix_pay(self, X):
        pay = X["annual_fixed_pay"].astype(float).copy()
        mid = X["role_id"].map(self.role_mid_).astype(float)
        typo = (self._implied_mid(X) / mid) > self.typo_factor
        pay[typo.fillna(False)] = np.nan
        return pay

    def transform(self, X):
        X = pd.DataFrame(X).copy(); pay = self._fix_pay(X)
        ref = pd.Series([self.grp_median_.get((d, l), np.nan) for d, l in zip(X["department_id"], X["job_level"])], index=X.index)
        ref = ref.fillna(X["department_id"].map(self.dept_median_)).fillna(self.global_median_)
        X["pay_vs_dept_median"] = pay / ref
        X["log_annual_pay"] = np.log1p(pay)
        X["raise_stagnation_ratio"] = X["months_since_last_raise"] / (X["tenure_months"].clip(lower=0) + 1)
        X["never_promoted"] = (X["promotions_received"] == 0).astype(float)
        m = X["months_since_last_raise"]
        X["raise_recency_bucket"] = pd.cut(m, [-np.inf, 6, 12, 18, np.inf], labels=["0-6m", "6-12m", "12-18m", "18m+"]).astype(object).where(m.notna(), "no raise on record")
        ot = X["avg_overtime_hours_per_month_3m"] if "avg_overtime_hours_per_month_3m" in X else pd.Series(np.nan, index=X.index)
        X["overtime_flag"] = (ot > self.overtime_p75_).astype(float) if not np.isnan(self.overtime_p75_) else 0.0
        return X


NUMERIC_DERIVED = ["log_annual_pay", "pay_vs_dept_median", "raise_stagnation_ratio", "never_promoted", "overtime_flag"]
CATEGORICAL_DERIVED = ["raise_recency_bucket"]


def feature_lists(include_monthly: bool = True):
    num = [c for c in C.NUMERIC_CORE if c != "annual_fixed_pay"] + NUMERIC_DERIVED
    monthly = list(C.MONTHLY_ALL) if include_monthly else []
    return num, monthly, list(C.CATEGORICAL_CORE) + CATEGORICAL_DERIVED


def input_columns(include_monthly: bool = True):
    """Raw columns the pipeline needs (annual_fixed_pay is consumed by Derive)."""
    cols = C.NUMERIC_CORE + C.CATEGORICAL_CORE + (C.MONTHLY_ALL if include_monthly else [])
    assert_safe_input_columns(cols); return cols


def build_pipeline(include_monthly: bool = True) -> Pipeline:
    num, monthly, cat = feature_lists(include_monthly)
    blocks = [("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                                ("scale", StandardScaler())]), num)]
    if monthly:
        blocks.append(("monthly", Pipeline([("impute", SimpleImputer(strategy="median")),
                                            ("scale", StandardScaler())]), monthly))
    blocks.append(("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
                                    ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=15,
                                                             sparse_output=False))]), cat))
    prep = ColumnTransformer(blocks, remainder="drop", verbose_feature_names_out=False)
    prep.set_output(transform="pandas")          # clean/derive already return DataFrames
    return Pipeline([("clean", RawCleaner()), ("derive", Derive()), ("prep", prep)])


def make_xy(df: pd.DataFrame, include_monthly: bool = True):
    """X holds only allowed raw columns; y is the target. The target can never be in X."""
    cols = input_columns(include_monthly)
    missing = [c for c in cols if c not in df.columns]
    if missing: raise KeyError(f"Missing input columns: {missing}")
    return df[cols].copy(), df[C.TARGET].astype(int).copy()


def representative_rows(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Deterministic, representative raw rows: messy values, missing values, clean master row."""
    picks = []
    for mask in [(df.tenure_months < 0), df.work_location.isin(["BENGALURU", "Madras", "delhi ncr"]) & df.last_engagement_score.isna(),
                 df.get("att_attendance_rate_mean_3m", pd.Series(False, index=df.index)).notna() & df.last_engagement_score.notna()]:
        idx = df.index[mask]
        if len(idx): picks.append(idx[0])
    for i in df.index:
        if len(picks) >= n: break
        if i not in picks: picks.append(i)
    return df.loc[picks[:n]]


FEATURE_DOCS = [
    ("pay_vs_dept_median", "derived", "annual pay / median pay of same department+job level; medians fitted on training rows only", "snapshot at observation_date"),
    ("raise_stagnation_ratio", "derived", "months_since_last_raise / (tenure_months + 1)", "snapshot at observation_date"),
    ("never_promoted", "derived", "1 if promotions_received == 0 (explains the blank months_since_last_promotion)", "snapshot at observation_date"),
    ("raise_recency_bucket", "derived (level-up)", "months_since_last_raise in buckets 0-6, 6-12, 12-18, 18+ months, or 'no raise on record'; one-hot encoded", "snapshot at observation_date"),
    ("overtime_flag", "derived (level-up)", "1 if avg_overtime_hours_per_month_3m is above the 75th percentile of the TRAINING rows", "snapshot at observation_date; threshold fitted on training only"),
    ("log_annual_pay", "derived", "log1p(annual_fixed_pay) after blanking pay-typos (>3x role mid-point)", "snapshot at observation_date"),
    ("att_attendance_rate_mean_3m", "attendance_monthly_features", "mean attendance_rate over last 3 complete months", "months with month_end <= observation_date"),
    ("att_unpaid_absence_sum_3m", "attendance_monthly_features", "sum unpaid_absence_days over last 3 months", "same"),
    ("att_late_arrivals_sum_3m", "attendance_monthly_features", "sum late_arrivals over last 3 months", "same"),
    ("att_overtime_mean_3m", "attendance_monthly_features", "mean overtime_hours over last 3 months", "same"),
    ("att_avg_hours_mean_3m", "attendance_monthly_features", "mean avg_hours_worked over last 3 months", "same"),
    ("att_wfh_share_3m", "attendance_monthly_features", "sum wfh_days / sum present_days over last 3 months", "same"),
    ("att_half_days_sum_3m", "attendance_monthly_features", "sum half_days_unpaid over last 3 months", "same"),
    ("pay_variable_pay_mean_3m", "payroll_monthly_features", "mean variable_pay over last 3 months (exact duplicate rows removed first)", "same"),
    ("pay_lop_days_sum_3m", "payroll_monthly_features", "sum loss-of-pay days over last 3 months", "same"),
    ("pay_overtime_pay_mean_3m", "payroll_monthly_features", "mean overtime_pay over last 3 months", "same"),
    ("pay_variable_share_3m", "payroll_monthly_features", "sum variable_pay / sum gross_pay over last 3 months", "same"),
    ("pay_net_to_gross_3m", "payroll_monthly_features", "sum net_pay / sum gross_pay over last 3 months", "same"),
]
