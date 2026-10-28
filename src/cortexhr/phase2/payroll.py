"""Payroll anomaly detection for CortexHR Phase 2 (Tasks 6 and 7).

Design in one paragraph
-----------------------
Rules first, model second.  Finance teams trust a rule they can read, so every
hard payroll fact (duplicates, negative or zero pay, arithmetic that does not
add up, overtime above the policy cap, PF missing) is a transparent rule.  An
IsolationForest on engineered ratios sits behind the rules to catch "odd
combinations" no rule describes.  The model is unsupervised: no label column
is ever passed to it.  Every flag gets a reason code (same vocabulary as
``sample_anomaly_results``), a plain sentence a clerk can read, and an estimate
of the rupees at risk.

Things worth knowing
--------------------
* ``expected_gross`` = fixed gross * payable_days / days_in_month + overtime
  pay + variable pay.  Checked on the supplied data: only 35 of 7,355 rows
  deviate by more than Rs 1, so the formula is sound.
* ``money_at_risk`` is an estimate of "how much looks wrong" used to rank the
  review queue.  It is NOT an audited loss figure.
* The detector is saved/loaded as one object (``PayrollDetector``) that also
  carries the reference history needed for single-row API scoring (duplicate
  keys, each employee's usual PF / net ratio) and the score scaling learned on
  the training data, so one row is scored on the same scale as the batch.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_VERSION = "payroll-iforest-rules-v2"
SOURCE_DATASET = "payroll_monthly_features"
OVERTIME_CAP_HOURS = 50.0          # hr_faq FAQ-08: overtime capped at 50 hours a month
PF_PAID_SHARE_THRESHOLD = 0.8      # PF_NOT_DEDUCTED needs the employee to normally pay PF
RULE_SCORE_FLOOR = 0.90            # a rule hit never gets a lower anomaly_score than this

ID_COLS = ["payroll_item_id", "run_id", "employee_id", "month", "department_id", "role_id", "variable_pay_type"]
NUMERIC_BASE = ["monthly_fixed_gross", "payable_days", "lop_days", "overtime_hours_paid", "overtime_pay",
                "variable_pay", "gross_pay", "pf_employee", "esi_employee", "professional_tax", "tds",
                "other_deductions", "total_deductions", "net_pay"]
COMPONENT_COLS = ["pf_employee", "esi_employee", "professional_tax", "tds", "other_deductions"]
FEATURE_COLS = ["gross_to_fixed_ratio", "deduction_ratio", "net_to_gross_ratio", "variable_to_fixed_ratio",
                "lop_day_ratio", "overtime_pay_ratio", "fixed_gross_log", "gross_pay_log", "net_pay_log",
                "gross_vs_expected_ratio", "net_identity_gap_ratio", "overtime_over_cap_hours",
                "payable_days", "lop_days", "overtime_hours_paid", "overtime_pay", "variable_pay",
                "pf_employee", "esi_employee", "professional_tax", "tds", "other_deductions", "total_deductions"]

# Highest severity first.  The first code a row triggers is its "primary" reason.
RULE_CODES = ["DUPLICATE_PAYMENT", "NEGATIVE_NET_PAY", "ZERO_PAY", "GROSS_ABOVE_NORM", "GROSS_BELOW_EXPECTED",
              "NET_PAY_MISMATCH", "DEDUCTION_TOTAL_MISMATCH", "DEDUCTIONS_EXCEED_GROSS", "OVERTIME_ABOVE_CAP",
              "OVERTIME_NOT_ELIGIBLE", "PF_NOT_DEDUCTED", "NEGATIVE_COMPONENT"]
MODEL_CODE = "STATISTICAL_OUTLIER"
NOT_FLAGGED = "NOT_FLAGGED"
OFFICIAL_COLUMNS = ["result_id", "source_dataset", "record_id", "employee_id", "period", "anomaly_score",
                    "flagged", "reason_code", "reason_text", "money_at_risk", "model_version", "scored_at"]

# One short, generic sentence per code (documentation; rows get a specific sentence with their own numbers).
REASON_GLOSSARY = {
    "DUPLICATE_PAYMENT": "Two payslips exist for the same employee and payroll run.",
    "NEGATIVE_NET_PAY": "Net pay is below zero because deductions are larger than gross pay.",
    "ZERO_PAY": "Gross or net pay is zero although the employee has payable days.",
    "GROSS_ABOVE_NORM": "Gross pay is higher than fixed pay, overtime and variable pay explain.",
    "GROSS_BELOW_EXPECTED": "Gross pay is lower than fixed pay, overtime and variable pay explain.",
    "NET_PAY_MISMATCH": "Gross minus deductions does not equal the net pay shown.",
    "DEDUCTION_TOTAL_MISMATCH": "The listed deductions do not add up to the total deductions shown.",
    "DEDUCTIONS_EXCEED_GROSS": "Total deductions are larger than gross pay.",
    "OVERTIME_ABOVE_CAP": "Paid overtime hours are above the monthly policy cap.",
    "OVERTIME_NOT_ELIGIBLE": "Overtime was paid to an employee who is not overtime-eligible.",
    "PF_NOT_DEDUCTED": "No PF was deducted although this employee normally pays PF.",
    "NEGATIVE_COMPONENT": "At least one pay or deduction component is negative.",
    MODEL_CODE: "Unusual combination of pay figures compared with other payslips.",
    NOT_FLAGGED: "No issue found.",
}


# --------------------------------------------------------------------------- loading and audit
def load_payroll(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = set(ID_COLS + NUMERIC_BASE)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Payroll data missing required columns: {missing}")
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    for c in NUMERIC_BASE:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def expected_gross(df: pd.DataFrame) -> pd.Series:
    """Gross pay that fixed pay, payable days, overtime and variable pay explain."""
    month = pd.to_datetime(df["month"], errors="coerce")
    dim = month.dt.days_in_month.astype("float64")
    fallback = (df["payable_days"] + df["lop_days"]).clip(lower=1)
    dim = dim.where(dim.notna(), fallback)
    return df["monthly_fixed_gross"] * df["payable_days"] / dim + df["overtime_pay"] + df["variable_pay"]


def audit_payroll(df: pd.DataFrame) -> dict:
    eg = expected_gross(df)
    tol = np.maximum(1.0, 0.002 * eg.abs())
    return {
        "rows": int(len(df)), "columns": int(df.shape[1]),
        "duplicate_payroll_item_ids": int(df.payroll_item_id.duplicated().sum()),
        "duplicate_employee_run_pairs": int(df.duplicated(["employee_id", "run_id"]).sum()),
        "missing_cells": int(df.isna().sum().sum()),
        "negative_net_pay": int((df.net_pay < 0).sum()),
        "zero_net_pay": int((df.net_pay == 0).sum()),
        "negative_gross_pay": int((df.gross_pay < 0).sum()),
        "gross_net_identity_violations": int(((df.gross_pay - df.total_deductions - df.net_pay).abs() > 0.05).sum()),
        "gross_vs_expected_violations": int(((df.gross_pay - eg).abs() > tol).sum()),
        "overtime_hours_above_cap": int((df.overtime_hours_paid > OVERTIME_CAP_HOURS).sum()),
    }


# --------------------------------------------------------------------------- features
def engineer_payroll_features(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric ratios that make odd rows stand out.  Identifiers and labels never enter the matrix."""
    x = df.copy()
    eps = 1e-6
    eg = expected_gross(x)
    x["expected_gross"] = eg
    x["gross_to_fixed_ratio"] = x.gross_pay / x.monthly_fixed_gross.clip(lower=eps)
    x["deduction_ratio"] = x.total_deductions / x.gross_pay.abs().clip(lower=eps)
    x["net_to_gross_ratio"] = x.net_pay / x.gross_pay.abs().clip(lower=eps)
    x["variable_to_fixed_ratio"] = x.variable_pay / x.monthly_fixed_gross.clip(lower=eps)
    x["lop_day_ratio"] = x.lop_days / (x.payable_days + x.lop_days).clip(lower=eps)
    x["overtime_pay_ratio"] = x.overtime_pay / x.monthly_fixed_gross.clip(lower=eps)
    x["fixed_gross_log"] = np.log1p(x.monthly_fixed_gross.clip(lower=0))
    x["gross_pay_log"] = np.log1p(x.gross_pay.clip(lower=0))
    x["net_pay_log"] = np.log1p(x.net_pay.clip(lower=0))
    x["gross_vs_expected_ratio"] = (x.gross_pay / eg.clip(lower=eps)).clip(upper=50)
    x["net_identity_gap_ratio"] = (x.gross_pay - x.total_deductions - x.net_pay).abs() / x.gross_pay.abs().clip(lower=1.0)
    x["overtime_over_cap_hours"] = (x.overtime_hours_paid - OVERTIME_CAP_HOURS).clip(lower=0)
    x[FEATURE_COLS] = x[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
    return x


# --------------------------------------------------------------------------- reference history
@dataclass
class PayrollReference:
    """History kept with the detector so one API row can be judged against what we already know."""
    key_items: dict = field(default_factory=dict)       # (employee_id, run_id) -> tuple(payroll_item_id)
    item_keys: dict = field(default_factory=dict)       # payroll_item_id -> (employee_id, run_id)
    emp_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    global_net_ratio: float = 0.85
    global_tds_ratio: float = 0.08
    default_pf: float = 1800.0


def build_reference(df: pd.DataFrame) -> PayrollReference:
    x = df.reset_index(drop=True)
    key_items = {k: tuple(sorted(g)) for k, g in x.groupby(["employee_id", "run_id"]).payroll_item_id}
    item_keys = {i: (e, r) for i, e, r in zip(x.payroll_item_id, x.employee_id, x.run_id)}
    gross = x.gross_pay.where(x.gross_pay > 0)
    tmp = pd.DataFrame({"employee_id": x.employee_id, "net_ratio": x.net_pay / gross, "tds_ratio": x.tds / gross,
                        "pf_paid": (x.pf_employee > 0).astype(float), "pf_pos": x.pf_employee.where(x.pf_employee > 0)})
    # Medians are robust to the few broken rows that live in the same history.
    emp = tmp.groupby("employee_id").agg(net_ratio=("net_ratio", "median"), tds_ratio=("tds_ratio", "median"),
                                         pf_paid_share=("pf_paid", "mean"), pf_median=("pf_pos", "median"),
                                         rows=("pf_paid", "size"))
    g_net = float(np.nanmedian(tmp.net_ratio.clip(0, 1)))
    g_tds = float(np.nanmedian(tmp.tds_ratio.clip(0, 1)))
    pf_pos = x.pf_employee[x.pf_employee > 0]
    return PayrollReference(key_items=key_items, item_keys=item_keys, emp_stats=emp,
                            global_net_ratio=g_net if np.isfinite(g_net) else 0.85,
                            global_tds_ratio=g_tds if np.isfinite(g_tds) else 0.08,
                            default_pf=float(pf_pos.median()) if len(pf_pos) else 1800.0)


def _emp_value(ref, employees: pd.Series, column: str, default: float) -> np.ndarray:
    if ref is None or ref.emp_stats.empty or column not in ref.emp_stats.columns:
        return np.full(len(employees), default, dtype=float)
    return employees.map(ref.emp_stats[column]).astype(float).fillna(default).to_numpy()


# --------------------------------------------------------------------------- duplicates
def _duplicate_info(x: pd.DataFrame, ref):
    """Return (is_duplicate, is_later_copy, other_item_id) arrays.

    A row is a duplicate when a *different* payroll item exists for the same employee and run, either in the batch
    being scored or in the reference history (this is what makes the rule work for a single API row).  Items
    already in the history count as earlier than items that only appear in the batch.
    """
    keys = list(zip(x.employee_id, x.run_id))
    items = x.payroll_item_id.astype(str).tolist()
    batch_ids = set(items)
    batch_by_key: dict = {}
    for k, i in zip(keys, items):
        batch_by_key.setdefault(k, set()).add(i)
    n = len(x)
    dup = np.zeros(n, dtype=bool); later = np.zeros(n, dtype=bool); other = [""] * n
    for r, (k, i) in enumerate(zip(keys, items)):
        ref_ids = set(ref.key_items.get(k, ())) if ref is not None else set()
        ref_only = sorted(ref_ids - batch_ids)
        in_batch = sorted(batch_by_key[k] | (ref_ids & batch_ids))
        order = ref_only + [b for b in in_batch if b not in ref_only]
        reused = ref is not None and i in ref.item_keys and ref.item_keys[i] != k   # item id already used elsewhere
        if len(order) > 1:
            dup[r] = True
            first = order[0]
            later[r] = i != first
            other[r] = first if i != first else next(b for b in order if b != i)
        elif reused:
            dup[r] = True; later[r] = True
            other[r] = i
    return dup, later, np.array(other, dtype=object)


# --------------------------------------------------------------------------- rules
def _rule_table(x: pd.DataFrame, ref):
    """Evaluate every rule.  ``x`` must already contain engineered features and a clean 0..n-1 index."""
    eg = x.expected_gross
    tol = np.maximum(1.0, 0.002 * eg.abs())
    gross_pos = x.gross_pay > 0
    dup, later, other = _duplicate_info(x, ref)
    comp_sum = x[COMPONENT_COLS].sum(axis=1)
    pf_share = pd.Series(_emp_value(ref, x.employee_id, "pf_paid_share", 0.0), index=x.index)
    elig = x["overtime_eligible"].astype(str).str.upper() if "overtime_eligible" in x.columns else pd.Series("Y", index=x.index)
    rules = {
        "DUPLICATE_PAYMENT": pd.Series(dup, index=x.index),
        "NEGATIVE_NET_PAY": x.net_pay < 0,
        "ZERO_PAY": (x.gross_pay == 0) | (x.net_pay == 0),
        "GROSS_ABOVE_NORM": gross_pos & ((x.gross_pay - eg) > tol),
        "GROSS_BELOW_EXPECTED": gross_pos & ((eg - x.gross_pay) > tol),
        "NET_PAY_MISMATCH": gross_pos & ((x.gross_pay - x.total_deductions - x.net_pay).abs() > 1.0),
        "DEDUCTION_TOTAL_MISMATCH": gross_pos & ((comp_sum - x.total_deductions).abs() > 1.0),
        "DEDUCTIONS_EXCEED_GROSS": gross_pos & (x.total_deductions > x.gross_pay + 0.05),
        "OVERTIME_ABOVE_CAP": x.overtime_hours_paid > OVERTIME_CAP_HOURS,
        "OVERTIME_NOT_ELIGIBLE": (elig == "N") & (x.overtime_pay > 0),
        "PF_NOT_DEDUCTED": gross_pos & (x.pf_employee == 0) & (pf_share >= PF_PAID_SHARE_THRESHOLD),
        "NEGATIVE_COMPONENT": x[[c for c in NUMERIC_BASE if c != "net_pay"]].lt(0).any(axis=1),
    }
    clean = {k: v.fillna(False).astype(bool).to_numpy() for k, v in rules.items()}
    return clean, later, other


def payroll_rule_flags(df: pd.DataFrame, reference=None) -> pd.DataFrame:
    """Transparent rules.  One boolean column per rule code plus ``rule_flagged`` and ``rule_reason_codes``.

    Flags are candidates for human review, not proof of an error.
    """
    x = engineer_payroll_features(df.reset_index(drop=True))
    clean, _, _ = _rule_table(x, reference)
    out = pd.DataFrame({"payroll_item_id": x.payroll_item_id, "employee_id": x.employee_id, "run_id": x.run_id, "month": x.month})
    for code in RULE_CODES:
        out[code] = clean[code]
    out["rule_flagged"] = out[RULE_CODES].any(axis=1)
    out["rule_reason_codes"] = ["|".join(c for c in RULE_CODES if clean[c][r]) for r in range(len(out))]
    out.index = df.index
    return out


# --------------------------------------------------------------------------- the detector object
@dataclass
class PayrollDetector:
    pipeline: Pipeline
    score_lo: float
    score_hi: float
    contamination: float
    reference: PayrollReference
    feature_cols: list
    trained_rows: int
    version: str = MODEL_VERSION

    def raw_scores(self, feats: pd.DataFrame) -> np.ndarray:
        return -self.pipeline.decision_function(feats[self.feature_cols])     # higher = more unusual

    def scaled_scores(self, feats: pd.DataFrame) -> np.ndarray:
        raw = self.raw_scores(feats)
        span = self.score_hi - self.score_lo
        return np.clip((raw - self.score_lo) / span, 0.0, 1.0) if span > 0 else np.zeros(len(raw))

    def model_flags(self, feats: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(feats[self.feature_cols]) == -1

    def save(self, path) -> Path:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path) -> "PayrollDetector":
        obj = joblib.load(path)
        if not isinstance(obj, PayrollDetector):
            raise TypeError(f"{path} does not contain a PayrollDetector")
        return obj


def fit_isolation_forest(df: pd.DataFrame, contamination: float = 0.03, random_state: int = 42) -> PayrollDetector:
    """Fit the unsupervised detector on engineered features only.  There is no label argument by design."""
    x = engineer_payroll_features(df.reset_index(drop=True))
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                     ("scaler", StandardScaler()),
                     ("detector", IsolationForest(n_estimators=300, contamination=contamination,
                                                  random_state=random_state, n_jobs=-1))])
    pipe.fit(x[FEATURE_COLS])
    raw = -pipe.decision_function(x[FEATURE_COLS])
    return PayrollDetector(pipeline=pipe, score_lo=float(np.nanmin(raw)), score_hi=float(np.nanmax(raw)),
                           contamination=contamination, reference=build_reference(df),
                           feature_cols=list(FEATURE_COLS), trained_rows=int(len(x)))


# --------------------------------------------------------------------------- money at risk and reasons

# Plain-words names for the figures that can make a payslip look unusual (used for model-only flags).
_DRIVER_TEXT = {
    "gross_to_fixed_ratio": ("gross pay as a share of monthly fixed pay", "pct"),
    "deduction_ratio": ("deductions as a share of gross pay", "pct"),
    "net_to_gross_ratio": ("net pay as a share of gross pay", "pct"),
    "variable_to_fixed_ratio": ("variable pay as a share of fixed pay", "pct"),
    "lop_day_ratio": ("share of the month unpaid (loss of pay)", "pct"),
    "overtime_pay_ratio": ("overtime pay as a share of fixed pay", "pct"),
    "gross_vs_expected_ratio": ("gross pay compared with what the inputs explain", "x"),
    "net_identity_gap_ratio": ("gap between gross minus deductions and net pay", "pct"),
    "overtime_hours_paid": ("overtime hours paid", "num"), "overtime_pay": ("overtime pay", "inr"),
    "variable_pay": ("variable pay", "inr"), "tds": ("TDS deducted", "inr"), "other_deductions": ("other deductions", "inr"),
    "pf_employee": ("PF deducted", "inr"), "professional_tax": ("professional tax", "inr"), "esi_employee": ("ESI deducted", "inr"),
    "total_deductions": ("total deductions", "inr"), "payable_days": ("number of payable days", "num"), "lop_days": ("number of loss-of-pay days", "num"),
}


_DRIVER_FAMILY = {"variable_to_fixed_ratio": "variable", "variable_pay": "variable",
                  "overtime_pay_ratio": "overtime", "overtime_pay": "overtime", "overtime_hours_paid": "overtime",
                  "gross_to_fixed_ratio": "gross", "gross_vs_expected_ratio": "gross",
                  "lop_day_ratio": "days", "lop_days": "days", "payable_days": "days"}      # everything else = "deductions"


def _fmt_driver(value, kind: str) -> str:
    if kind == "pct": return f"{100 * float(value):.0f}%"
    if kind == "x": return f"{float(value):.2f}x"
    if kind == "inr": return _inr(value)
    return f"{float(value):.0f}"


def _model_drivers(model, x: pd.DataFrame, mask: np.ndarray) -> dict:
    """For model-flagged rows: the (up to) two figures furthest from the typical payslip, in plain words.

    Uses the fitted imputer/scaler (training medians, means and spreads), so it needs no labels and works for
    detectors that were saved before this function existed.
    """
    out = {}
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return out
    cols = list(model.feature_cols); n = len(cols)
    imputer, scaler = model.pipeline.named_steps["imputer"], model.pipeline.named_steps["scaler"]
    Xi = imputer.transform(x.iloc[idx][cols])[:, :n]
    scale = np.where(scaler.scale_[:n] == 0, 1.0, scaler.scale_[:n])
    z = (Xi - scaler.mean_[:n]) / scale
    median = imputer.statistics_[:n]
    usable = [i for i, c in enumerate(cols) if c in _DRIVER_TEXT]
    for pos, row_i in enumerate(idx):
        order = sorted(usable, key=lambda i: -abs(z[pos, i]))
        picked, seen = [], set()
        for i in order:                                    # at most one figure per family, so the two reasons differ
            fam = _DRIVER_FAMILY.get(cols[i], "deductions")
            if abs(z[pos, i]) >= 2.0 and fam not in seen:
                picked.append(i); seen.add(fam)
            if len(picked) == 2: break
        parts = []
        for i in picked:
            label, kind = _DRIVER_TEXT[cols[i]]
            typical = "most payslips have none" if abs(median[i]) < 1e-9 else f"typical {_fmt_driver(median[i], kind)}"
            parts.append(f"{label} is {_fmt_driver(Xi[pos, i], kind)} ({typical})")
        out[int(row_i)] = " and ".join(parts)
    return out

def _inr(v) -> str:
    try:
        return f"Rs {float(v):,.0f}"
    except (TypeError, ValueError):
        return "Rs ?"


def _money_at_risk(x: pd.DataFrame, rules: dict, later: np.ndarray, ref):
    """Return (money_at_risk, expected_net).  A prioritisation estimate, not an audit figure.

    For a rule-flagged row it is the largest exposure among the rules it triggered (for example gross minus
    expected gross for GROSS_ABOVE_NORM).  For a model-only flag it is the gap between the net pay paid and the net
    pay expected from the expected gross and this employee's usual net ratio.
    """
    eg = x.expected_gross.to_numpy(float)
    gross = x.gross_pay.to_numpy(float); net = x.net_pay.to_numpy(float); ded = x.total_deductions.to_numpy(float)
    ot_h = x.overtime_hours_paid.to_numpy(float); ot_pay = x.overtime_pay.to_numpy(float)
    net_ratio = np.clip(_emp_value(ref, x.employee_id, "net_ratio", ref.global_net_ratio if ref else 0.85), 0, 1)
    tds_ratio = np.clip(_emp_value(ref, x.employee_id, "tds_ratio", ref.global_tds_ratio if ref else 0.08), 0, 1)
    pf_med = _emp_value(ref, x.employee_id, "pf_median", ref.default_pf if ref else 1800.0)
    expected_net = eg * net_ratio
    gap_net = np.abs(net - expected_net)
    cap_excess = np.where(ot_h > OVERTIME_CAP_HOURS, (ot_h - OVERTIME_CAP_HOURS) / np.maximum(ot_h, 1.0) * ot_pay, 0.0)
    tds_excess = np.clip(x.tds.to_numpy(float) - tds_ratio * eg, 0, None)
    exposure = {
        "DUPLICATE_PAYMENT": np.where(later, np.abs(net), 0.0),
        "NEGATIVE_NET_PAY": np.maximum(np.abs(net), tds_excess),
        "ZERO_PAY": np.maximum(expected_net, 0.0),
        "GROSS_ABOVE_NORM": np.clip(gross - eg, 0, None),
        "GROSS_BELOW_EXPECTED": np.clip(eg - gross, 0, None),
        "NET_PAY_MISMATCH": np.abs(gross - ded - net),
        "DEDUCTION_TOTAL_MISMATCH": np.abs(x[COMPONENT_COLS].sum(axis=1).to_numpy(float) - ded),
        "DEDUCTIONS_EXCEED_GROSS": np.clip(ded - gross, 0, None),
        "OVERTIME_ABOVE_CAP": cap_excess,
        "OVERTIME_NOT_ELIGIBLE": ot_pay,
        "PF_NOT_DEDUCTED": pf_med,
        "NEGATIVE_COMPONENT": np.zeros(len(x)),
    }
    money = np.zeros(len(x)); any_rule = np.zeros(len(x), dtype=bool)
    for code in RULE_CODES:
        hit = rules[code]
        money = np.where(hit, np.maximum(money, exposure[code]), money)
        any_rule |= hit
    return np.round(np.where(any_rule, money, gap_net), 2), expected_net


def _sentence(code: str, r) -> str:
    if code == "DUPLICATE_PAYMENT":
        if r["_later"]:
            return (f"This payslip repeats employee {r['employee_id']}'s payment for run {r['run_id']} "
                    f"(earlier item {r['_other']}); paying it would send {_inr(abs(r['net_pay']))} a second time. Hold it and confirm.")
        return (f"A second payslip ({r['_other']}) exists for the same employee and run {r['run_id']}; "
                f"the later copy is the likely duplicate and this one is listed for comparison.")
    if code == "NEGATIVE_NET_PAY":
        return (f"Net pay is {_inr(r['net_pay'])} (below zero): deductions of {_inr(r['total_deductions'])} are larger than gross pay of "
                f"{_inr(r['gross_pay'])}. Check TDS and other deductions before paying.")
    if code == "ZERO_PAY":
        return (f"Gross or net pay is zero although {r['payable_days']:.0f} days are payable; the employee would receive nothing. "
                f"Confirm leave or payment-hold status.")
    if code == "GROSS_ABOVE_NORM":
        ratio = r["gross_pay"] / r["expected_gross"] if r["expected_gross"] > 0 else float("inf")
        return (f"Gross pay {_inr(r['gross_pay'])} is {ratio:.1f}x the {_inr(r['expected_gross'])} that fixed pay, overtime and variable pay explain. "
                f"Look for an extra digit or a wrong allowance.")
    if code == "GROSS_BELOW_EXPECTED":
        pct = 100 * (1 - r["gross_pay"] / r["expected_gross"]) if r["expected_gross"] > 0 else 100.0
        return (f"Gross pay {_inr(r['gross_pay'])} is {pct:.0f}% below the {_inr(r['expected_gross'])} that fixed pay, overtime and variable pay explain; "
                f"the employee may be underpaid.")
    if code == "NET_PAY_MISMATCH":
        calc = r["gross_pay"] - r["total_deductions"]
        return (f"Gross minus deductions is {_inr(calc)} but net pay shows {_inr(r['net_pay'])} (difference {_inr(abs(calc - r['net_pay']))}); "
                f"the payslip arithmetic does not add up.")
    if code == "DEDUCTION_TOTAL_MISMATCH":
        return f"The listed deductions add up to {_inr(r['_comp_sum'])} but total deductions show {_inr(r['total_deductions'])}."
    if code == "DEDUCTIONS_EXCEED_GROSS":
        return f"Total deductions {_inr(r['total_deductions'])} exceed gross pay {_inr(r['gross_pay'])}."
    if code == "OVERTIME_ABOVE_CAP":
        return f"{r['overtime_hours_paid']:.0f} overtime hours were paid; policy caps overtime at {OVERTIME_CAP_HOURS:.0f} hours a month."
    if code == "OVERTIME_NOT_ELIGIBLE":
        return f"Overtime pay of {_inr(r['overtime_pay'])} was paid to an employee who is not marked overtime-eligible."
    if code == "PF_NOT_DEDUCTED":
        return "No PF was deducted this month although this employee paid PF in most of their other months."
    if code == "NEGATIVE_COMPONENT":
        return "At least one pay or deduction component is negative; check the entry."
    if code == MODEL_CODE:
        if r.get("_has_rules", False):
            return "The overall pattern of pay figures is also unusual compared with other payslips."
        drivers = str(r.get("_drivers", "") or "")
        if drivers:
            return f"No rule is broken, but this payslip is unusual: its {drivers}. A reviewer should check these figures."
        return ("No rule is broken and no single figure stands out, but the combination of pay figures is unusual "
                "compared with other payslips; a reviewer should check the components.")
    return REASON_GLOSSARY[NOT_FLAGGED]


def _compose_reason(row, codes: list, max_sentences: int = 3) -> str:
    if not codes:
        return REASON_GLOSSARY[NOT_FLAGGED]
    parts = [_sentence(c, row) for c in codes[:max_sentences]]
    if len(codes) > max_sentences:
        parts.append(f"({len(codes) - max_sentences} more check(s) also triggered; see reason_code.)")
    return " ".join(parts)


# --------------------------------------------------------------------------- scoring
def score_payroll(df: pd.DataFrame, model=None, contamination: float = 0.03):
    """Score payslips.  Returns ``(scored_frame, detector)`` ranked by flagged then money at risk.

    If no detector is supplied one is fitted on ``df`` (batch use).  A supplied detector is used as-is, so a single
    row is scored on the same scale and with the same history as the batch (API use).
    """
    base = df.reset_index(drop=True)
    if model is None:
        model = fit_isolation_forest(base, contamination=contamination)
    ref = model.reference
    x = engineer_payroll_features(base)
    clean, later, other = _rule_table(x, ref)
    if_score = model.scaled_scores(x)
    if_flag = model.model_flags(x)
    rule_any = np.any([clean[c] for c in RULE_CODES], axis=0)
    score = np.where(rule_any, np.maximum(if_score, RULE_SCORE_FLOOR), if_score)
    money, expected_net = _money_at_risk(x, clean, later, ref)

    drivers = _model_drivers(model, x, if_flag & ~rule_any)
    work = x.assign(_later=later, _other=other, _comp_sum=x[COMPONENT_COLS].sum(axis=1), _has_rules=rule_any,
                    _drivers=pd.Series({i: drivers.get(i, "") for i in range(len(x))}).reindex(range(len(x))).fillna("").to_numpy())
    flagged_arr = rule_any | if_flag
    codes_per_row, texts = [], []
    for r in range(len(x)):
        if not flagged_arr[r]:
            codes_per_row.append([]); texts.append(REASON_GLOSSARY[NOT_FLAGGED]); continue
        codes = [c for c in RULE_CODES if clean[c][r]]
        if if_flag[r]:
            codes = codes + [MODEL_CODE]
        codes_per_row.append(codes)
        texts.append(_compose_reason(work.iloc[r], codes))

    out = base[["payroll_item_id", "run_id", "employee_id", "month", "gross_pay", "total_deductions", "net_pay"]].copy()
    out["expected_gross"] = x.expected_gross.round(2).to_numpy()
    out["expected_net"] = np.round(expected_net, 2)
    out["anomaly_score"] = np.round(score, 6)
    out["iforest_flagged"] = if_flag
    out["rule_flagged"] = rule_any
    for c in RULE_CODES:
        out[c] = clean[c]
    out["rule_reason_codes"] = ["|".join(c for c in RULE_CODES if clean[c][r]) for r in range(len(x))]
    out["flagged"] = flagged_arr
    out["reason_code"] = ["|".join(c) if c else NOT_FLAGGED for c in codes_per_row]
    out["reason_text"] = texts
    out["money_at_risk"] = np.where(flagged_arr, money, 0.0)
    out["model_version"] = model.version
    out = out.sort_values(["flagged", "money_at_risk", "anomaly_score", "payroll_item_id"], ascending=[False, False, False, True]).reset_index(drop=True)
    return out, model


def to_official_format(scored: pd.DataFrame, source_dataset: str = SOURCE_DATASET, scored_at=None,
                       result_prefix: str = "P2-", flagged_only: bool = False) -> pd.DataFrame:
    """Rename and order columns exactly like ``sample_anomaly_results``.

    Rows are ranked flagged-first, then by money at risk (largest first), THEN numbered, so P2-00001 is the item
    a clerk should open first.
    """
    s = scored.sort_values(["flagged", "money_at_risk", "anomaly_score", "payroll_item_id"], ascending=[False, False, False, True]).reset_index(drop=True)
    if flagged_only:
        s = s[s.flagged].reset_index(drop=True)
    stamp = scored_at or pd.Timestamp.now(tz="UTC").isoformat()
    period = pd.to_datetime(s.month, errors="coerce").dt.strftime("%Y-%m-%d")
    out = pd.DataFrame({
        "result_id": [f"{result_prefix}{i + 1:05d}" for i in range(len(s))],
        "source_dataset": source_dataset,
        "record_id": s.payroll_item_id, "employee_id": s.employee_id, "period": period,
        "anomaly_score": s.anomaly_score.round(4),
        "flagged": np.where(s.flagged, "Y", "N"),
        "reason_code": s.reason_code, "reason_text": s.reason_text,
        "money_at_risk": s.money_at_risk.round(2),
        "model_version": s.model_version, "scored_at": stamp,
    })
    return out[OFFICIAL_COLUMNS]


_MONTH_RE = re.compile(r"^(\d{4})-(\d{2})(?:-(\d{2}))?$")


def parse_month(value) -> pd.Timestamp:
    """Accept YYYY-MM or YYYY-MM-DD and return the first day of that month.  Raises ValueError otherwise."""
    m = _MONTH_RE.match(str(value).strip())
    if not m or not 1 <= int(m.group(2)) <= 12:
        raise ValueError("month must look like 2026-09 or 2026-09-01")
    return pd.Timestamp(year=int(m.group(1)), month=int(m.group(2)), day=1)


# --------------------------------------------------------------------------- evaluation against the vault labels
def evaluate_by_type(predictions: pd.DataFrame, labels: pd.DataFrame, id_col: str = "payroll_item_id",
                     type_col: str = "error_type", label_col=None,
                     normal_values=("", "none", "None", "normal", "NORMAL", "nan", "NaN")) -> pd.DataFrame:
    """Per-error-type recall and pooled precision, plus a rules-versus-model breakdown.

    Use only with the authorised CX-06 file, and only after predictions are frozen.  ``labels`` may list only the
    injected errors or every row; any scored row that is not a labelled error counts as a normal row.

    * recall_T    = flagged rows of type T / all rows of type T.
    * precision_T = TP_T / (TP_T + FP_normal).  A false alarm has no error type, so FP_normal is shared by all
                    types.  Read recall per type and compare precision with the ``overall`` row.
    * rules_recall / model_recall / caught_by_* show which detector caught each type (what the rules caught that the
      model missed).
    """
    pred = predictions.copy()
    pred[id_col] = pred[id_col].astype(str)
    cols = [c for c in dict.fromkeys([id_col, type_col, label_col]) if c]
    lab = labels[cols].copy()
    lab[id_col] = lab[id_col].astype(str)
    if label_col:
        is_err = lab[label_col].astype(str).str.lower().isin(["1", "1.0", "true", "y", "yes"])
    else:
        is_err = ~lab[type_col].astype(str).isin(normal_values)
    err = lab[is_err].drop_duplicates(id_col)
    pred["_is_error"] = pred[id_col].isin(set(err[id_col]))
    flagged = pred.flagged.astype(bool)
    rules = pred["rule_flagged"].astype(bool) if "rule_flagged" in pred else flagged
    model = pred["iforest_flagged"].astype(bool) if "iforest_flagged" in pred else flagged
    fp_normal = int((flagged & ~pred._is_error).sum())
    n_err = int(pred._is_error.sum())
    keep = [id_col, "flagged"] + [c for c in ("rule_flagged", "iforest_flagged") if c in pred]
    merged = err.merge(pred[keep], on=id_col, how="left")
    rows = []
    for typ, g in merged.groupby(type_col, dropna=False):
        hit = g.flagged.fillna(False).astype(bool)
        r_hit = g["rule_flagged"].fillna(False).astype(bool) if "rule_flagged" in g else hit
        m_hit = g["iforest_flagged"].fillna(False).astype(bool) if "iforest_flagged" in g else hit
        tp = int(hit.sum()); fn = int((~hit).sum())
        rows.append({"error_type": str(typ), "support": len(g), "tp": tp, "fn": fn, "fp_pooled": fp_normal,
                     "precision": tp / (tp + fp_normal) if tp + fp_normal else 0.0, "recall": tp / len(g) if len(g) else 0.0,
                     "rules_recall": float(r_hit.mean()), "model_recall": float(m_hit.mean()),
                     "caught_by_rules_missed_by_model": int((r_hit & ~m_hit).sum()),
                     "caught_by_model_missed_by_rules": int((m_hit & ~r_hit).sum()),
                     "not_in_predictions": int(g.flagged.isna().sum())})
    tp_all = int((flagged & pred._is_error).sum())
    rows.append({"error_type": "overall", "support": n_err, "tp": tp_all, "fn": n_err - tp_all, "fp_pooled": fp_normal,
                 "precision": tp_all / (tp_all + fp_normal) if tp_all + fp_normal else 0.0,
                 "recall": tp_all / max(n_err, 1),
                 "rules_recall": float((rules & pred._is_error).sum() / max(n_err, 1)),
                 "model_recall": float((model & pred._is_error).sum() / max(n_err, 1)),
                 "caught_by_rules_missed_by_model": int((rules & ~model & pred._is_error).sum()),
                 "caught_by_model_missed_by_rules": int((model & ~rules & pred._is_error).sum()),
                 "not_in_predictions": 0})
    return pd.DataFrame(rows)
