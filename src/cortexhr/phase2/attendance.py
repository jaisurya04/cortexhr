"""Attendance anomaly detector for the Phase 2 API (Task 8).

Phase 1 produced attendance flags inside a notebook.  The API needs the same idea as a saved, versioned object, so
this module wraps an unsupervised IsolationForest (no labels, ever) plus a few readable rules into
``AttendanceDetector``.  The rules reuse the domain logic of ``cortexhr.attendance`` (zero attendance not explained by
paid leave, implausible hours, late-arrival spike) and add the checks the earlier API had.

Single rows cannot use "compare with this employee's own history" features, so this detector uses the raw monthly
features only (the same ones listed in ``anomaly_baselines``).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .payroll import OFFICIAL_COLUMNS

MODEL_VERSION = "attendance-iforest-rules-v2"
SOURCE_DATASET = "attendance_monthly_features"
RULE_SCORE_FLOOR = 0.90
NEUTRAL_HOURS = 8.6                  # training median of avg_hours_worked
LONG_UNPAID_STREAK_DAYS = 5          # one full working week of unpaid absence in a row
EXTREME_OVERTIME_HOURS = 80.0
FEATURES = ["scheduled_days", "present_days", "half_days_unpaid", "paid_leave_days", "unpaid_absence_days",
            "longest_unpaid_streak_days", "late_arrivals", "avg_hours_worked", "overtime_hours", "attendance_rate"]
RULE_CODES = ["NO_ATTENDANCE", "LONG_UNPAID_ABSENCE", "IMPLAUSIBLE_HOURS", "EXTREME_OVERTIME",
              "LATE_ARRIVALS_EXCEED_SCHEDULE", "FREQUENT_LATE"]
MODEL_CODE = "PATTERN_SHIFT"           # same name the sample sheet uses for the attendance model
NOT_FLAGGED = "NOT_FLAGGED"
GLOSSARY = {
    "NO_ATTENDANCE": "No presence recorded for the whole month and paid leave does not explain it.",
    "LONG_UNPAID_ABSENCE": f"Unpaid absence of {LONG_UNPAID_STREAK_DAYS} or more days in a row.",
    "IMPLAUSIBLE_HOURS": "Average hours per present day are impossible or very unlikely (over 12, or under 3).",
    "EXTREME_OVERTIME": f"Monthly overtime above {EXTREME_OVERTIME_HOURS:.0f} hours.",
    "LATE_ARRIVALS_EXCEED_SCHEDULE": "More late arrivals than scheduled working days.",
    "FREQUENT_LATE": "Late arrivals are far above what almost every other employee records in a month.",
    MODEL_CODE: "This month's attendance pattern is unusual compared with other employee-months.",
    NOT_FLAGGED: "No configured attendance rule or model flag triggered; this does not prove the month is normal.",
}


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the model matrix.  Missing optional columns are derived, never invented as 'perfect' values."""
    x = df.copy()
    sched = x["scheduled_days"].astype(float)
    present = x["present_days"].astype(float)
    if "paid_leave_days" not in x: x["paid_leave_days"] = 0.0
    if "half_days_unpaid" not in x: x["half_days_unpaid"] = 0.0
    # Sources that do not carry hours (for example the mock HR summary) get the training median, so the hours
    # features neither raise nor lower the score; the hour rules cannot fire without real hours.
    if "avg_hours_worked" not in x: x["avg_hours_worked"] = NEUTRAL_HOURS
    x["avg_hours_worked"] = pd.to_numeric(x["avg_hours_worked"], errors="coerce").fillna(NEUTRAL_HOURS)
    if "overtime_hours" not in x: x["overtime_hours"] = 0.0
    if "late_arrivals" not in x: x["late_arrivals"] = 0.0
    if "unpaid_absence_days" not in x:
        x["unpaid_absence_days"] = (sched - present - x["paid_leave_days"]).clip(lower=0)
    if "longest_unpaid_streak_days" not in x: x["longest_unpaid_streak_days"] = 0.0
    if "attendance_rate" not in x:
        x["attendance_rate"] = np.where(sched > 0, present / sched.clip(lower=1), 0.0)
    return x[FEATURES].astype(float)


@dataclass
class AttendanceDetector:
    pipeline: Pipeline
    score_lo: float
    score_hi: float
    late_spike_threshold: float
    contamination: float
    trained_rows: int
    version: str = MODEL_VERSION

    def scaled_scores(self, feats: pd.DataFrame) -> np.ndarray:
        raw = -self.pipeline.decision_function(feats[FEATURES])
        span = self.score_hi - self.score_lo
        return np.clip((raw - self.score_lo) / span, 0.0, 1.0) if span > 0 else np.zeros(len(raw))

    def model_flags(self, feats: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(feats[FEATURES]) == -1

    def save(self, path) -> Path:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path) -> "AttendanceDetector":
        obj = joblib.load(path)
        if not isinstance(obj, AttendanceDetector):
            raise TypeError(f"{path} does not contain an AttendanceDetector")
        return obj


def fit_attendance_detector(df: pd.DataFrame, contamination: float = 0.03, random_state: int = 42) -> AttendanceDetector:
    feats = prepare_features(df)
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("detector", IsolationForest(n_estimators=300, contamination=contamination,
                                                  random_state=random_state, n_jobs=-1))])
    pipe.fit(feats)
    raw = -pipe.decision_function(feats)
    return AttendanceDetector(pipeline=pipe, score_lo=float(raw.min()), score_hi=float(raw.max()),
                              late_spike_threshold=float(df["late_arrivals"].quantile(0.995)),
                              contamination=contamination, trained_rows=int(len(df)))


def _rules(raw: pd.DataFrame, feats: pd.DataFrame, det: AttendanceDetector) -> dict:
    sched, present = feats.scheduled_days, feats.present_days
    covered = feats.paid_leave_days >= 0.8 * sched
    late = feats.late_arrivals
    hours = feats.avg_hours_worked
    ot = feats.overtime_hours
    return {
        "NO_ATTENDANCE": (sched > 0) & (present == 0) & ~covered,
        "LONG_UNPAID_ABSENCE": feats.longest_unpaid_streak_days >= LONG_UNPAID_STREAK_DAYS,
        "IMPLAUSIBLE_HOURS": (hours > 12) | ((hours < 3) & (present > 0)),
        "EXTREME_OVERTIME": ot > EXTREME_OVERTIME_HOURS,
        "LATE_ARRIVALS_EXCEED_SCHEDULE": late > sched,
        "FREQUENT_LATE": late > det.late_spike_threshold,
    }


def _sentence(code: str, r) -> str:
    if code == "NO_ATTENDANCE":
        return f"No presence recorded for {r['scheduled_days']:.0f} scheduled days and paid leave does not explain it."
    if code == "LONG_UNPAID_ABSENCE":
        return f"Unpaid absence of {r['longest_unpaid_streak_days']:.0f} days in a row (a full working week or more)."
    if code == "IMPLAUSIBLE_HOURS":
        return f"Average of {r['avg_hours_worked']:.1f} hours per present day is outside a believable range (3 to 12)."
    if code == "EXTREME_OVERTIME":
        return f"{r['overtime_hours']:.0f} overtime hours in one month is above the {EXTREME_OVERTIME_HOURS:.0f}-hour review limit."
    if code == "LATE_ARRIVALS_EXCEED_SCHEDULE":
        return f"{r['late_arrivals']:.0f} late arrivals in only {r['scheduled_days']:.0f} scheduled days."
    if code == "FREQUENT_LATE":
        return f"{r['late_arrivals']:.0f} late arrivals is higher than almost every other employee-month."
    if code == MODEL_CODE:
        if r.get("_has_rules", False):
            return "The overall attendance pattern is also unusual compared with other employee-months."
        return "No single rule is broken, but this month's attendance pattern is unusual compared with other employee-months."
    return GLOSSARY[NOT_FLAGGED]


def score_attendance(df: pd.DataFrame, detector: AttendanceDetector, ids: pd.DataFrame | None = None) -> pd.DataFrame:
    """Score rows with a saved detector.  Returns internal columns; use ``to_official_format`` for the API shape."""
    base = df.reset_index(drop=True)
    feats = prepare_features(base)
    rules = {k: v.fillna(False).astype(bool).to_numpy() for k, v in _rules(base, feats, detector).items()}
    model_score = detector.scaled_scores(feats)
    model_flag = detector.model_flags(feats)
    rule_any = np.any([rules[c] for c in RULE_CODES], axis=0)
    score = np.where(rule_any, np.maximum(model_score, RULE_SCORE_FLOOR), model_score)
    flagged = rule_any | model_flag
    work = feats.assign(_has_rules=rule_any)
    codes_all, texts = [], []
    for r in range(len(base)):
        if not flagged[r]:
            codes_all.append([]); texts.append(GLOSSARY[NOT_FLAGGED]); continue
        codes = [c for c in RULE_CODES if rules[c][r]] + ([MODEL_CODE] if model_flag[r] else [])
        codes_all.append(codes)
        row = work.iloc[r]
        parts = [_sentence(c, row) for c in codes[:3]]
        if len(codes) > 3:
            parts.append(f"({len(codes) - 3} more check(s) also triggered; see reason_code.)")
        texts.append(" ".join(parts))
    out = pd.DataFrame({
        "record_id": base["record_id"] if "record_id" in base else [f"API-{i + 1}" for i in range(len(base))],
        "employee_id": base["employee_id"] if "employee_id" in base else "UNKNOWN",
        "month": base["month"] if "month" in base else pd.NaT,
        "anomaly_score": np.round(score, 6), "flagged": flagged, "rule_flagged": rule_any, "model_flagged": model_flag,
        "reason_code": ["|".join(c) if c else NOT_FLAGGED for c in codes_all], "reason_text": texts,
        "money_at_risk": np.nan, "model_version": detector.version})
    return out.sort_values(["flagged", "anomaly_score"], ascending=[False, False]).reset_index(drop=True)


def to_official_format(scored: pd.DataFrame, scored_at=None, result_prefix: str = "AN-") -> pd.DataFrame:
    s = scored.sort_values(["flagged", "anomaly_score"], ascending=[False, False]).reset_index(drop=True)
    stamp = scored_at or pd.Timestamp.now(tz="UTC").isoformat()
    period = pd.to_datetime(s.month, errors="coerce").dt.strftime("%Y-%m-%d")
    out = pd.DataFrame({
        "result_id": [f"{result_prefix}{i + 1:04d}" for i in range(len(s))], "source_dataset": SOURCE_DATASET,
        "record_id": s.record_id, "employee_id": s.employee_id, "period": period,
        "anomaly_score": s.anomaly_score.round(4), "flagged": np.where(s.flagged, "Y", "N"),
        "reason_code": s.reason_code, "reason_text": s.reason_text, "money_at_risk": s.money_at_risk,
        "model_version": s.model_version, "scored_at": stamp})
    return out[OFFICIAL_COLUMNS]
