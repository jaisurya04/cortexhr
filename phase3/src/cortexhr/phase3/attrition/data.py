"""Training and scoring frames for attrition v2. Reuses the Phase 1 loading, cleaning, de-duplication and cutoff-safe monthly join."""
from __future__ import annotations
import numpy as np
import pandas as pd
from ... import config as C, data_loading as D, features as F, runs
from ..common.io import read_csv
from ..common.validation import require_columns, require_unique

EXPECTED_SCORING_ROWS = 430


def load_training_frame() -> dict:
    """Phase 1/2 frame: stratified 80/20 split (seed 42) and the SAME 5 folds on the training part as Task 3."""
    return runs.load_model_frame(include_monthly=True)


def load_scoring_frame() -> dict:
    """attrition_scoring_2026 -> same input columns as training (monthly aggregates joined with the same cutoff rule)."""
    s = read_csv("attrition_scoring_2026")
    s["observation_date"] = pd.to_datetime(s["observation_date"])      # CSV stores dates as text; the Phase 1 join needs datetimes
    require_columns(s, ["employee_id", "observation_date"] + C.CATEGORICAL_CORE + C.NUMERIC_CORE, "attrition_scoring_2026")
    require_unique(s, "employee_id", "attrition_scoring_2026")
    forbidden = [c for c in s.columns if c in C.LEAKAGE_CONFIRMED + [C.TARGET]]
    if forbidden: raise ValueError(f"scoring file contains post-outcome columns: {forbidden}")
    att, pay = D.load_sheet(C.SHEET_ATT), D.load_sheet(C.SHEET_PAY)
    joined, rep = D.join_monthly_features(s, att, pay, window=3)
    X = joined[F.input_columns(True)].copy()
    return {"raw": s, "joined": joined, "X": X, "join_report": rep, "n_rows": len(s)}


def scoring_audit(scoring: dict, training: dict) -> pd.DataFrame:
    """One-row-per-check audit table (schema, size, duplicates, missing, categories, ranges, leakage)."""
    s, X, tr = scoring["raw"], scoring["X"], training["X"]
    unseen = {c: sorted(set(X[c].dropna()) - set(tr[c].dropna())) for c in C.CATEGORICAL_CORE}
    rows = [
        ("row count equals 430", len(s) == EXPECTED_SCORING_ROWS, len(s)),
        ("employee_id unique", not s.employee_id.duplicated().any(), int(s.employee_id.duplicated().sum())),
        ("all pipeline input columns present", all(c in X.columns for c in F.input_columns(True)), len(F.input_columns(True))),
        ("no post-outcome / target column", not any(c in s.columns for c in C.LEAKAGE_CONFIRMED + [C.TARGET]), 0),
        ("single observation date", s.observation_date.nunique() == 1, str(s.observation_date.iloc[0])),
        ("no unseen categories", all(len(v) == 0 for v in unseen.values()), {k: v for k, v in unseen.items() if v}),
        ("tenure_months >= 0", bool((s.tenure_months.dropna() >= 0).all()), int((s.tenure_months < 0).sum())),
        ("attendance_rate_3m within 0..1", bool(s.attendance_rate_3m.dropna().between(0, 1).all()), int((~s.attendance_rate_3m.dropna().between(0, 1)).sum())),
        ("monthly aggregates joined", scoring["join_report"]["attendance_matched_rows"] > 0, scoring["join_report"]["attendance_matched_rows"]),
    ]
    return pd.DataFrame(rows, columns=["check", "passed", "detail"])
