"""Load and validate the headcount history and workforce events (read-only, from the Phase 3 data copy)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

HISTORY_FILE, EVENTS_FILE, SAMPLE_FILE = "monthly_headcount_history.csv", "workforce_events.csv", "sample_forecast.csv"
SAMPLE_COLUMNS = ["forecast_id", "metric", "department_id", "forecast_month", "forecast_value", "lower_80", "upper_80",
                  "method", "trained_through", "model_version"]
MIN_SERIES_HEADCOUNT = 5          # D00 (Executive Office, 1 person) is not forecast, exactly as in sample_forecast


class DataError(ValueError):
    """Raised when the source tables do not satisfy the expected contract."""


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_history(csv_dir: Path) -> pd.DataFrame:
    """Monthly history for the company total (``ALL``) and every department, validated."""
    df = pd.read_csv(Path(csv_dir) / HISTORY_FILE, parse_dates=["month"])
    need = {"month", "department_id", "headcount_end", "hires", "exits", "transfers_in", "transfers_out", "headcount_start"}
    if need - set(df.columns):
        raise DataError(f"history is missing columns: {sorted(need - set(df.columns))}")
    if df.duplicated(["month", "department_id"]).any():
        raise DataError("duplicate (month, department_id) rows in history")
    if df[list(need - {"month", "department_id"})].isna().any().any():
        raise DataError("missing values in history counts")
    ident = df.headcount_start + df.hires - df.exits + df.transfers_in - df.transfers_out - df.headcount_end
    if ident.abs().max() != 0:
        raise DataError("headcount identity (start + hires - exits + transfers_in - transfers_out = end) is violated")
    return df.sort_values(["department_id", "month"]).reset_index(drop=True)


def load_events(csv_dir: Path) -> pd.DataFrame:
    ev = pd.read_csv(Path(csv_dir) / EVENTS_FILE, parse_dates=["start_date", "end_date"])
    if (ev.end_date < ev.start_date).any():
        raise DataError("an event ends before it starts")
    return ev


def series_ids(history: pd.DataFrame) -> list[str]:
    """``ALL`` first, then every department whose peak headcount reaches :data:`MIN_SERIES_HEADCOUNT`."""
    peak = history.groupby("department_id").headcount_end.max()
    ids = [d for d in sorted(peak.index) if d != "ALL" and peak[d] >= MIN_SERIES_HEADCOUNT]
    return ["ALL"] + ids


def get_series(history: pd.DataFrame, department_id: str, column: str) -> pd.Series:
    """One monthly series with a regular month-start index (gaps would raise)."""
    s = history[history.department_id == department_id].set_index("month")[column].astype(float).sort_index()
    s.index = pd.DatetimeIndex(s.index, freq="MS")
    return s


def attrition_rate_pct(exits: pd.Series, headcount_end: pd.Series) -> pd.Series:
    """Monthly attrition rate used by the company: exits / headcount at the start of the month x 100."""
    start = headcount_end.shift(1)
    return (exits / start * 100).dropna()
