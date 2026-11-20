"""Which way of widening the improved model's 80% interval actually works on data it has not seen?   python scripts/interval_calibration_lab.py

Reads outputs/forecast/backtest_rolling_origin.csv (raw SARIMAX rolling forecasts for every series) and scores five calibration methods
WALK-FORWARD: at every origin the width factor uses only errors whose target month was already known at that origin. Nothing is scored on
the data it was fitted on. Writes outputs/forecast/interval_calibration_lab.csv and prints a table.

Methods: raw | series_window12 (one factor per series, last 12 origins) | series_expanding | pooled_expanding (one factor per metric from ALL series,
all earlier errors: far more data per factor) | pooled_window12.
The methods are fixed here BEFORE the results are read; whichever is adopted is chosen by the table below and the choice is written down."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cortexhr_phase4.forecasting.intervals import COVERAGE, MIN_CALIBRATION_ROWS  # noqa: E402


def score_column(df: pd.DataFrame) -> pd.Series:
    return (df["actual"] - df["forecast"]).abs() / np.maximum(df["upper_80"] - df["forecast"], 1e-9)


def walk_forward(df: pd.DataFrame, group_cols: list[str], window: int | None, min_rows: int = MIN_CALIBRATION_ROWS) -> pd.DataFrame:
    """Scored rows with a calibrated half-width factor ``k``; groups share one factor (group_cols), window = number of latest origins or None."""
    df = df.assign(_s=score_column(df))
    origins = sorted(df["origin"].unique())
    out = []
    for i, o in enumerate(origins):
        past_origins = origins[max(0, i - window):i] if window else origins[:i]
        known = df[df["origin"].isin(past_origins) & (df["month"] <= o)]
        now = df[df["origin"] == o]
        if known.empty or now.empty:
            continue
        ks = known.groupby(group_cols)["_s"].agg(lambda s: float(np.quantile(s, COVERAGE)) if len(s) >= min_rows else np.nan).rename("k").reset_index()
        merged = now.merge(ks, on=group_cols, how="left").dropna(subset=["k"])
        out.append(merged)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def coverage(df: pd.DataFrame, k: pd.Series | float = 1.0) -> tuple[float, float]:
    half = (df["upper_80"] - df["forecast"]) * k
    lo, hi = df["forecast"] - half, df["forecast"] + half
    return float(((df["actual"] >= lo) & (df["actual"] <= hi)).mean()), float((hi - lo).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="improved")
    ap.add_argument("--csv", default=str(ROOT / "outputs" / "forecast" / "backtest_rolling_origin.csv"))
    a = ap.parse_args()
    roll = pd.read_csv(a.csv, parse_dates=["origin", "month"])
    roll = roll[roll["model"] == a.model].copy()
    if roll.empty:
        raise SystemExit(f"no rows for model {a.model!r} in {a.csv}")
    series = ["metric", "department_id"]
    methods = {"series_window12": (series, 12), "series_expanding": (series, None), "pooled_expanding": (["metric"], None), "pooled_window12": (["metric"], 12)}
    rows = []
    scored = {name: walk_forward(roll, g, w) for name, (g, w) in methods.items()}
    # compare every method on exactly the rows ALL of them could score
    keys = ["metric", "department_id", "origin", "h"]
    common = roll[keys].merge(*[d[keys] for d in scored.values()][:1], on=keys)
    for d in list(scored.values())[1:]:
        common = common.merge(d[keys], on=keys)
    base = roll.merge(common, on=keys)
    for scope, sel in {"ALL series pooled": base, "headcount_end/ALL only": base[(base.metric == "headcount_end") & (base.department_id == "ALL")]}.items():
        cov, w = coverage(sel)
        rows.append({"scope": scope, "method": "raw", "rows": len(sel), "coverage_80": cov, "mean_width": w})
        for name, d in scored.items():
            d = d.merge(common, on=keys)
            d = d[(d.metric == "headcount_end") & (d.department_id == "ALL")] if "only" in scope else d
            cov, w = coverage(d, d["k"])
            rows.append({"scope": scope, "method": name, "rows": len(d), "coverage_80": cov, "mean_width": w})
    out = pd.DataFrame(rows).round(4)
    out.to_csv(ROOT / "outputs" / "forecast" / "interval_calibration_lab.csv", index=False)
    print(out.to_string(index=False))
    print("\nTarget coverage 0.80. Rows are the same for every method inside a scope; coverage is walk-forward (no look-ahead).")


if __name__ == "__main__":
    main()
