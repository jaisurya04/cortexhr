"""Reproduce the interval-calibration check: how often does the baseline's '80%' interval really cover the rolling-origin actuals?

Compares signed-quantile intervals with symmetric absolute-error (split-conformal style) intervals. Writes
outputs/forecast/interval_calibration_study.csv. Run:  python scripts/interval_calibration_study.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cortexhr_phase4.core.settings import load_settings  # noqa: E402
from cortexhr_phase4.forecasting import backtest as bt, data  # noqa: E402
from cortexhr_phase4.forecasting.baseline import SeasonalMean, SeasonalNaiveDrift  # noqa: E402
from cortexhr_phase4.forecasting.intervals import empirical_interval  # noqa: E402


def main() -> None:
    s = load_settings(); hist = data.load_history(s.data_dir); rows = []
    jobs = [("headcount_end", d, SeasonalNaiveDrift()) for d in data.series_ids(hist)] + [("exits", "ALL", SeasonalMean(3))]
    for metric, dept, model in jobs:
        y = data.get_series(hist, dept, metric)
        for label, sym in (("signed quantiles", False), ("symmetric absolute error", True)):
            def fn(tr, hz, n, sym=sym, model=model):
                p = model.forecast(tr, hz)
                lo, hi = empirical_interval(p, bt.rolling_errors(tr, model.forecast, hz), symmetric=sym)
                return p, lo, hi
            r = bt.rolling_backtest(y, 6, fn, bt.first_origin_with_interval(6), 1)
            rows.append({"metric": metric, "department_id": dept, "interval_method": label, **bt.score(r)})
    out = pd.DataFrame(rows); s.forecast_dir.mkdir(parents=True, exist_ok=True)
    out.round(4).to_csv(s.forecast_dir / "interval_calibration_study.csv", index=False)
    print(out.groupby(["metric", "interval_method"])[["coverage_80", "mean_interval_width"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
