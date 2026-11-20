"""Seasonal baselines: the numbers any better model has to beat.

* ``SeasonalNaiveDrift`` (levels, e.g. headcount): "same as last year plus trend" =
  value 12 months earlier + the latest year-on-year change.
* ``SeasonalMean`` (noisy counts, e.g. exits): average of the same calendar month over the last ``years`` years.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class SeasonalNaiveDrift:
    name = "seasonal naive + yearly drift"

    def forecast(self, y: pd.Series, horizon: int) -> np.ndarray:
        if len(y) < 13:
            raise ValueError("need at least 13 months of history")
        v = y.to_numpy(float)
        drift = v[-1] - v[-13]                      # latest year-on-year change
        out = []
        for h in range(1, horizon + 1):
            ref = len(v) - 12 + (h - 1) % 12        # same calendar month, last year (for h <= 12)
            out.append(v[ref] + drift * (1 + (h - 1) // 12))
        return np.asarray(out)


class SeasonalMean:
    name = "same-month average of last 3 years"

    def __init__(self, years: int = 3) -> None:
        self.years = years

    def forecast(self, y: pd.Series, horizon: int) -> np.ndarray:
        if len(y) < 12 * self.years:
            raise ValueError(f"need at least {12 * self.years} months of history")
        n, v = len(y), y.to_numpy(float)
        out = []
        for h in range(1, horizon + 1):
            idx = [n - 12 * k + (h - 1) % 12 for k in range(1, self.years + 1)]
            out.append(float(np.mean(v[idx])))
        return np.asarray(out)
