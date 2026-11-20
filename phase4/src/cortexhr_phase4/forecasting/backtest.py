"""Backtesting: hide the last known months, forecast them from earlier data only, and measure the error."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from . import metrics
from .intervals import COVERAGE, empirical_interval

PointFn = Callable[[pd.Series, int], np.ndarray]
MIN_TRAIN = 36               # the 3-year seasonal-mean baseline needs 36 months, so every model is scored on the same origins
MIN_ERRORS_FOR_INTERVAL = 8


def rolling_errors(y: pd.Series, point_fn: PointFn, horizon: int, min_train: int = MIN_TRAIN) -> dict[int, np.ndarray]:
    """``actual - forecast`` per horizon from every origin that has enough history and a known target (no look-ahead)."""
    errs: dict[int, list[float]] = {h: [] for h in range(1, horizon + 1)}
    v = y.to_numpy(float)
    for n in range(min_train, len(y)):
        pts = point_fn(y.iloc[:n], min(horizon, len(y) - n))
        for h, p in enumerate(pts, start=1):
            errs[h].append(v[n + h - 1] - p)
    return {h: np.asarray(e) for h, e in errs.items()}


def baseline_with_interval(point_fn: PointFn, y: pd.Series, horizon: int, min_train: int = MIN_TRAIN, coverage: float = COVERAGE):
    """Baseline point forecast plus an 80% interval calibrated on the baseline's own past errors inside ``y``."""
    point = point_fn(y, horizon)
    lo, hi = empirical_interval(point, rolling_errors(y, point_fn, horizon, min_train), coverage)
    return point, lo, hi


def first_origin_with_interval(horizon: int, min_train: int = MIN_TRAIN) -> int:
    """Smallest training length whose inner history yields >= 8 errors at the longest horizon."""
    return min_train + horizon + MIN_ERRORS_FOR_INTERVAL - 1


def evaluate_origin(y: pd.Series, n: int, horizon: int, fc_fn: Callable[[pd.Series, int, int], tuple]) -> pd.DataFrame:
    """Forecast months ``n .. n+horizon-1`` from ``y[:n]`` only; ``fc_fn(train, horizon, n)`` returns (point, lo, hi)."""
    train = y.iloc[:n]
    horizon = min(horizon, len(y) - n)
    point, lo, hi = fc_fn(train, horizon, n)
    tgt = y.iloc[n:n + horizon]
    return pd.DataFrame({"origin": train.index[-1], "h": np.arange(1, horizon + 1), "month": tgt.index, "actual": tgt.to_numpy(),
                         "forecast": point, "lower_80": lo, "upper_80": hi})


def rolling_backtest(y: pd.Series, horizon: int, fc_fn, start: int, step: int = 1) -> pd.DataFrame:
    """Evaluate many origins (``start, start+step, ...``) that still have a full ``horizon`` of known actuals."""
    frames = [evaluate_origin(y, n, horizon, fc_fn) for n in range(start, len(y) - horizon + 1, step)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def score(bt: pd.DataFrame) -> dict:
    """Accuracy summary of a backtest table."""
    return metrics.summarise(bt.actual, bt.forecast, bt.lower_80, bt.upper_80)
