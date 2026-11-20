"""Forecast accuracy measures. All functions take aligned arrays of actuals and forecasts."""
from __future__ import annotations

import numpy as np


def mae(a, f) -> float:
    a, f = np.asarray(a, float), np.asarray(f, float)
    return float(np.mean(np.abs(a - f)))


def rmse(a, f) -> float:
    a, f = np.asarray(a, float), np.asarray(f, float)
    return float(np.sqrt(np.mean((a - f) ** 2)))


def mape(a, f) -> float:
    """Mean absolute percentage error in %, ignoring months where the actual is 0."""
    a, f = np.asarray(a, float), np.asarray(f, float)
    ok = a != 0
    return float(np.mean(np.abs((a[ok] - f[ok]) / a[ok])) * 100) if ok.any() else float("nan")


def bias(a, f) -> float:
    """Mean error (actual - forecast); positive = the forecast was too low."""
    return float(np.mean(np.asarray(a, float) - np.asarray(f, float)))


def coverage(a, lo, hi) -> float:
    """Share of actuals inside the interval (target 0.80 for an 80% interval)."""
    a, lo, hi = (np.asarray(x, float) for x in (a, lo, hi))
    return float(np.mean((a >= lo) & (a <= hi)))


def summarise(a, f, lo=None, hi=None) -> dict:
    out = {"n": int(len(a)), "mae": mae(a, f), "rmse": rmse(a, f), "mape_pct": mape(a, f), "bias": bias(a, f)}
    if lo is not None and hi is not None:
        out["coverage_80"] = coverage(a, lo, hi)
        out["mean_interval_width"] = float(np.mean(np.asarray(hi, float) - np.asarray(lo, float)))
    return out
