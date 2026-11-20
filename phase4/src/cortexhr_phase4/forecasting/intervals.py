"""80% prediction intervals: empirical (from past backtest errors) and parametric (AR(1) accumulated error)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

COVERAGE = 0.80
Z80 = float(norm.ppf(0.5 + COVERAGE / 2))           # 1.2816


def empirical_interval(point: np.ndarray, errors_by_h: dict[int, np.ndarray], coverage: float = COVERAGE, floor: float | None = 0.0,
                       symmetric: bool = True):
    """Interval around each point forecast from past backtest errors at the same horizon.

    ``errors_by_h[h]`` holds ``actual - forecast`` for horizon ``h`` (1-based); at least 8 are needed.
    ``symmetric=True`` (default) uses the ``coverage`` quantile of ``|error|`` (split-conformal style), which stays honest when the
    forecast is biased; ``symmetric=False`` shifts by the (1-c)/2 and (1+c)/2 quantiles of the signed errors.
    """
    lo_q, hi_q = (1 - coverage) / 2, 1 - (1 - coverage) / 2
    lo, hi = [], []
    for h, p in enumerate(point, start=1):
        e = np.asarray(errors_by_h.get(h, []), float)
        if len(e) < 8:
            raise ValueError(f"only {len(e)} past errors at horizon {h}; need at least 8 to calibrate an interval")
        if symmetric:
            q = float(np.quantile(np.abs(e), coverage))
            lo.append(p - q); hi.append(p + q)
        else:
            lo.append(p + np.quantile(e, lo_q)); hi.append(p + np.quantile(e, hi_q))
    lo, hi = np.asarray(lo), np.asarray(hi)
    if floor is not None:
        lo = np.maximum(lo, floor)
    return lo, hi


def ar1_cumulative_sd(phi: float, sigma2: float, horizon: int) -> np.ndarray:
    """Std-dev of the h-step *cumulative* forecast error of an AR(1) series (sum of its next h values).

    The cumulative error is ``sum_i (sum_{j=0}^{h-i} phi^j) * eps_i`` so its variance is
    ``sigma2 * sum_{i=1}^{h} ((1 - phi^(h-i+1)) / (1 - phi))^2``.  Parameter uncertainty is ignored (documented limit).
    """
    if not -1 < phi < 1:
        raise ValueError("AR(1) coefficient must be inside (-1, 1) for a stationary net-change series")
    sd = []
    for h in range(1, horizon + 1):
        w = np.array([(1 - phi ** (h - i + 1)) / (1 - phi) if phi != 0 else 1.0 for i in range(1, h + 1)])
        sd.append(np.sqrt(sigma2 * np.sum(w ** 2)))
    return np.asarray(sd)


def normal_interval(point: np.ndarray, sd: np.ndarray, coverage: float = COVERAGE, floor: float | None = 0.0):
    z = float(norm.ppf(0.5 + coverage / 2))
    lo, hi = point - z * sd, point + z * sd
    return (np.maximum(lo, floor) if floor is not None else lo), hi


# ---------------------------------------------------------------- honest width calibration for the improved model
MIN_CALIBRATION_ROWS = 30


def fit_width_scale(actual, forecast, hi, coverage: float = COVERAGE) -> float:
    """Factor ``k`` such that ``forecast +/- k * (hi - forecast)`` would have covered ``coverage`` of the PAST actuals (split-conformal style).

    Uses the upper half-width, so a lower bound clipped at 0 does not distort it. ``k`` > 1 means the model's own interval was too narrow."""
    a, f, h = (np.asarray(x, float) for x in (actual, forecast, hi))
    half = np.maximum(h - f, 1e-9)
    return float(np.quantile(np.abs(a - f) / half, coverage))


def apply_width_scale(point, hi, k: float, floor: float | None = 0.0):
    point, hi = np.asarray(point, float), np.asarray(hi, float)
    half = np.maximum(hi - point, 0.0) * k
    lo, hi2 = point - half, point + half
    return (np.maximum(lo, floor) if floor is not None else lo), hi2


def out_of_sample_check(roll, coverage: float = COVERAGE, train_share: float = 2 / 3) -> dict | None:
    """Fit ``k`` on the EARLIER rolling origins and measure coverage on the LATER ones (data the scale never saw).

    ``roll`` has columns origin, h, actual, forecast, lower_80, upper_80. Neighbouring origins overlap, so the test rows are not independent:
    treat the result as indicative, not as a guarantee."""
    origins = sorted(roll["origin"].unique())
    cut = int(len(origins) * train_share)
    if len(roll) < MIN_CALIBRATION_ROWS or cut < 1 or cut >= len(origins):
        return None
    early, late = roll[roll["origin"].isin(origins[:cut])], roll[roll["origin"].isin(origins[cut:])]
    k = fit_width_scale(early["actual"], early["forecast"], early["upper_80"], coverage)
    lo, hi = apply_width_scale(late["forecast"], late["upper_80"], k)
    a = late["actual"].to_numpy(float)
    return {"k_from_early_origins": k, "n_test": int(len(late)), "coverage_raw": float(np.mean((a >= late["lower_80"]) & (a <= late["upper_80"]))),
            "coverage_calibrated": float(np.mean((a >= lo) & (a <= hi))), "mean_width_calibrated": float(np.mean(hi - lo))}


WINDOW_ORIGINS = 12      # chosen BEFORE looking at results: the most recent year of forecast origins (one full seasonal cycle)


def walk_forward_calibration(roll, coverage: float = COVERAGE, window_origins: int = WINDOW_ORIGINS, min_rows: int = MIN_CALIBRATION_ROWS):
    """Honest, sequential calibration (adaptive split-conformal): at every origin the width factor is fitted ONLY on errors that were already
    known at that moment (target month <= origin) and that come from the most recent ``window_origins`` origins, then applied to that origin.

    Nothing is fitted on the data it is scored on, and an early bad period stops inflating the factor once it leaves the window.
    Returns the scored rows (raw and calibrated bounds) or None when there is too little history.
    ``roll`` has columns origin, h, month, actual, forecast, lower_80, upper_80."""
    origins = sorted(roll["origin"].unique())
    out = []
    for i, o in enumerate(origins):
        known = roll[(roll["month"] <= o) & (roll["origin"].isin(origins[max(0, i - window_origins):i]))]
        if len(known) < min_rows:
            continue
        k = fit_width_scale(known["actual"], known["forecast"], known["upper_80"], coverage)
        now = roll[roll["origin"] == o].copy()
        now["lo_cal"], now["hi_cal"] = apply_width_scale(now["forecast"], now["upper_80"], k)
        now["scale_used"] = k
        out.append(now)
    return pd.concat(out, ignore_index=True) if out else None


def final_width_scale(roll, coverage: float = COVERAGE, window_origins: int = WINDOW_ORIGINS) -> float | None:
    """Factor for the real forecast: fitted on the most recent ``window_origins`` origins' resolved errors."""
    origins = sorted(roll["origin"].unique())
    recent = roll[roll["origin"].isin(origins[-window_origins:])]
    return fit_width_scale(recent["actual"], recent["forecast"], recent["upper_80"], coverage) if len(recent) >= MIN_CALIBRATION_ROWS else None


# ---------------------------------------------------------------- adopted method: ONE width factor per metric from ALL series (pooled), last 12 origins
def _scores(df):
    return (df["actual"] - df["forecast"]).abs() / np.maximum(df["upper_80"] - df["forecast"], 1e-9)


def pooled_walk_forward(roll, coverage: float = COVERAGE, window_origins: int = WINDOW_ORIGINS, min_rows: int = MIN_CALIBRATION_ROWS):
    """Pooled adaptive split-conformal, scored walk-forward. At each origin the factor of a metric is the ``coverage`` quantile of
    |actual - forecast| / half-width over ALL series, using only errors already known at that origin (target month <= origin) from the latest
    ``window_origins`` origins. Pooling gives many more errors per factor than 27 origins of one series.

    ``roll`` columns: metric, department_id, origin, h, month, actual, forecast, lower_80, upper_80. Returns the scored rows (with lo_cal, hi_cal, k) or None."""
    roll = roll.assign(_s=_scores(roll))
    origins = sorted(roll["origin"].unique())
    out = []
    for i, o in enumerate(origins):
        known = roll[roll["origin"].isin(origins[max(0, i - window_origins):i]) & (roll["month"] <= o)]
        now = roll[roll["origin"] == o]
        ks = known.groupby("metric")["_s"].agg(lambda x: float(np.quantile(x, coverage)) if len(x) >= min_rows else np.nan).rename("k").reset_index()
        m = now.merge(ks, on="metric", how="left").dropna(subset=["k"])
        if len(m):
            m["lo_cal"], m["hi_cal"] = apply_width_scale(m["forecast"], m["upper_80"], m["k"].to_numpy())
            out.append(m)
    return pd.concat(out, ignore_index=True) if out else None


def pooled_final_scale(roll, coverage: float = COVERAGE, window_origins: int = WINDOW_ORIGINS, min_rows: int = MIN_CALIBRATION_ROWS) -> dict:
    """Factor per metric for the real forecast: latest ``window_origins`` origins, all series."""
    recent = roll[roll["origin"].isin(sorted(roll["origin"].unique())[-window_origins:])]
    return {m: float(np.quantile(_scores(g), coverage)) for m, g in recent.groupby("metric") if len(g) >= min_rows}
