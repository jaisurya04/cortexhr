"""Improved model: regression with AR(1) errors (statsmodels SARIMAX) on the monthly net change, with workforce events
and yearly seasonality as regressors. This is the *only* module that imports statsmodels."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .intervals import ar1_cumulative_sd, normal_interval

try:  # statsmodels is a required dependency of the project (requirements.txt); the guard only gives a clear error.
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except ImportError:  # pragma: no cover - exercised only where statsmodels is not installed
    SARIMAX = None

NAME = "events + seasonality regression with AR(1) errors (statsmodels SARIMAX)"


class ModelUnavailable(RuntimeError):
    """statsmodels is not installed."""


class ModelFitError(RuntimeError):
    """The model could not be fitted for this series."""


def available() -> bool:
    return SARIMAX is not None


def statsmodels_version() -> str | None:
    if SARIMAX is None:
        return None
    import statsmodels
    return statsmodels.__version__


def _usable(X: pd.DataFrame) -> list[str]:
    """Drop regressors that never vary inside the training window (they cannot be estimated)."""
    return [c for c in X.columns if X[c].nunique() > 1]


def _fit(endog: pd.Series, X: pd.DataFrame):
    if SARIMAX is None:
        raise ModelUnavailable("statsmodels is not installed; run: pip install -r requirements.txt")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return SARIMAX(endog, exog=X, order=(1, 0, 0), trend="c").fit(disp=False, maxiter=200)
        except Exception as exc:  # numerical failure on a tiny series
            raise ModelFitError(f"{type(exc).__name__}: {exc}") from exc


def _ar1(res) -> tuple[float, float]:
    names = list(res.params.index)
    phi = float(res.params[next(n for n in names if n.startswith("ar.L1"))])
    return phi, float(res.params["sigma2"])


def forecast_level(y: pd.Series, design: pd.DataFrame, horizon: int, floor: float | None = 0.0):
    """Headcount-type level: model the net change, accumulate it, and widen the interval with the AR(1) error recursion.

    ``design`` covers the training months and the ``horizon`` future months (events + Fourier terms).
    """
    net = y.diff().dropna()
    cols = _usable(design.loc[net.index])
    res = _fit(net, design.loc[net.index, cols])
    fut = design.iloc[len(y):len(y) + horizon][cols]
    if len(fut) != horizon:
        raise ModelFitError("design matrix does not cover the forecast horizon")
    mean = np.asarray(res.get_forecast(steps=horizon, exog=fut).predicted_mean, float)
    phi, s2 = _ar1(res)
    point = y.iloc[-1] + np.cumsum(mean)
    lo, hi = normal_interval(point, ar1_cumulative_sd(phi, s2, horizon), floor=floor)
    return point, lo, hi


def forecast_count(y: pd.Series, design: pd.DataFrame, horizon: int, floor: float | None = 0.0):
    """Count series (monthly exits): regression with AR(1) errors, interval from the model's own forecast standard error."""
    cols = _usable(design.loc[y.index])
    res = _fit(y, design.loc[y.index, cols])
    fut = design.iloc[len(y):len(y) + horizon][cols]
    if len(fut) != horizon:
        raise ModelFitError("design matrix does not cover the forecast horizon")
    fc = res.get_forecast(steps=horizon, exog=fut)
    point, se = np.asarray(fc.predicted_mean, float), np.asarray(fc.se_mean, float)
    lo, hi = normal_interval(np.maximum(point, 0) if floor is not None else point, se, floor=floor)
    return (np.maximum(point, 0) if floor is not None else point), lo, hi
