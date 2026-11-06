"""Reusable data validators. Every function raises ValueError with a readable message."""
from __future__ import annotations
import numpy as np
import pandas as pd


def require_columns(df: pd.DataFrame, cols, name="frame") -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")


def require_unique(df: pd.DataFrame, col: str, name="frame") -> None:
    dup = df[col][df[col].duplicated()].unique().tolist()
    if dup:
        raise ValueError(f"{name}: duplicate values in {col}: {dup[:5]}")


def require_subset(values, universe, what="ids") -> None:
    bad = sorted(set(values) - set(universe))
    if bad:
        raise ValueError(f"{what} not found in reference: {bad[:5]} (n={len(bad)})")


def require_probability(p, name="score") -> None:
    a = np.asarray(p, float)
    if np.isnan(a).any() or (a < 0).any() or (a > 1).any():
        raise ValueError(f"{name}: values must be in [0, 1] and not NaN")


def profile(df: pd.DataFrame) -> pd.DataFrame:
    """One-line-per-column data quality profile."""
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str), "missing": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(2), "unique": df.nunique(dropna=True),
    })
