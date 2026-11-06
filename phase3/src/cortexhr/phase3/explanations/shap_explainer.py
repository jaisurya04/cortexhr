"""Task 14 explanations with the real `shap` library (PermutationExplainer), then folded into HR-readable groups.

`explain_with_shap` runs shap on every requested row of the 43 raw model inputs, checks the additivity property
    sum_f phi_f(x) + base_value = f(x)
and adds up the feature values inside each HR group (see groups.py). Values are in PROBABILITY units (percentage-point
contribution to the 12-month risk score), because the explained function is the calibrated pipeline's predict_proba.

`shap` is a required dependency for Task 14 (requirements.txt). If it is missing this module raises ImportError;
there is no fallback estimator, so the report can never be produced from anything other than shap.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def explain_with_shap(predict, X: pd.DataFrame, background: pd.DataFrame, n_rows=None, groups=None, seed=42, max_background=50):
    """Run shap.Explainer(..., algorithm="permutation") and optionally aggregate feature values into HR groups.

    `predict` must accept a DataFrame and return one probability per row. `n_rows=None` explains every row of X.
    Returns a dict with raw feature-level SHAP values, base values, predictions, additivity error and (when `groups`
    is given) summed group values. Raises ImportError if shap is not installed.
    """
    try:
        import shap
    except Exception as e:
        raise ImportError("Task 14 needs the real shap library: pip install -r requirements.txt") from e
    if not isinstance(X, pd.DataFrame) or not isinstance(background, pd.DataFrame):
        raise TypeError("X and background must be pandas DataFrames")
    if list(X.columns) != list(background.columns):
        raise ValueError("X and background must have identical columns in identical order")
    if X.empty or background.empty:
        raise ValueError("X and background must not be empty")
    cols = list(X.columns)
    take = len(X) if n_rows is None else max(1, min(int(n_rows), len(X)))
    sample_raw = X.head(take).copy().reset_index(drop=True)
    bg_raw = background.sample(min(len(background), max(1, int(max_background))), random_state=seed).copy().reset_index(drop=True)
    # SHAP's tabular masker expects numeric arrays. Encode categorical columns as stable category IDs, then decode every
    # masked sample before calling the real sklearn pipeline. Permutation masking swaps observed category IDs; it never
    # interpolates between categories.
    category_maps = {}
    encoded_bg = pd.DataFrame(index=bg_raw.index)
    encoded_sample = pd.DataFrame(index=sample_raw.index)
    for col in cols:
        if pd.api.types.is_numeric_dtype(bg_raw[col]) and pd.api.types.is_numeric_dtype(sample_raw[col]):
            encoded_bg[col] = pd.to_numeric(bg_raw[col], errors="coerce").astype(float)
            encoded_sample[col] = pd.to_numeric(sample_raw[col], errors="coerce").astype(float)
        else:
            vals = pd.concat([bg_raw[col], sample_raw[col]], ignore_index=True).dropna().astype(str).unique().tolist()
            mapping = {value: i for i, value in enumerate(vals)}
            category_maps[col] = mapping
            encoded_bg[col] = bg_raw[col].map(lambda v: np.nan if pd.isna(v) else mapping[str(v)]).astype(float)
            encoded_sample[col] = sample_raw[col].map(lambda v: np.nan if pd.isna(v) else mapping[str(v)]).astype(float)

    def _predict(values):
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 1: arr = arr.reshape(1, -1)
        frame = pd.DataFrame(arr, columns=cols)
        for col, mapping in category_maps.items():
            reverse = {float(code): value for value, code in mapping.items()}
            frame[col] = frame[col].map(lambda code: np.nan if pd.isna(code) else reverse.get(float(code), np.nan))
        for col in cols:
            if col not in category_maps:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
        pred = np.asarray(predict(frame), dtype=float).reshape(-1)
        if len(pred) != len(frame) or not np.isfinite(pred).all():
            raise ValueError("predict must return one finite numeric prediction per input row")
        return pred

    masker = shap.maskers.Independent(encoded_bg)
    explainer = shap.Explainer(_predict, masker, algorithm="permutation", seed=seed)
    explanation = explainer(encoded_sample, max_evals=2 * len(cols) + 1)
    values = np.asarray(explanation.values, dtype=float)
    base = np.asarray(explanation.base_values, dtype=float).reshape(-1)
    predictions = _predict(encoded_sample)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2-D SHAP matrix, got shape {values.shape}")
    raw = pd.DataFrame(values, columns=cols)
    grouped = None
    if groups is not None:
        grouped = pd.DataFrame({name: raw[list(group_cols)].sum(axis=1) for name, group_cols in groups.items()})
    return {
        "shap_version": getattr(shap, "__version__", "unknown"),
        "feature_values": raw,
        "group_values": grouped,
        "base_values": pd.Series(base, name="base_value"),
        "predictions": pd.Series(predictions, name="prediction"),
        "additivity_max_error": float(np.max(np.abs(values.sum(axis=1) + base - predictions))),
        "row_indices": sample_raw.index.to_list(),
        "background_rows": len(bg_raw),
        "max_evals": 2 * len(cols) + 1,
    }


def additivity_error(phi: pd.DataFrame, base: pd.Series, fx: pd.Series) -> float:
    """Largest |sum of contributions + base value - prediction| over all rows (should be ~1e-15)."""
    return float(np.max(np.abs(phi.sum(axis=1).to_numpy() + base.to_numpy() - fx.to_numpy())))


def top_drivers(phi: pd.DataFrame, n=3) -> pd.DataFrame:
    """Top-n groups by absolute contribution (direction kept): columns driver_1..n and phi_1..n."""
    a = phi.to_numpy(); idx = np.argsort(-np.abs(a), axis=1, kind="stable")[:, :n]; names = np.array(phi.columns)
    out = {}
    for k in range(n):
        out[f"driver_{k+1}"] = names[idx[:, k]]; out[f"phi_{k+1}"] = a[np.arange(len(a)), idx[:, k]]
    return pd.DataFrame(out)
