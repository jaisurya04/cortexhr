"""Shapley-value feature attribution (Monte Carlo permutation estimator, Strumbelj & Kononenko 2014) over feature GROUPS.

Why not the `shap` package here: it is not installed in the build environment and cannot be verified. This estimator is model-agnostic,
works on the full calibrated pipeline and satisfies the efficiency property exactly:
    sum_g phi_g(x) = f(x) - mean_b f(background_b used)      (checked in tests)
Values are in PROBABILITY units (percentage-point contribution to the 12-month risk score), not log-odds.
When the `shap` package is available, `shap_cross_check` runs its real PermutationExplainer on a reproducible sample and can aggregate feature attributions into the same HR-readable groups.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def shapley_values(predict, X: pd.DataFrame, background: pd.DataFrame, groups: dict[str, list[str]], n_permutations=60, seed=42, antithetic=True):
    """Return (phi DataFrame [n x groups], base_value Series, fx Series). `predict(DataFrame) -> prob array`.
    Variance reduction: every random permutation is also run in reverse order with the same background rows (antithetic pairs),
    and background rows are drawn without replacement inside each pass. Efficiency still holds exactly."""
    rng = np.random.default_rng(seed); names = list(groups); n = len(X)
    X = X.reset_index(drop=True); bg = background.reset_index(drop=True)
    phi = np.zeros((n, len(names))); base = np.zeros(n); fx = np.asarray(predict(X), float); passes = 0
    for _ in range(n_permutations):
        order = rng.permutation(len(names)); rows = rng.integers(0, len(bg), n)
        for o in ((order, order[::-1]) if antithetic else (order,)):
            cur = bg.iloc[rows].reset_index(drop=True)[X.columns].copy(); prev = np.asarray(predict(cur), float); base += prev
            for gi in o:
                for c in groups[names[gi]]:
                    cur[c] = X[c].to_numpy()
                now = np.asarray(predict(cur), float); phi[:, gi] += now - prev; prev = now
            passes += 1
    phi /= passes; base /= passes
    return pd.DataFrame(phi, columns=names), pd.Series(base, name="base_value"), pd.Series(fx, name="prediction")


def additivity_error(phi: pd.DataFrame, base: pd.Series, fx: pd.Series) -> float:
    return float(np.max(np.abs(phi.sum(axis=1).to_numpy() + base.to_numpy() - fx.to_numpy())))


def top_drivers(phi: pd.DataFrame, n=3) -> pd.DataFrame:
    """Top-n groups by absolute contribution (direction kept): columns driver_1..n and value_1..n."""
    a = phi.to_numpy(); idx = np.argsort(-np.abs(a), axis=1, kind="stable")[:, :n]; names = np.array(phi.columns)
    out = {}
    for k in range(n):
        out[f"driver_{k+1}"] = names[idx[:, k]]; out[f"phi_{k+1}"] = a[np.arange(len(a)), idx[:, k]]
    return pd.DataFrame(out)


def stability(phi_a: pd.DataFrame, phi_b: pd.DataFrame, n=3) -> dict:
    """Agreement between two independent Monte Carlo runs: mean correlation and share of employees whose top-n set is identical."""
    ta, tb = top_drivers(phi_a, n), top_drivers(phi_b, n)
    same = np.mean([set(ta.iloc[i, ::2]) == set(tb.iloc[i, ::2]) for i in range(len(ta))])
    overlap = np.mean([len(set(ta.iloc[i, ::2]) & set(tb.iloc[i, ::2])) / n for i in range(len(ta))])
    corr = float(np.corrcoef(phi_a.to_numpy().ravel(), phi_b.to_numpy().ravel())[0, 1])
    return {"phi_correlation": corr, "top_n_identical_share": float(same), "mean_top_n_overlap": float(overlap)}


def shap_cross_check(predict, X: pd.DataFrame, background: pd.DataFrame, n_rows=20, groups=None, seed=42, max_background=50):
    """Run the real SHAP PermutationExplainer and optionally aggregate feature values into HR groups.

    `predict` must accept a DataFrame and return one scalar prediction per row. The result includes
    raw feature-level SHAP values, base values, predictions, additivity error and (when `groups`
    is supplied) summed group values. Returns None only when SHAP cannot be imported.
    """
    try:
        import shap
    except Exception:
        return None
    if not isinstance(X, pd.DataFrame) or not isinstance(background, pd.DataFrame):
        raise TypeError("X and background must be pandas DataFrames")
    if list(X.columns) != list(background.columns):
        raise ValueError("X and background must have identical columns in identical order")
    if X.empty or background.empty:
        raise ValueError("X and background must not be empty")
    cols = list(X.columns)
    sample_raw = X.head(max(1, min(int(n_rows), len(X)))).copy().reset_index(drop=True)
    bg_raw = background.sample(min(len(background), max(1, int(max_background))), random_state=seed).copy().reset_index(drop=True)
    # SHAP's tabular masker expects numeric arrays. Encode categorical columns as
    # stable category IDs, then decode every masked sample before calling the real
    # sklearn pipeline. Permutation masking swaps observed category IDs; it never
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
