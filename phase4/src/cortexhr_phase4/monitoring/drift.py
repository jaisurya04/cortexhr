"""Feature drift between the training reference and the current scoring population, using the project's ``psi()`` helper."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..core import legacy
from . import status

REFERENCE_FILE, CURRENT_FILE = "attrition_dataset.csv", "attrition_scoring_2026.csv"
N_FEATURES = 8


def _helper_psi():
    legacy.bootstrap()
    from cortexhr_helpers import psi  # Phase 1-3 starter helper (read-only)
    return psi


def select_features(importance: pd.DataFrame, reference: pd.DataFrame, current: pd.DataFrame, k: int = N_FEATURES) -> list[str]:
    """The ``k`` most important NUMERIC features (Phase 3 permutation importance, largest AUC drop first) present in both tables."""
    both = set(reference.columns) & set(current.columns)
    out = []
    for f in importance.sort_values("auc_drop", ascending=False).feature:
        if f in both and pd.api.types.is_numeric_dtype(reference[f]) and reference[f].nunique() > 2:
            out.append(f)
        if len(out) == k:
            break
    if len(out) < k:
        raise ValueError(f"only {len(out)} numeric features shared by both tables; need {k}")
    return out


def feature_drift(reference: pd.DataFrame, current: pd.DataFrame, features: list[str], importance: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per feature: PSI, status, means, missing rates."""
    psi = _helper_psi()
    imp = importance.set_index("feature").auc_drop.to_dict() if importance is not None else {}
    rows = []
    for f in features:
        r, c = reference[f].dropna(), current[f].dropna()
        if len(r) < 30 or len(c) < 30:
            value = float("nan")
        else:
            value = float(psi(r.to_numpy(float), c.to_numpy(float)))
        rows.append({"feature": f, "importance_auc_drop": imp.get(f, np.nan), "psi": value, "status": status.psi_status(value),
                     "reference_mean": float(r.mean()), "current_mean": float(c.mean()),
                     "shift_pct_of_reference_sd": float((c.mean() - r.mean()) / (r.std() or 1) * 100),
                     "reference_missing_pct": float(reference[f].isna().mean() * 100), "current_missing_pct": float(current[f].isna().mean() * 100),
                     "n_reference": int(len(r)), "n_current": int(len(c)),
                     "current_top_value": float(c.mode().iloc[0]), "current_top_value_share_pct": float((c == c.mode().iloc[0]).mean() * 100)})
    return pd.DataFrame(rows)


def run_drift(data_dir: Path, importance_csv: Path) -> dict:
    """Load the two attrition tables and score drift on the 8 most important numeric features."""
    ref, cur = pd.read_csv(Path(data_dir) / REFERENCE_FILE), pd.read_csv(Path(data_dir) / CURRENT_FILE)
    imp = pd.read_csv(importance_csv)
    feats = select_features(imp, ref, cur)
    table = feature_drift(ref, cur, feats, imp)
    return {"status": status.worst(table.status), "table": table, "reference": REFERENCE_FILE, "current": CURRENT_FILE,
            "n_reference": int(len(ref)), "n_current": int(len(cur)), "features": feats,
            "note": "PSI compares the distribution of each feature in the training data with the 2026 scoring population. "
                    "It shows that inputs moved, not that the model is wrong."}
