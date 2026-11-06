"""Task 15: pay-band recommender. Job-related features only; protected attributes (gender, age) are NEVER inputs, they are used afterwards to measure gaps."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from ..common.config import load_config
from ..common.io import read_csv
from ..common.paths import VAULT

TARGET = "offered_annual_fixed"
NUMERIC_JOB = ["job_level", "log_band_mid", "log_market_index", "prior_experience_years", "skills_count"]
CATEGORICAL_JOB = ["department_id", "role_id", "work_location", "highest_education"]
JOB_FEATURES = NUMERIC_JOB + CATEGORICAL_JOB                   # chosen model: job-related only
WITH_GAP_FEATURES = JOB_FEATURES + ["career_gap_months"]       # comparison model (shows what copying history does)
NEVER_INPUTS = ["hire_id", "hire_date", "source", "gender", "age_years", "age_band", "offered_annual_fixed"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy(); d["log_band_mid"] = np.log(d["band_mid_2026"]); d["log_market_index"] = np.log(d["market_index"]); return d


def load_salary() -> pd.DataFrame:
    d = add_features(read_csv("salary_training")); d["pay_vs_band"] = d[TARGET] / (d["band_mid_2026"] * d["market_index"]); return d


def vault_gender() -> pd.DataFrame | None:
    p = VAULT / "unlock_CX-15_protected_attributes.csv"
    if not p.exists(): return None
    v = pd.read_csv(p); return v[v.record_type == "salary_hire"].drop_duplicates("person_ref").set_index("person_ref")[["gender", "age_years", "age_band"]]


def split(df: pd.DataFrame, seed: int | None = None):
    seed = seed or load_config()["seed"]
    tr, te = train_test_split(df, test_size=0.2, random_state=seed); return tr.reset_index(drop=True), te.reset_index(drop=True)


def folds(n: int, seed: int | None = None):
    return KFold(load_config()["salary"]["n_splits"], shuffle=True, random_state=seed or load_config()["seed"]).split(np.arange(n))


def preprocessor(features: list[str], scale=True) -> ColumnTransformer:
    num = [c for c in features if c not in CATEGORICAL_JOB]; cat = [c for c in features if c in CATEGORICAL_JOB]
    return ColumnTransformer([("num", StandardScaler() if scale else "passthrough", num), ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=False), cat)])


def ridge(features, alpha=3.0) -> Pipeline: return Pipeline([("prep", preprocessor(features)), ("model", Ridge(alpha=alpha))])


def hgb(features, **p) -> Pipeline:
    base = dict(learning_rate=0.05, max_iter=300, max_depth=3, min_samples_leaf=20, l2_regularization=1.0, random_state=load_config()["seed"]); base.update(p)
    return Pipeline([("prep", preprocessor(features, scale=False)), ("model", HistGradientBoostingRegressor(**base))])


class RoleMeanBaseline:
    """The task's simple baseline: average offered salary for the role (falls back to the overall mean for unseen roles)."""
    def __init__(self, adjust_market=False): self.adjust_market = adjust_market
    def fit(self, X, y):
        d = X[["role_id"]].copy(); v = np.asarray(y, float); d["v"] = v / X["market_index"].to_numpy() if self.adjust_market else v
        self.means_ = d.groupby("role_id")["v"].mean(); self.global_ = float(d["v"].mean()); return self
    def predict(self, X):
        m = X["role_id"].map(self.means_).fillna(self.global_).to_numpy(); return m * X["market_index"].to_numpy() if self.adjust_market else m


def metrics(y, pred) -> dict:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    return {"MAE_rupees": mean_absolute_error(y, pred), "RMSE_rupees": float(np.sqrt(mean_squared_error(y, pred))), "R2": r2_score(y, pred),
            "median_abs_pct_error": float(np.median(np.abs(y - pred) / y) * 100), "MAE_log": mean_absolute_error(np.log(y), np.log(np.clip(pred, 1, None)))}


def oof_predict(make, features, df: pd.DataFrame, log_target=True) -> np.ndarray:
    """Out-of-fold predictions on `df` (5 folds). Models on the log target are exponentiated back to rupees."""
    out = np.zeros(len(df)); y = df[TARGET].to_numpy(float)
    for tr, va in folds(len(df)):
        m = make(); Xtr = df.iloc[tr][features] if features else df.iloc[tr]; Xva = df.iloc[va][features] if features else df.iloc[va]
        m.fit(Xtr, np.log(y[tr]) if log_target else y[tr]); p = m.predict(Xva); out[va] = np.exp(p) if log_target else p
    return out


def band_quantiles(y, oof_pred) -> dict:
    """Spread of the log error out-of-fold -> low/high multipliers for the recommended band."""
    q = load_config()["salary"]["band_quantiles"]; r = np.log(np.asarray(y, float) / np.asarray(oof_pred, float))
    return {"low": float(np.quantile(r, q[0])), "mid": 0.0, "high": float(np.quantile(r, q[2]))}


def recommend_band(model, df: pd.DataFrame, mult: dict, features=JOB_FEATURES) -> pd.DataFrame:
    """Low / mid / high band (rupees per year). Always advice for a manager and HR, never a final offer."""
    d = add_features(df) if "log_band_mid" not in df else df
    mid = np.exp(model.predict(d[features]))
    return pd.DataFrame({"recommended_low": np.round(mid * np.exp(mult["low"]), -2), "recommended_mid": np.round(mid, -2), "recommended_high": np.round(mid * np.exp(mult["high"]), -2)}, index=df.index)
