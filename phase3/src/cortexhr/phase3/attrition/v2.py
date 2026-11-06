"""Task 13: attrition model v2. Reuses the Phase 1 pipeline, split and folds; compares models on the SAME folds."""
from __future__ import annotations
import hashlib, importlib, platform, sys
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from cortexhr import config as C
from cortexhr import data_loading as D
from cortexhr import features as F
from cortexhr import modeling as M
from cortexhr import runs as R
from ..common.config import load_config
from ..common.io import read_csv, utc_now
from ..common.paths import CSV, MODELS


# ------------------------------------------------------------------ data
def load_training(include_monthly: bool = True) -> dict:
    """Phase 1 frame: exact duplicates dropped, cutoff-safe monthly join, stratified 80/20 split, 5 stratified folds (seed 42)."""
    return R.load_model_frame(include_monthly=include_monthly)


def load_scoring(include_monthly: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """The 430 active employees, joined with the same cutoff-safe monthly aggregates as training."""
    sc = read_csv("attrition_scoring_2026")
    att = D.load_sheet(C.SHEET_ATT); pay = D.load_sheet(C.SHEET_PAY)
    sc = sc.copy(); sc["observation_date"] = pd.to_datetime(sc["observation_date"])
    joined, rep = D.join_monthly_features(sc, att, pay, window=3)
    cols = F.input_columns(include_monthly)
    missing = [c for c in cols if c not in joined.columns]
    if missing: raise KeyError(f"scoring data lacks model inputs: {missing}")
    return joined, joined[cols].copy(), rep


# ------------------------------------------------------------- models
def optional_libs() -> dict:
    out = {}
    for m in ("lightgbm", "xgboost", "shap"):
        try: out[m] = importlib.import_module(m).__version__
        except Exception: out[m] = None
    return out


def make_hgb(**params) -> HistGradientBoostingClassifier:
    base = dict(learning_rate=0.05, max_iter=200, max_depth=3, max_leaf_nodes=8, min_samples_leaf=40, l2_regularization=1.0,
                class_weight="balanced", early_stopping=False, random_state=C.SEED)
    base.update(params); return HistGradientBoostingClassifier(**base)


def wrap(est, include_monthly=True) -> Pipeline:
    return Pipeline([("features", F.build_pipeline(include_monthly)), ("model", est)])



def make_lightgbm(**params):
    """Build a real LightGBM candidate with the same Phase 1 preprocessing pipeline."""
    try:
        from lightgbm import LGBMClassifier
    except Exception as exc:
        raise RuntimeError("LightGBM is not available; install the optional lightgbm dependency") from exc
    base = dict(
        objective="binary", n_estimators=160, learning_rate=0.04, max_depth=3, num_leaves=7,
        min_child_samples=35, reg_lambda=2.0, class_weight="balanced",
        random_state=C.SEED, n_jobs=1, verbosity=-1,
    )
    base.update(params)
    return wrap(LGBMClassifier(**base))


def lightgbm_tuning_grid() -> list[dict]:
    """Small predeclared LightGBM search; evaluated only on the existing Task 3 training folds."""
    return [
        dict(n_estimators=120, learning_rate=0.03, num_leaves=7, max_depth=3, min_child_samples=35, reg_lambda=2.0),
        dict(n_estimators=180, learning_rate=0.04, num_leaves=7, max_depth=3, min_child_samples=50, reg_lambda=4.0),
        dict(n_estimators=160, learning_rate=0.025, num_leaves=15, max_depth=4, min_child_samples=50, reg_lambda=5.0),
    ]

def baseline_models(include_monthly=True) -> dict:
    z = M.model_zoo(include_monthly)
    return {k: z[k] for k in ("Logistic Regression (balanced)", "Random Forest (300, balanced)")} if "Random Forest (300, balanced)" in z else \
           {k: v for k, v in z.items() if "Logistic" in k or "Forest" in k}


# --------------------------------------------------------------- metrics
def threshold_metrics(y, p, threshold=0.5) -> dict:
    y = np.asarray(y); p = np.asarray(p); pred = (p >= threshold).astype(int); tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(precision=precision_score(y, pred, zero_division=0), recall=recall_score(y, pred), f1=f1_score(y, pred, zero_division=0),
                balanced_accuracy=balanced_accuracy_score(y, pred), tn=tn, fp=fp, fn=fn, tp=tp)


def top_fraction_metrics(y, p, frac=0.10) -> dict:
    y = np.asarray(y); k = max(1, int(len(y) * frac)); flag = np.zeros(len(y), int); flag[np.argsort(-np.asarray(p))[:k]] = 1
    return dict(precision_top10=precision_score(y, flag), recall_top10=recall_score(y, flag))


def expected_calibration_error(y, p, bins=10) -> float:
    y = np.asarray(y, float); p = np.asarray(p, float); q = pd.qcut(p, bins, duplicates="drop"); df = pd.DataFrame({"y": y, "p": p, "q": q})
    g = df.groupby("q", observed=True).agg(n=("y", "size"), y=("y", "mean"), p=("p", "mean"))
    return float(((g["y"] - g["p"]).abs() * g["n"]).sum() / g["n"].sum())


def fold_table(model, X, y, folds, name, extra=None) -> pd.DataFrame:
    """Per-fold metrics. The model is cloned and refit inside each fold (preprocessing included)."""
    rows = []
    for i, (tr, va) in enumerate(folds):
        m = clone(model).fit(X.iloc[tr], y.iloc[tr]); p = m.predict_proba(X.iloc[va])[:, 1]; yv = y.iloc[va]
        rows.append({"model": name, "fold": i, "roc_auc": roc_auc_score(yv, p), "pr_auc": average_precision_score(yv, p), "brier": brier_score_loss(yv, p),
                     **threshold_metrics(yv, p), **top_fraction_metrics(yv, p)})
    return pd.DataFrame(rows)


def summarise(ft: pd.DataFrame) -> pd.DataFrame:
    cols = ["roc_auc", "pr_auc", "precision", "recall", "f1", "balanced_accuracy", "brier", "precision_top10", "recall_top10"]
    g = ft.groupby("model")[cols]; mean, sd = g.mean().add_suffix("_mean"), g.std(ddof=1).add_suffix("_sd")
    return pd.concat([mean, sd], axis=1).reset_index()


def paired_difference(ft: pd.DataFrame, a: str, b: str, metric="roc_auc", n_boot=2000, seed=C.SEED) -> dict:
    """Mean fold difference a - b and a bootstrap interval (folds are few, so the interval is wide on purpose)."""
    da = ft[ft.model == a].sort_values("fold")[metric].to_numpy(); db = ft[ft.model == b].sort_values("fold")[metric].to_numpy(); d = da - db
    rng = np.random.default_rng(seed); boots = [rng.choice(d, len(d), replace=True).mean() for _ in range(n_boot)]
    return {"metric": metric, "a": a, "b": b, "mean_diff": float(d.mean()), "sd_diff": float(d.std(ddof=1)), "boot_ci_low": float(np.percentile(boots, 2.5)),
            "boot_ci_high": float(np.percentile(boots, 97.5)), "folds_a_better": int((d > 0).sum()), "n_folds": int(len(d))}


def bootstrap_oof_auc_diff(y, p_a, p_b, n_boot=1000, seed=C.SEED) -> dict:
    """Bootstrap the AUC difference on out-of-fold scores (rows resampled, same rows for both models)."""
    y = np.asarray(y); p_a = np.asarray(p_a); p_b = np.asarray(p_b); rng = np.random.default_rng(seed); n = len(y); ds = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max(): continue
        ds.append(roc_auc_score(y[i], p_a[i]) - roc_auc_score(y[i], p_b[i]))
    ds = np.array(ds); return {"mean_diff": float(ds.mean()), "ci_low": float(np.percentile(ds, 2.5)), "ci_high": float(np.percentile(ds, 97.5)), "p_a_better": float((ds > 0).mean())}


# --------------------------------------------------------------- tuning
def tuning_grid() -> list[dict]:
    """A deliberately small, controlled search (16 settings): shallow trees, strong regularisation."""
    out = []
    for lr, it in ((0.03, 250), (0.05, 150), (0.08, 100)):
        for depth, leaves in ((2, 4), (3, 8)):
            out.append(dict(learning_rate=lr, max_iter=it, max_depth=depth, max_leaf_nodes=leaves, min_samples_leaf=40, l2_regularization=1.0))
    out += [dict(learning_rate=0.05, max_iter=150, max_depth=3, max_leaf_nodes=8, min_samples_leaf=msl, l2_regularization=l2)
            for msl, l2 in ((20, 1.0), (80, 1.0), (40, 5.0), (40, 0.0))]
    out += [dict(learning_rate=0.05, max_iter=150, max_depth=4, max_leaf_nodes=16, min_samples_leaf=60, l2_regularization=2.0),
            dict(learning_rate=0.1, max_iter=60, max_depth=3, max_leaf_nodes=8, min_samples_leaf=40, l2_regularization=1.0)]
    return out


def run_tuning(X, y, folds, include_monthly=True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (experiment log, per-fold table). Selection uses mean CV PR-AUC then ROC-AUC; the test set is never touched."""
    log, folds_all = [], []
    for i, params in enumerate(tuning_grid()):
        name = f"HGB-{i:02d}"; ft = fold_table(wrap(make_hgb(**params), include_monthly), X, y, folds, name); folds_all.append(ft)
        log.append({"experiment_id": name, "model": "HistGradientBoosting", "params": str(params), "validation": "5-fold stratified (Task 3 folds)",
                    "roc_auc_mean": ft.roc_auc.mean(), "roc_auc_sd": ft.roc_auc.std(ddof=1), "pr_auc_mean": ft.pr_auc.mean(), "pr_auc_sd": ft.pr_auc.std(ddof=1),
                    "brier_mean": ft.brier.mean(), "recall_mean": ft.recall.mean(), "precision_mean": ft.precision.mean()})
    lg = pd.DataFrame(log); best = lg.sort_values(["pr_auc_mean", "roc_auc_mean"], ascending=False).iloc[0]["experiment_id"]
    lg["decision"] = np.where(lg.experiment_id == best, "selected (highest mean CV PR-AUC)", "not selected"); 
    lg["reason"] = np.where(lg.experiment_id == best, "best mean PR-AUC; ROC-AUC reported alongside", "lower mean PR-AUC than selected setting")
    return lg, pd.concat(folds_all, ignore_index=True)


def best_params(log: pd.DataFrame) -> dict:
    import ast
    return ast.literal_eval(log.loc[log.decision.str.startswith("selected"), "params"].iloc[0])


# ---------------------------------------------------------- calibration
def calibrated_oof(model, X, y, folds, method: str) -> np.ndarray:
    """Out-of-fold probabilities. Calibration is learned inside each training fold (cv=3), never on the validation fold or test set."""
    if method == "none":
        return cross_val_predict(clone(model), X, y, cv=folds, method="predict_proba")[:, 1]
    cal = CalibratedClassifierCV(clone(model), method=method, cv=3, ensemble=False)
    return cross_val_predict(cal, X, y, cv=folds, method="predict_proba")[:, 1]


# ---------------------------------------------------------- artifact
def file_sha256(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def environment() -> dict:
    import sklearn, pandas, numpy, scipy
    return {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "pandas": pandas.__version__, "numpy": numpy.__version__,
            "scipy": scipy.__version__, "optional_libraries": optional_libs()}


# ------------------------------------------------------------ v2 candidate
from sklearn.base import BaseEstimator, ClassifierMixin


class BlendClassifier(ClassifierMixin, BaseEstimator):
    """Equal-weight average of the predicted probabilities of several full pipelines (no weight search -> no weight overfitting)."""

    def __init__(self, estimators=None):
        self.estimators = estimators

    def fit(self, X, y):
        self.models_ = [clone(e).fit(X, y) for e in self.estimators]
        self.classes_ = np.array([0, 1]); return self

    def predict_proba(self, X):
        p = np.mean([m.predict_proba(X)[:, 1] for m in self.models_], axis=0); return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def make_v2(params: dict, include_monthly=True) -> BlendClassifier:
    lr = baseline_models(include_monthly)["Logistic Regression (balanced)"]
    return BlendClassifier([lr, wrap(make_hgb(**params), include_monthly)])


def choose_simplest_within(summary: pd.DataFrame, order: list[str], metric="pr_auc_mean", tol=0.005) -> str:
    """Prefer the simplest candidate whose mean CV metric is within `tol` of the best (one-standard-error-style rule)."""
    s = summary.set_index("model")[metric]; best = s.max()
    for name in order:
        if name in s.index and best - s[name] <= tol: return name
    return s.idxmax()


def verdict(paired: dict) -> dict:
    """Pre-declared rule: the gain is 'meaningful' only if the bootstrap interval of the mean fold difference is entirely above 0."""
    meaningful = paired["boot_ci_low"] > 0
    return {"gain_is_meaningful": bool(meaningful), "mean_diff": paired["mean_diff"], "ci": [paired["boot_ci_low"], paired["boot_ci_high"]],
            "text": ("Gain is real: the 95% interval of the paired fold difference is above 0." if meaningful else
                     "Gain is NOT proven: the 95% interval of the paired fold difference includes 0 (or is below it), so the simpler baseline is not beaten.")}
