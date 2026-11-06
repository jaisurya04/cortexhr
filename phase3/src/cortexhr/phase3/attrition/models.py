"""Attrition v2: candidate models, same-fold evaluation, controlled tuning, calibration, blending, versioned saving."""
from __future__ import annotations
import hashlib, importlib, platform, sys
import numpy as np
import pandas as pd
import sklearn
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
import joblib
from ... import features as F, modeling as M, config as C
from ..common.config import load_config
from ..common.io import save_json, utc_now
from ..common.paths import MODELS, METADATA

SEED = C.SEED


def have(pkg: str) -> bool:
    try: importlib.import_module(pkg); return True
    except Exception: return False


def top_fraction_flags(p, frac=0.10):
    p = np.asarray(p); k = max(1, int(round(len(p) * frac))); f = np.zeros(len(p), int); f[np.argsort(-p, kind="stable")[:k]] = 1; return f


def expected_calibration_error(y, p, bins=10) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float); edges = np.quantile(p, np.linspace(0, 1, bins + 1)); edges[-1] += 1e-9
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1); e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any(): e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def metric_row(y, p, threshold=0.5, top_frac=0.10) -> dict:
    y = np.asarray(y); p = np.asarray(p); yh = (p >= threshold).astype(int); fl = top_fraction_flags(p, top_frac)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p), "precision@0.5": precision_score(y, yh, zero_division=0),
            "recall@0.5": recall_score(y, yh), "f1@0.5": f1_score(y, yh, zero_division=0), "balanced_accuracy@0.5": balanced_accuracy_score(y, yh),
            "precision@top10%": precision_score(y, fl, zero_division=0), "recall@top10%": recall_score(y, fl), "brier": brier_score_loss(y, p),
            "ece": expected_calibration_error(y, p), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


# ------------------------------------------------------------------ models
class ProbaBlend(BaseEstimator, ClassifierMixin):
    """Weighted mean of the probabilities of several fitted pipelines (a plain, explainable ensemble)."""
    def __init__(self, estimators=None, weights=None):
        self.estimators, self.weights = estimators, weights
    def fit(self, X, y):
        self.fitted_ = [clone(e).fit(X, y) for e in self.estimators]; self.classes_ = np.array([0, 1]); return self
    def predict_proba(self, X):
        w = np.asarray(self.weights if self.weights is not None else [1 / len(self.fitted_)] * len(self.fitted_), float)
        p = sum(wi * m.predict_proba(X)[:, 1] for wi, m in zip(w, self.fitted_)) / w.sum(); return np.column_stack([1 - p, p])
    def predict(self, X): return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def pipeline(est) -> Pipeline:
    return Pipeline([("features", F.build_pipeline(True)), ("model", est)])


def baseline_lr() -> Pipeline:
    return M.model_zoo(True)["Logistic Regression (balanced)"]


def hgb(**kw) -> Pipeline:
    p = dict(random_state=SEED); p.update(kw); return pipeline(HGB(**p))


def optional_boosters(params: dict | None = None) -> dict:
    """LightGBM / XGBoost candidates are added only when the package is installed (skipped otherwise, and logged)."""
    out, log = {}, []
    for name, mod in (("LightGBM", "lightgbm"), ("XGBoost", "xgboost")):
        if have(mod):
            m = importlib.import_module(mod)
            est = m.LGBMClassifier(random_state=SEED, n_estimators=200, learning_rate=0.05, max_depth=3, verbose=-1) if mod == "lightgbm" else \
                  m.XGBClassifier(random_state=SEED, n_estimators=200, learning_rate=0.05, max_depth=3, eval_metric="logloss")
            out[name + " (defaults+)"] = pipeline(est); log.append((name, "available"))
        else:
            log.append((name, "NOT installed in this environment: skipped (HistGradientBoosting used)"))
    return out, log


def calibrated(model, method: str, cv=3):
    return model if method == "none" else CalibratedClassifierCV(clone(model), method=method, cv=cv)


# -------------------------------------------------------------- evaluation
def oof_and_folds(model, X, y, folds, name="model", calibration="none"):
    """Out-of-fold P(left) with the SAME folds for every model; returns (oof_scores, per-fold metrics frame)."""
    y = np.asarray(y); oof = np.zeros(len(y)); rows = []
    for i, (tr, va) in enumerate(folds):
        m = calibrated(model, calibration).fit(X.iloc[tr], y[tr]); p = m.predict_proba(X.iloc[va])[:, 1]; oof[va] = p
        rows.append({"model": name, "calibration": calibration, "fold": i + 1, **metric_row(y[va], p)})
    return oof, pd.DataFrame(rows)


def summarise_folds(fm: pd.DataFrame, cols=("roc_auc", "pr_auc", "precision@0.5", "recall@0.5", "f1@0.5", "balanced_accuracy@0.5",
                                            "precision@top10%", "recall@top10%", "brier", "ece")) -> pd.DataFrame:
    g = fm.groupby(["model", "calibration"], sort=False)
    mean = g[list(cols)].mean().add_suffix("_mean"); sd = g[list(cols)].std(ddof=1).add_suffix("_sd")
    return pd.concat([mean, sd], axis=1).reset_index()


def paired_vs_baseline(fm: pd.DataFrame, baseline: str, metric="roc_auc") -> pd.DataFrame:
    """Per-fold paired differences against the baseline (same folds), with a paired t-test (5 folds: indicative only)."""
    b = fm[fm.model == baseline].sort_values("fold")[metric].to_numpy(); rows = []
    for name, g in fm.groupby("model", sort=False):
        if name == baseline: continue
        d = g.sort_values("fold")[metric].to_numpy() - b
        t = stats.ttest_rel(g.sort_values("fold")[metric], b) if len(d) > 1 and d.std(ddof=1) > 0 else None
        rows.append({"model": name, "metric": metric, "mean_diff": d.mean(), "sd_diff": d.std(ddof=1), "folds_better": int((d > 0).sum()), "n_folds": len(d),
                     "paired_t_p": float(t.pvalue) if t is not None else np.nan})
    return pd.DataFrame(rows)


def bootstrap_auc_diff(y, p_new, p_base, n=1000, seed=SEED) -> dict:
    rng = np.random.default_rng(seed); y = np.asarray(y); d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max(): continue
        d.append(roc_auc_score(y[i], p_new[i]) - roc_auc_score(y[i], p_base[i]))
    d = np.asarray(d); return {"mean": float(d.mean()), "ci_low": float(np.percentile(d, 2.5)), "ci_high": float(np.percentile(d, 97.5)), "n_boot": len(d)}


# ----------------------------------------------------------------- tuning
PARAM_SPACE = {"learning_rate": [0.02, 0.03, 0.05, 0.08, 0.1], "max_depth": [2, 3, 4, 5], "max_leaf_nodes": [7, 15, 31],
               "min_samples_leaf": [20, 40, 60, 100], "l2_regularization": [0.0, 1.0, 5.0, 10.0], "max_iter": [100, 200, 300],
               "max_features": [0.5, 0.8, 1.0]}


def tune_hgb(X, y, folds, n_iter=24, seed=SEED, metric="pr_auc") -> pd.DataFrame:
    """Controlled random search on the TRAINING folds only. Every trial is logged with fold metrics, mean, sd and a decision."""
    rows = []
    for i, params in enumerate(ParameterSampler(PARAM_SPACE, n_iter=n_iter, random_state=seed)):
        _, fm = oof_and_folds(hgb(**params), X, y, folds, name=f"HGB-trial-{i:02d}")
        rows.append({"experiment_id": f"T{i:02d}", "model": "HistGradientBoosting", "validation": "5-fold stratified (same folds as Task 3)", **params,
                     **{f"fold{r.fold}_{metric}": r[metric] for _, r in fm.iterrows()}, "roc_auc_mean": fm.roc_auc.mean(), "roc_auc_sd": fm.roc_auc.std(ddof=1),
                     "pr_auc_mean": fm.pr_auc.mean(), "pr_auc_sd": fm.pr_auc.std(ddof=1)})
    log = pd.DataFrame(rows).sort_values("pr_auc_mean", ascending=False).reset_index(drop=True)
    best_pr = log.pr_auc_mean.iloc[0]; best_sd = log.pr_auc_sd.iloc[0]
    # one-standard-error rule: prefer the simplest trial within 1 SD of the best (guards against picking noise)
    log["within_1sd_of_best"] = log.pr_auc_mean >= best_pr - best_sd
    log["complexity"] = log.max_iter * log.max_depth
    chosen = log[log.within_1sd_of_best].sort_values(["complexity", "pr_auc_mean"], ascending=[True, False]).index[0]
    log["decision"] = ["selected (simplest within 1 SD of best)" if i == chosen else ("candidate" if w else "rejected: worse than best by > 1 SD") for i, w in zip(log.index, log.within_1sd_of_best)]
    log["reason"] = ["Simplest configuration whose mean PR-AUC is within one SD of the best trial." if i == chosen else "" for i in log.index]
    return log


# ---------------------------------------------------------------- saving
def dependency_versions() -> dict:
    out = {"python": platform.python_version(), "scikit-learn": sklearn.__version__}
    for p in ("numpy", "pandas", "scipy", "joblib"): out[p] = importlib.import_module(p).__version__
    return out


def sha256(path) -> str:
    h = hashlib.sha256(); h.update(open(path, "rb").read()); return h.hexdigest()


def save_versioned(model, name: str, version: str, meta: dict) -> dict:
    """Never overwrites Phase 1/2 artifacts: files are written as <name>.joblib / <name>.json inside this phase's artifacts."""
    MODELS.mkdir(parents=True, exist_ok=True); METADATA.mkdir(parents=True, exist_ok=True)
    mp = MODELS / f"{name}.joblib"; joblib.dump(model, mp)
    full = {"model_version": version, "saved_at_utc": utc_now(), "seed": SEED, "dependency_versions": dependency_versions(),
            "artifact": mp.name, "artifact_sha256": sha256(mp), **meta}
    save_json(full, METADATA / f"{name}.json"); return full
