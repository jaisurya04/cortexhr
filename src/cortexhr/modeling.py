"""Baselines: split, folds, models, out-of-fold probabilities, SMOTE comparison."""
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from . import config as C
from .features import build_pipeline
from .evaluation import binary_report


def make_split(X, y, test_size=C.TEST_SIZE, seed=C.SEED):
    """Stratified 80/20 split with a fixed seed. The test part is used ONCE, at the very end."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def make_folds(X, y, n_splits=C.N_SPLITS, seed=C.SEED):
    """Explicit list of (train_idx, valid_idx) so EVERY model uses exactly the same folds."""
    return list(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y))


def model_zoo(include_monthly=True, seed=C.SEED) -> dict:
    mk = lambda est: Pipeline([("features", build_pipeline(include_monthly)), ("model", est)])
    return {
        "Dummy (always stays)": mk(DummyClassifier(strategy="most_frequent")),
        "Logistic Regression (balanced)": mk(LogisticRegression(class_weight="balanced", max_iter=2000, solver="lbfgs", random_state=seed)),
        "Decision Tree (depth 4, balanced)": mk(DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=seed)),
        "Random Forest (300, balanced)": mk(RandomForestClassifier(n_estimators=300, class_weight="balanced", min_samples_leaf=3, n_jobs=1, random_state=seed)),
    }


def oof_scores(model, X, y, folds) -> np.ndarray:
    """Out-of-fold P(left) for every row: each row is scored by a model that never saw it (preprocessing refit per fold)."""
    p = cross_val_predict(model, X, y, cv=folds, method="predict_proba")
    return p[:, list(model.classes_ if hasattr(model, "classes_") else [0, 1]).index(1)] if False else p[:, 1]


def fold_metrics(y, scores, folds, name, threshold=0.5) -> pd.DataFrame:
    rows = []
    for i, (_, te) in enumerate(folds, 1):
        r = binary_report(np.asarray(y)[te], np.asarray(scores)[te], threshold=threshold); r.update(model=name, fold=i); rows.append(r)
    return pd.DataFrame(rows)


def summarise(fm: pd.DataFrame) -> dict:
    out = {"model": fm.model.iloc[0]}
    for m in ["roc_auc", "pr_auc", "precision", "recall", "f1", "accuracy", "top_precision", "top_recall"]:
        out[f"{m}_mean"] = fm[m].mean()
        if m in ("roc_auc", "pr_auc"): out[f"{m}_sd"] = fm[m].std(ddof=1)
    out["folds_with_undefined_precision"] = int((~fm.precision_defined).sum()); return out


def evaluate_zoo(X, y, folds, include_monthly=True, models=None):
    zoo = model_zoo(include_monthly); zoo = {k: v for k, v in zoo.items() if models is None or k in models}
    oof, tables, summ = {}, [], []
    for name, m in zoo.items():
        s = oof_scores(m, X, y, folds); oof[name] = s
        fm = fold_metrics(y, s, folds, name); tables.append(fm); summ.append(summarise(fm))
    return oof, pd.concat(tables, ignore_index=True), pd.DataFrame(summ)


def tree_overfit_demo(X, y, folds, include_monthly=True, seed=C.SEED):
    """Unlimited-depth tree: near-perfect on training rows, much weaker on unseen rows."""
    m = Pipeline([("features", build_pipeline(include_monthly)), ("model", DecisionTreeClassifier(random_state=seed))])
    train_auc = roc_auc_score(y, m.fit(X, y).predict_proba(X)[:, 1])
    cv_auc = roc_auc_score(y, oof_scores(m, X, y, folds))
    return {"train_auc_in_sample": train_auc, "cv_auc_out_of_fold": cv_auc}


# ---------------- SMOTE (own implementation; imbalanced-learn not installable offline) ----------------
class SimpleSMOTE:
    """Synthetic minority oversampling: new points on segments between a minority row and one of its k nearest
    minority neighbours. Applied to TRAINING folds only (never to validation/test rows)."""
    def __init__(self, k_neighbors=5, random_state=C.SEED): self.k, self.rs = k_neighbors, random_state
    def fit_resample(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y).astype(int); rng = np.random.default_rng(self.rs)
        Xm = X[y == 1]; n_new = int((y == 0).sum() - (y == 1).sum())
        if n_new <= 0 or len(Xm) < 2: return X, y
        k = min(self.k, len(Xm) - 1); nn = NearestNeighbors(n_neighbors=k + 1).fit(Xm)
        neigh = nn.kneighbors(Xm, return_distance=False)[:, 1:]
        base = rng.integers(0, len(Xm), n_new); pick = neigh[base, rng.integers(0, k, n_new)]
        gap = rng.random((n_new, 1)); Xn = Xm[base] + gap * (Xm[pick] - Xm[base])
        return np.vstack([X, Xn]), np.r_[y, np.ones(n_new, int)]


def oof_scores_smote(X, y, folds, include_monthly=True, seed=C.SEED, weighted=False):
    """LR trained on SMOTE-balanced TRAINING folds; validation rows stay untouched (real distribution)."""
    X = X.reset_index(drop=True); y = pd.Series(np.asarray(y)); oof = np.zeros(len(y))
    for tr, te in folds:
        fe = build_pipeline(include_monthly).fit(X.iloc[tr], y.iloc[tr])
        Ztr, Zte = fe.transform(X.iloc[tr]).to_numpy(), fe.transform(X.iloc[te]).to_numpy()
        Xs, ys = SimpleSMOTE(random_state=seed).fit_resample(Ztr, y.iloc[tr])
        lr = LogisticRegression(max_iter=2000, class_weight="balanced" if weighted else None, random_state=seed).fit(Xs, ys)
        oof[te] = lr.predict_proba(Zte)[:, 1]
    return oof
