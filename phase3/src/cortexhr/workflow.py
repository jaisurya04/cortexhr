"""Helpers for the 12-stage workflow shown in every notebook: EDA, cleaning report, tuning, interpretation."""
from __future__ import annotations
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.inspection import permutation_importance
from . import config as C


def rate_by(df, col, target=C.TARGET, bins=None, q=None):
    s = df[col]
    if q: s = pd.qcut(s, q, duplicates="drop")
    elif bins is not None: s = pd.cut(s, bins)
    g = df.groupby(s, observed=True)[target].agg(employees="size", leavers="sum")
    g["leaver_rate_pct"] = (100 * g.leavers / g.employees).round(1); return g.reset_index()


def plot_eda(df, target=C.TARGET):
    fig, ax = plt.subplots(2, 3, figsize=(14, 7)); base = 100 * df[target].mean()
    specs = [("department_id", None, None), ("job_level", None, None), ("work_mode", None, None),
             ("tenure_months", [0, 12, 24, 48, 96, 400], None), ("comp_ratio", None, 4), ("last_engagement_score", None, 4)]
    for a, (c, bins, q) in zip(ax.ravel(), specs):
        t = rate_by(df.assign(**{c: df[c].where(df[c] >= 0) if c == "tenure_months" else df[c]}), c, target, bins, q)
        a.bar(t[c].astype(str), t.leaver_rate_pct, color="#4C78A8"); a.axhline(base, c="r", ls="--", lw=1)
        a.set_title(f"Leaver % by {c}", fontsize=9); a.tick_params(axis="x", labelrotation=45, labelsize=7)
    fig.suptitle(f"EDA: attrition rate by group (red line = overall {base:.1f}%)", fontsize=11); fig.tight_layout(); return fig


def numeric_vs_target(df, cols, target=C.TARGET):
    t = df.groupby(target)[cols].mean().T.rename(columns={0: "mean_stayed", 1: "mean_left"})
    t["relative_diff_pct"] = (100 * (t.mean_left - t.mean_stayed) / t.mean_stayed.abs().replace(0, np.nan)).round(1)
    return t.round(3).sort_values("relative_diff_pct", key=abs, ascending=False)


def cleaning_report(raw, cleaned):
    """Counts of each issue before and after cleaning."""
    def counts(d):
        return {"exact duplicate rows": int(d.drop(columns=["row_id"], errors="ignore").duplicated().sum()),
                "negative tenure": int((d.tenure_months < 0).sum()), "negative total experience": int((d.total_experience_years < 0).sum()),
                "experience shorter than tenure": int((d.total_experience_years < d.tenure_months / 12 - 0.05).sum()),
                "attendance_rate_3m > 1": int((d.attendance_rate_3m > 1).sum()), "wfh_share_3m outside 0-1": int(((d.wfh_share_3m > 1) | (d.wfh_share_3m < 0)).sum()),
                "distinct city spellings": int(d.work_location.nunique()), "rows": len(d)}
    return pd.DataFrame({"before": counts(raw), "after": counts(cleaned)})


def tune_grid(make_model, grid: dict, X, y, folds):
    """Cross-validated search on TRAINING folds only (the test set is never used for tuning)."""
    rows = []
    for name, params in grid.items():
        s = cross_val_predict(make_model(**params), X, y, cv=folds, method="predict_proba")[:, 1]
        fold_auc = [roc_auc_score(np.asarray(y)[te], s[te]) for _, te in folds]
        rows.append({"setting": name, "cv_roc_auc": np.mean(fold_auc), "cv_roc_auc_sd": np.std(fold_auc, ddof=1), "cv_pr_auc": average_precision_score(y, s)})
    return pd.DataFrame(rows).sort_values("cv_roc_auc", ascending=False).reset_index(drop=True)


def logreg_coefficients(fitted, top=12):
    fe, lr = fitted.named_steps["features"], fitted.named_steps["model"]
    names = fe.named_steps["prep"].get_feature_names_out()
    t = pd.DataFrame({"feature": names, "coefficient": lr.coef_[0]}); t["abs"] = t.coefficient.abs()
    return t.sort_values("abs", ascending=False).head(top).drop(columns="abs").reset_index(drop=True)


def permutation_table(fitted, X_valid, y_valid, n_repeats=5, seed=C.SEED, top=12):
    """Drop in ROC-AUC when one RAW input column is shuffled (computed on validation rows, never test)."""
    r = permutation_importance(fitted, X_valid, y_valid, scoring="roc_auc", n_repeats=n_repeats, random_state=seed, n_jobs=1)
    t = pd.DataFrame({"column": X_valid.columns, "auc_drop": r.importances_mean, "sd": r.importances_std})
    return t.sort_values("auc_drop", ascending=False).head(top).reset_index(drop=True)


def plot_confusion_counts(tn, fp, fn, tp, title="Confusion matrix"):
    cm = np.array([[tn, fp], [fn, tp]]); fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({100 * cm[i, j] / cm[i].sum():.0f}% of row)", ha="center", va="center", fontsize=10,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["predicted stay", "predicted leave"]); ax.set_yticks([0, 1], ["actual stay", "actual leave"])
    ax.set_title(title, fontsize=10); fig.tight_layout(); return fig


def plot_confusion(y_true, scores, threshold=0.5, title="Confusion matrix"):
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(np.asarray(y_true).astype(int), (np.asarray(scores) >= threshold).astype(int), labels=[0, 1]).ravel()
    return plot_confusion_counts(tn, fp, fn, tp, f"{title} (threshold {threshold:g})")


def plot_confusion_grid(y_true, scores: dict, threshold=0.5):
    from sklearn.metrics import confusion_matrix
    fig, axes = plt.subplots(1, len(scores), figsize=(3.9 * len(scores), 3.8)); axes = np.atleast_1d(axes)
    for ax, (nm, s) in zip(axes, scores.items()):
        tn, fp, fn, tp = confusion_matrix(np.asarray(y_true).astype(int), (np.asarray(s) >= threshold).astype(int), labels=[0, 1]).ravel()
        cm = np.array([[tn, fp], [fn, tp]]); ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2): ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["pred stay", "pred leave"], fontsize=7); ax.set_yticks([0, 1], ["actual stay", "actual leave"], fontsize=7); ax.set_title(nm, fontsize=8)
    fig.tight_layout(); return fig
