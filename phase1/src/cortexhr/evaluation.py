"""Honest evaluation: binary_report, risk_bands, top-10% metrics, calibration, lift/gains, catch_rate.

NOTE: the official starter file `cortexhr_helpers.py` (binary_report, risk_bands, catch_rate) was NOT supplied.
These are independent implementations of the behaviour described in the task text.
"""
from __future__ import annotations
import math
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve, precision_recall_curve)

from .official import H
BAND_ORDER = ["High", "Medium", "Low"]


def rank_order(scores) -> np.ndarray:
    """Positions sorted by score DESCENDING; ties broken by original row position (deterministic)."""
    s = np.asarray(scores, dtype=float)
    return np.lexsort((np.arange(len(s)), -s))


def top_fraction_mask(scores, frac: float = 0.10) -> np.ndarray:
    """Boolean mask of the top ceil(frac*n) scored rows (ties resolved by row position)."""
    n = len(scores); k = int(math.ceil(frac * n - 1e-12)); m = np.zeros(n, bool); m[rank_order(scores)[:k]] = True
    return m


def binary_report(y_true, y_score, threshold: float = 0.5, top_frac: float = 0.10, zero_division: int = 0) -> dict:
    """ROC-AUC, PR-AUC, threshold metrics, confusion matrix, and top-fraction precision/recall/lift.
    Undefined precision/F1 (no positive predictions) are reported as `zero_division` and flagged."""
    y = np.asarray(y_true).astype(int); s = np.asarray(y_score, dtype=float); pred = (s >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    top = top_fraction_mask(s, top_frac); k = int(top.sum()); tp_top = int(y[top].sum()); base = y.mean()
    return {
        "n": len(y), "positives": int(y.sum()), "base_rate": base,
        "roc_auc": roc_auc_score(y, s), "pr_auc": average_precision_score(y, s),
        "threshold": threshold, "precision": precision_score(y, pred, zero_division=zero_division),
        "recall": recall_score(y, pred, zero_division=zero_division), "f1": f1_score(y, pred, zero_division=zero_division),
        "accuracy": float((pred == y).mean()), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "precision_defined": bool(tp + fp > 0), "brier": brier_score_loss(y, np.clip(s, 0, 1)),
        "top_frac": top_frac, "top_k": k, "top_precision": tp_top / k if k else float("nan"),
        "top_recall": tp_top / y.sum() if y.sum() else float("nan"),
        "top_lift": (tp_top / k) / base if k and base else float("nan"),
    }


def risk_bands(scores, high: float = 0.10, medium: float = 0.20) -> pd.Series:
    """High = top 10% of scores, Medium = next 20%, Low = remaining 70%. Ties broken by row position."""
    s = np.asarray(scores, dtype=float); n = len(s)
    k_high = int(math.ceil(high * n - 1e-12)); k_med = int(math.ceil((high + medium) * n - 1e-12))
    order = rank_order(s); lab = np.empty(n, dtype=object)
    lab[order[:k_high]] = "High"; lab[order[k_high:k_med]] = "Medium"; lab[order[k_med:]] = "Low"
    return pd.Series(pd.Categorical(lab, categories=BAND_ORDER, ordered=True), index=getattr(scores, "index", None), name="risk_band")


def band_summary(y_true, bands) -> pd.DataFrame:
    y = pd.Series(np.asarray(y_true).astype(int)); b = pd.Series(np.asarray(bands), dtype=pd.CategoricalDtype(BAND_ORDER, ordered=True))
    g = pd.DataFrame({"band": b, "y": y}).groupby("band", observed=False)["y"].agg(employees="size", leavers="sum").reset_index()
    g["share_of_employees_pct"] = 100 * g.employees / g.employees.sum()
    g["leaver_rate_pct"] = 100 * g.leavers / g.employees
    g["share_of_all_leavers_pct"] = 100 * g.leavers / max(y.sum(), 1)
    g["lift_vs_base_rate"] = g.leaver_rate_pct / (100 * y.mean())
    return g


def gains_table(y_true, scores, n_bins: int = 10) -> pd.DataFrame:
    y = np.asarray(y_true).astype(int); order = rank_order(scores); ys = y[order]; n = len(y)
    rows = []
    for i in range(1, n_bins + 1):
        k = int(math.ceil(i / n_bins * n)); cap = ys[:k].sum() / max(y.sum(), 1)
        rows.append({"top_pct": 100 * i / n_bins, "k": k, "leavers_captured": int(ys[:k].sum()),
                     "capture_rate_pct": 100 * cap, "random_capture_pct": 100 * i / n_bins, "lift": cap / (i / n_bins)})
    return pd.DataFrame(rows)


def calibration_table(y_true, p, n_bins: int = 10, strategy: str = "quantile") -> pd.DataFrame:
    frac_pos, mean_pred = calibration_curve(y_true, np.clip(p, 0, 1), n_bins=n_bins, strategy=strategy)
    return pd.DataFrame({"mean_predicted": mean_pred, "observed_rate": frac_pos})


class Recalibrator:
    """Platt scaling fitted on out-of-fold TRAINING scores only; applied unchanged to held-out data.
    Monotone => ranking, ROC-AUC, PR-AUC, top-10% and risk bands are unchanged."""
    def fit(self, scores, y):
        z = self._logit(scores); self.lr_ = LogisticRegression(C=1e6, max_iter=1000).fit(z, np.asarray(y)); return self
    @staticmethod
    def _logit(s): s = np.clip(np.asarray(s, float), 1e-6, 1 - 1e-6); return np.log(s / (1 - s)).reshape(-1, 1)
    def predict(self, scores): return self.lr_.predict_proba(self._logit(scores))[:, 1]


def catch_rate(flagged, truth, universe=None) -> dict:
    """Official catch_rate (cortexhr_helpers) plus an `injected` alias of `real`."""
    r = dict(H.catch_rate(flagged, truth)); r["injected"] = r["real"]; return r


def official_report(y_true, y_score, top_fraction: float = 0.10) -> dict:
    """The course's own binary_report (top group = int(n * fraction), argsort order)."""
    return dict(H.binary_report(y_true, y_score, top_fraction))


def official_risk_bands(scores) -> pd.Series:
    """The company rule from cortexhr_helpers.risk_bands (quantile based: High >= P90, Medium >= P70)."""
    lab = H.risk_bands(np.asarray(scores, float))
    return pd.Series(pd.Categorical(lab, categories=BAND_ORDER, ordered=True), index=getattr(scores, "index", None), name="risk_band")


def parity_table(y_true, y_score) -> pd.DataFrame:
    """Official helper vs my extended report: AUCs must be identical; the top-10% group differs only by rounding/ties."""
    o, m = official_report(y_true, y_score), binary_report(y_true, y_score)
    rows = [("roc_auc", o["roc_auc"], m["roc_auc"]), ("pr_auc", o["pr_auc"], m["pr_auc"]), ("base_rate", o["base_rate"], m["base_rate"]),
            ("top-10% group size", max(1, int(len(y_true) * 0.10)), m["top_k"]), ("precision in top group", o["precision_at_top"], m["top_precision"]),
            ("recall in top group", o["recall_at_top"], m["top_recall"])]
    return pd.DataFrame(rows, columns=["metric", "official_helper", "my_extended_report"]).assign(difference=lambda d: d.my_extended_report - d.official_helper)


# ------------------------------ plots ------------------------------
def plot_calibration(curves: dict, title="Calibration: does a 30% score mean 3 in 10?"):
    fig, ax = plt.subplots(figsize=(5.5, 5)); ax.plot([0, 1], [0, 1], "k--", label="perfect")
    for name, t in curves.items(): ax.plot(t.mean_predicted, t.observed_rate, "o-", label=name)
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed leaver rate"); ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8); ax.set_xlim(0, 1); ax.set_ylim(0, 1); return fig


def plot_gains_lift(gains: dict, title="Cumulative gains vs random selection"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, g in gains.items():
        x = np.r_[0, g.top_pct]; axes[0].plot(x, np.r_[0, g.capture_rate_pct], "o-", label=name)
        axes[1].bar(g.top_pct - 5 + (list(gains).index(name) * 2 - 1), g.lift, width=2, label=name)
    axes[0].plot([0, 100], [0, 100], "k--", label="random"); axes[0].set_xlabel("% of employees reviewed (highest score first)")
    axes[0].set_ylabel("% of all leavers captured"); axes[0].set_title(title, fontsize=10); axes[0].legend(fontsize=8)
    axes[1].axhline(1, c="k", ls="--"); axes[1].set_xlabel("Score decile (top % of employees)"); axes[1].set_ylabel("Lift vs random")
    axes[1].set_title("Lift by decile (1.0 = random)", fontsize=10); axes[1].legend(fontsize=8); fig.tight_layout(); return fig


def plot_roc_pr(curves: dict):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for name, (y, s) in curves.items():
        fpr, tpr, _ = roc_curve(y, s); pr, rc, _ = precision_recall_curve(y, s)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, s):.3f})"); axes[1].plot(rc, pr, label=f"{name} (AP {average_precision_score(y, s):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--"); axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate"); axes[0].set_title("ROC curve", fontsize=10)
    base = np.mean(list(curves.values())[0][0]); axes[1].axhline(base, c="k", ls="--"); axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall curve (dashed = base rate)", fontsize=10); axes[0].legend(fontsize=8); axes[1].legend(fontsize=8); fig.tight_layout(); return fig


def plot_bands(summary: pd.DataFrame, title="Risk bands: leaver rate vs base rate"):
    fig, ax = plt.subplots(figsize=(5.5, 4)); cols = {"High": "#E45756", "Medium": "#F2B134", "Low": "#4C78A8"}
    ax.bar(summary.band.astype(str), summary.leaver_rate_pct, color=[cols[b] for b in summary.band.astype(str)])
    for i, r in summary.iterrows(): ax.text(i, r.leaver_rate_pct + .4, f"{r.leaver_rate_pct:.1f}%\n(n={int(r.employees)})", ha="center", fontsize=8)
    ax.set_ylabel("% who actually left"); ax.set_title(title, fontsize=10); ax.set_ylim(0, summary.leaver_rate_pct.max() * 1.3); return fig
