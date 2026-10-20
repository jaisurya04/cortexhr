"""Day 5: unsupervised attendance anomaly detection (IsolationForest + simple rules). Labels are NEVER used here."""
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
from . import config as C
from .evaluation import catch_rate

RAW_FEATURES = ["scheduled_days", "present_days", "half_days_unpaid", "paid_leave_days", "unpaid_absence_days",
                "longest_unpaid_streak_days", "late_arrivals", "avg_hours_worked", "overtime_hours", "attendance_rate"]
HISTORY_FEATURES = ["rate_vs_own_median", "late_vs_own_median", "hours_vs_own_median"]
LABEL_LIKE = {"is_anomaly", "anomaly_type", "anomaly", "label"}


def assert_no_labels(cols):
    bad = {c.lower() for c in cols} & LABEL_LIKE
    if bad: raise ValueError(f"Label-like columns must never be features: {sorted(bad)}")


def add_history_features(att: pd.DataFrame) -> pd.DataFrame:
    """Each month compared with the same employee's own typical month (label-free)."""
    d = att.copy(); g = d.groupby("employee_id")
    d["rate_vs_own_median"] = d["attendance_rate"] - g["attendance_rate"].transform("median")
    d["late_vs_own_median"] = d["late_arrivals"] - g["late_arrivals"].transform("median")
    d["hours_vs_own_median"] = d["avg_hours_worked"] - g["avg_hours_worked"].transform("median")
    return d


def rule_flags(att: pd.DataFrame) -> pd.DataFrame:
    """Simple, explainable rules from domain logic (no labels):
    R1 zero_attendance_month : scheduled to work but present 0 days in the whole month
    R2 implausible_hours     : average hours > 12 per day, or < 3 on days when the person was present"""
    r1 = (att["scheduled_days"] > 0) & (att["present_days"] == 0)
    r2 = (att["avg_hours_worked"] > 12) | ((att["avg_hours_worked"] < 3) & (att["present_days"] > 0))
    return pd.DataFrame({"rule_zero_attendance": r1.to_numpy(), "rule_implausible_hours": r2.to_numpy()}, index=att.index)


def fit_isolation_forest(X: pd.DataFrame, contamination: float = 0.03, seed: int = C.SEED, n_estimators: int = 300):
    assert_no_labels(X.columns)
    iso = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=seed).fit(X)
    score = -iso.score_samples(X)            # higher = easier to isolate = more unusual
    return iso, score, iso.predict(X) == -1


def detect(att: pd.DataFrame, contamination: float = 0.03, seed: int = C.SEED, use_history: bool = True) -> pd.DataFrame:
    d = add_history_features(att); feats = RAW_FEATURES + (HISTORY_FEATURES if use_history else [])
    _, score, flag = fit_isolation_forest(d[feats], contamination, seed)
    _, score_raw, flag_raw = fit_isolation_forest(d[RAW_FEATURES], contamination, seed)   # untuned baseline-style variant
    out = d[["attendance_feature_id", "employee_id", "month"]].copy()
    out["iforest_score"] = score; out["iforest_flag"] = flag; out["iforest_raw_only_flag"] = flag_raw
    out = pd.concat([out, rule_flags(d)], axis=1)
    out["rule_flag"] = out[["rule_zero_attendance", "rule_implausible_hours"]].any(axis=1)
    out["flagged"] = out["iforest_flag"] | out["rule_flag"]
    out["flag_reason"] = np.select([out.rule_zero_attendance, out.rule_implausible_hours, out.iforest_flag],
                                   ["rule: zero attendance in month", "rule: implausible hours", "model: unusual pattern"], "")
    # review priority: rule hits first (explainable, cheap to verify), then by model score
    out["priority_score"] = out["iforest_score"] + out["rule_flag"] * 1.0
    out["review_rank"] = np.nan
    fl = out[out.flagged].sort_values(["priority_score", "attendance_feature_id"], ascending=[False, True])
    out.loc[fl.index, "review_rank"] = np.arange(1, len(fl) + 1)
    return out


def review_queue(flags: pd.DataFrame, capacity: int) -> pd.DataFrame:
    """Reviewers can check `capacity` rows a month: take the highest-priority flags."""
    return flags[flags.flagged].sort_values("review_rank").head(capacity)


def load_labels(path):
    L = pd.read_csv(path) if str(path).lower().endswith(".csv") else pd.read_excel(path)
    L.columns = [c.strip() for c in L.columns]; return L


def measure(flags: pd.DataFrame, labels: pd.DataFrame, ks=(50, 100, 150, 221)) -> dict:
    """Catch rate of every detector variant against hidden labels (measurement ONLY)."""
    key = "attendance_feature_id" if "attendance_feature_id" in labels.columns else None
    if key is None: raise KeyError(f"Labels need attendance_feature_id; got {list(labels.columns)}")
    truth = set(labels[key]); rows = []
    variants = {"IsolationForest (raw features only)": flags.iforest_raw_only_flag, "IsolationForest (+own-history features)": flags.iforest_flag,
                "Rules only": flags.rule_flag, "Combined (IF + rules)": flags.flagged}
    for name, m in variants.items():
        r = catch_rate(flags.loc[m, key], truth); r["detector"] = name; rows.append(r)
    table = pd.DataFrame(rows)[["detector", "flagged", "injected", "caught", "precision", "recall"]]
    ranked = flags[flags.flagged].sort_values("review_rank")
    topk = pd.DataFrame([{"review_capacity": k, **{c: v for c, v in catch_rate(ranked.head(k)[key], truth).items() if c in ("flagged", "caught", "precision", "recall")}} for k in ks])
    tcol = next((c for c in labels.columns if "type" in c.lower()), None); by_type = None
    if tcol:
        lab = labels[[key, tcol]].drop_duplicates(); caught = set(flags.loc[flags.flagged, key])
        by_type = lab.assign(caught=lab[key].isin(caught)).groupby(tcol).agg(injected=("caught", "size"), caught=("caught", "sum")).assign(
            recall=lambda t: t.caught / t.injected, missed=lambda t: t.injected - t.caught).reset_index()
        for v, m in (("model_only_recall", flags.iforest_flag), ("rule_only_recall", flags.rule_flag)):
            s = set(flags.loc[m, key]); by_type[v] = [lab[lab[tcol] == t][key].isin(s).mean() for t in by_type[tcol]]
    return {"overall": table, "top_k": topk, "by_type": by_type}


# ---------- supplementary: labelled DAILY sample shipped in the main workbook (measurement only) ----------
def daily_proxy_eval(daily: pd.DataFrame, contamination: float = 0.03, seed: int = C.SEED) -> pd.DataFrame:
    d = daily.copy(); mins = lambda s: pd.to_datetime(s, format="%H:%M", errors="coerce").pipe(lambda t: t.dt.hour * 60 + t.dt.minute)
    X = pd.DataFrame({"hours_worked": d.hours_worked.fillna(0), "overtime_hours": d.overtime_hours.fillna(0),
                      "late": (d.late_flag == "Y").astype(int), "check_in_min": mins(d.check_in).fillna(-1), "check_out_min": mins(d.check_out).fillna(-1)})
    X = pd.concat([X, pd.get_dummies(d.status.fillna("NA"), prefix="status").astype(int)], axis=1)
    _, score, f_if = fit_isolation_forest(X, contamination, seed)
    present = d.status.isin(["PRESENT", "WFH"])
    r = (d.hours_worked > 12) | ((d.hours_worked < 3) & present & d.hours_worked.notna()) | (d.check_out.notna() & d.check_in.notna() & (mins(d.check_out) < mins(d.check_in)))
    truth = d.is_anomaly == 1; rows = []
    for name, m in {"Daily IsolationForest": pd.Series(f_if, index=d.index), "Daily rule (hours/clock)": r, "Daily combined": pd.Series(f_if, index=d.index) | r}.items():
        tp = int((m & truth).sum()); rows.append({"detector": name, "flagged": int(m.sum()), "injected": int(truth.sum()), "caught": tp,
                                                  "precision": tp / max(int(m.sum()), 1), "recall": tp / int(truth.sum())})
    return pd.DataFrame(rows)
