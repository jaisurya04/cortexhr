"""Data audit: schema, missingness, duplicates, impossible values, outliers, leakage evidence."""
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from . import config as C


def schema_inventory(df):
    return pd.DataFrame({
        "column": df.columns, "dtype": [str(t) for t in df.dtypes],
        "non_null": df.notna().sum().values, "missing": df.isna().sum().values,
        "missing_pct": (df.isna().mean() * 100).round(2).values,
        "n_unique": df.nunique(dropna=True).values,
        "example": [df[c].dropna().iloc[0] if df[c].notna().any() else None for c in df.columns],
    })


def missingness(df):
    t = schema_inventory(df)[["column", "missing", "missing_pct"]]
    return t[t.missing > 0].sort_values("missing_pct", ascending=False).reset_index(drop=True)


def duplicates(df, id_col="employee_id"):
    body = df.drop(columns=["row_id"], errors="ignore")
    return {"exact_duplicate_rows": int(body.duplicated().sum()),
            "duplicate_ids_extra_rows": int(df[id_col].duplicated().sum()),
            "ids_affected": sorted(df.loc[df[id_col].duplicated(keep=False), id_col].unique().tolist())}


def impossible_values(df):
    """Rule table: (name, column(s), boolean mask). Only rules whose columns exist are evaluated."""
    rules = []
    def add(name, cols, mask_fn):
        if set(cols) <= set(df.columns): rules.append((name, ", ".join(cols), mask_fn(df)))
    add("negative tenure", ["tenure_months"], lambda d: d.tenure_months < 0)
    add("negative total experience", ["total_experience_years"], lambda d: d.total_experience_years < 0)
    add("experience shorter than tenure", ["total_experience_years", "tenure_months"],
        lambda d: (d.total_experience_years < d.tenure_months / 12 - 0.05) & (d.total_experience_years >= 0) & (d.tenure_months >= 0))
    add("age below 18", ["age"], lambda d: d.age < 18)
    for c in ["attendance_rate_3m", "wfh_share_3m", "dept_attrition_rate_prior_12m"]:
        add(f"{c} above 100%", [c], lambda d, c=c: d[c] > 1)
        add(f"{c} below 0%", [c], lambda d, c=c: d[c] < 0)
    add("ratings outside 1-5", ["last_performance_rating"], lambda d: (d.last_performance_rating < 1) | (d.last_performance_rating > 5))
    add("engagement outside 1-5", ["last_engagement_score"], lambda d: (d.last_engagement_score < 1) | (d.last_engagement_score > 5))
    add("non-positive pay", ["annual_fixed_pay"], lambda d: d.annual_fixed_pay <= 0)
    add("exit before observation", ["exit_date", "observation_date"], lambda d: d.exit_date < d.observation_date)
    return pd.DataFrame([{"rule": n, "columns": c, "rows": int(m.sum()), "row_ids": d_ids(df, m)} for n, c, m in rules])


def d_ids(df, mask, k=8):
    ids = df.loc[mask, "row_id"].tolist() if "row_id" in df else df.index[mask].tolist()
    return ids[:k]


def iqr_outliers(df, cols=None, k=1.5):
    cols = cols or [c for c in df.select_dtypes("number").columns if c not in C.ID_COLS + [C.TARGET, "voluntary_left_within_12m"]]
    rows = []
    for c in cols:
        s = df[c].dropna(); q1, q3 = s.quantile(.25), s.quantile(.75); i = q3 - q1
        lo, hi = q1 - k * i, q3 + k * i
        rows.append({"column": c, "q1": q1, "q3": q3, "lower": lo, "upper": hi,
                     "iqr_outliers": int(((df[c] < lo) | (df[c] > hi)).sum())})
    return pd.DataFrame(rows)


def zscore_outliers(df, cols=None, thr=3.0):
    cols = cols or [c for c in df.select_dtypes("number").columns if c not in C.ID_COLS + [C.TARGET, "voluntary_left_within_12m"]]
    rows = []
    for c in cols:
        s = df[c]; sd = s.std(ddof=0)
        z = (s - s.mean()) / sd if sd > 0 else s * 0
        rows.append({"column": c, "mean": s.mean(), "std": sd, "z_outliers_abs_gt_3": int((z.abs() > thr).sum()),
                     "max_abs_z": float(z.abs().max())})
    return pd.DataFrame(rows)


def class_balance(df, target=C.TARGET):
    vc = df[target].value_counts().sort_index(); n = len(df); pos = int(vc.get(1, 0))
    maj = int(vc.max())
    return {"rows": n, "stayed": int(vc.get(0, 0)), "left": pos, "attrition_pct": round(100 * pos / n, 2),
            "majority_accuracy_pct": round(100 * maj / n, 2), "majority_recall_for_leavers": 0.0}


def _auc_numeric(x, y):
    x = pd.to_numeric(x, errors="coerce"); fill = x.min() - 1 if x.notna().any() else 0
    a = roc_auc_score(y, x.fillna(fill)); return max(a, 1 - a)   # direction-free strength


def _auc_categorical(x, y, seed=C.SEED):
    """Out-of-fold target-mean encoding AUC (no in-sample leakage)."""
    x = x.astype(object).where(x.notna(), "__NA__"); oof = np.zeros(len(x))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(x, y):
        m = y.iloc[tr].groupby(x.iloc[tr]).mean(); oof[te] = x.iloc[te].map(m).fillna(y.iloc[tr].mean()).values
    a = roc_auc_score(y, oof); return max(a, 1 - a)


def single_column_power(df, cols, target=C.TARGET):
    """How well does ONE column predict the target? (direction-free ROC-AUC, 0.5 = useless)."""
    y = df[target]; rows = []
    for c in cols:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s): s = (s - s.min()).dt.days
        auc = _auc_numeric(s, y) if pd.api.types.is_numeric_dtype(s) else _auc_categorical(s, y)
        rows.append({"column": c, "single_column_auc": round(auc, 4)})
    return pd.DataFrame(rows).sort_values("single_column_auc", ascending=False).reset_index(drop=True)


def missingness_indicator_power(df, cols, target=C.TARGET):
    """AUC of 'is this column filled in?' as a predictor of the target (reveals post-outcome fields)."""
    y = df[target]
    return pd.DataFrame([{"column": c, "filled_in_flag_auc": round(roc_auc_score(y, df[c].notna().astype(int)), 4),
                          "filled_when_left_pct": round(100 * df.loc[y == 1, c].notna().mean(), 1),
                          "filled_when_stayed_pct": round(100 * df.loc[y == 0, c].notna().mean(), 1)} for c in cols])


def leakage_review(df):
    """Classify every column as confirmed leakage / suspected / legitimate with evidence."""
    y = df[C.TARGET]; rows = []
    post = [c for c in C.LEAKAGE_CONFIRMED if c in df.columns and c != "voluntary_left_within_12m"]
    mi = missingness_indicator_power(df, post).set_index("column")
    for c in post:
        timing = ""
        if c.endswith("_date"):
            timing = f"; every non-null date is after observation_date in {int((df[c] > df.observation_date).sum())}/{int(df[c].notna().sum())} rows"
        rows.append({"column": c, "verdict": "CONFIRMED LEAKAGE", "evidence":
                     f"filled for {mi.loc[c,'filled_when_left_pct']}% of leavers and {mi.loc[c,'filled_when_stayed_pct']}% of stayers "
                     f"(filled-in flag AUC={mi.loc[c,'filled_in_flag_auc']}){timing}"})
    if "voluntary_left_within_12m" in df:
        sub = int(((df.voluntary_left_within_12m == 1) & (df[C.TARGET] == 0)).sum())
        rows.append({"column": "voluntary_left_within_12m", "verdict": "CONFIRMED LEAKAGE",
                     "evidence": f"derived from the exit event; 1 in {int(df.voluntary_left_within_12m.sum())} rows, all of them target=1 "
                                 f"(rows with voluntary=1 and target=0: {sub}); cannot exist before the outcome"})
    if "source" in df:
        auc = _auc_categorical(df["source"], y); rate = df.groupby("source")[C.TARGET].mean().round(3).to_dict()
        rows.append({"column": "source", "verdict": "SUSPECTED (excluded)",
                     "evidence": f"dataset-origin tag, not an employee attribute; label rates {rate}; single-column AUC={auc:.3f}"})
    if "observation_date" in df:
        a = _auc_numeric((df.observation_date - df.observation_date.min()).dt.days, y)
        rows.append({"column": "observation_date", "verdict": "SUSPECTED (excluded)",
                     "evidence": f"time stamp of the snapshot (2021-2025 for historical rows); single-column AUC={a:.3f}; excluded to avoid time proxies"})
    return pd.DataFrame(rows)


def pay_consistency_outliers(df, factor=3.0):
    """annual_fixed_pay / comp_ratio is the role mid-point; it should be ~constant inside a role.
    (Added after the vault comparison: it finds salary typos a one-column check misses.)"""
    if not {"annual_fixed_pay", "comp_ratio", "role_id"} <= set(df.columns): return df.iloc[0:0]
    mid = df.annual_fixed_pay / df.comp_ratio; ratio = mid / mid.groupby(df.role_id).transform("median")
    return df.loc[ratio > factor, [c for c in ["row_id", "employee_id", "role_id", "annual_fixed_pay", "comp_ratio"] if c in df]].assign(ratio_to_role_midpoint=ratio[ratio > factor].round(2))


def _row(problem, where, n, evidence, severity, treatment, reason):
    return {"problem": problem, "affected_column_or_location": where, "rows_affected": n,
            "evidence": evidence, "severity": severity, "proposed_treatment": treatment, "reason": reason}


def build_audit_table(df, att=None, pay=None):
    """Evidence-based findings; a finding is only listed when the data actually shows it."""
    F = []
    n = len(df)
    # 1 leakage
    lk = leakage_review(df); conf = lk[lk.verdict == "CONFIRMED LEAKAGE"]
    post_rows = int(df["exit_date"].notna().sum()) if "exit_date" in df else 0
    F.append(_row("Post-outcome (leakage) columns", ", ".join(conf.column), post_rows,
                  "; ".join(f"{r.column}: {r.evidence}" for r in conf.itertuples())[:600], "Critical",
                  "Drop from every model input", "They only exist after a person leaves, so a model would read the answer (benchmark: leaky LR scores AUC 1.00)."))
    # 2 duplicates
    d = duplicates(df)
    if d["exact_duplicate_rows"]:
        F.append(_row("Exact duplicate snapshots", "employee_id", d["exact_duplicate_rows"],
                      f"{d['exact_duplicate_rows']} rows repeat another row in all columns; ids {d['ids_affected']}", "High",
                      "Keep first occurrence, drop repeats", "Duplicates inflate counts and can sit in both train and test folds."))
    # 3 impossible
    imp = impossible_values(df); imp = imp[imp.rows > 0]
    for r in imp.itertuples():
        sev = "High" if "tenure" in r.rule or "experience" in r.rule else "Medium"
        treat = ("Set to missing, impute on training data only" if ("negative" in r.rule or "shorter" in r.rule)
                 else "Clip to the valid 0-1 range")
        F.append(_row(f"Impossible value: {r.rule}", r.columns, r.rows, f"{r.rows} rows violate the rule (row_ids e.g. {r.row_ids[:5]})",
                      sev, treat, "Values outside physical/logical limits are entry or generation errors."))
    # 4 inconsistent categories
    if "work_location" in df:
        from .data_loading import canonical_city
        raw, can = df.work_location.nunique(), canonical_city(df.work_location).nunique()
        chg = int((df.work_location != canonical_city(df.work_location)).sum())
        if raw != can:
            F.append(_row("Inconsistent city spellings", "work_location", chg, f"{raw} spellings map to {can} real cities "
                          f"(e.g. {sorted(df.work_location.unique())[:6]})", "Medium", "Map to canonical names", "One city split in many categories dilutes one-hot features."))
    # 5 missingness
    ms = missingness(df); ms = ms[~ms.column.isin(C.LEAKAGE_CONFIRMED)]
    F.append(_row("Missing values in predictors", ", ".join(f"{r.column} ({r.missing_pct}%)" for r in ms.itertuples()),
                  int(ms.missing.max()) if len(ms) else 0, "Largest: " + (f"{ms.iloc[0].column} {ms.iloc[0].missing_pct}%" if len(ms) else "none"),
                  "Medium", "Median/most-frequent imputation fitted on training only + missing-indicator", "Some blanks are meaningful (never promoted, no survey reply)."))
    if "months_since_last_promotion" in df:
        nv = int((df.promotions_received == 0).sum()); mp = int(df.months_since_last_promotion.isna().sum())
        F.append(_row("Structural missingness: never promoted", "months_since_last_promotion", mp,
                      f"{mp} blanks vs {nv} rows with promotions_received=0 -> blank means 'never promoted', not 'unknown'", "Low",
                      "Derive never_promoted flag; impute number", "Blank carries information; deleting rows would bias the sample."))
    # 6 class imbalance
    cb = class_balance(df)
    F.append(_row("Class imbalance", C.TARGET, cb["left"], f"{cb['attrition_pct']}% left; always predicting 'stay' = {cb['majority_accuracy_pct']}% accuracy and 0% recall",
                  "High", "Stratified splits/folds, class weights, PR-AUC/recall instead of accuracy", "Accuracy rewards ignoring the minority class."))
    # 7 outliers
    iq = iqr_outliers(df); zs = zscore_outliers(df)
    top = iq.sort_values("iqr_outliers", ascending=False).head(3)
    F.append(_row("Statistical outliers (IQR and |z|>3)", ", ".join(top.column), int(top.iqr_outliers.max()),
                  f"IQR: {', '.join(f'{r.column}={r.iqr_outliers}' for r in top.itertuples())}; |z|>3 in {int((zs.z_outliers_abs_gt_3>0).sum())} columns "
                  f"(max z={zs.max_abs_z.max():.1f}, e.g. annual_fixed_pay)", "Low",
                  "Keep and investigate; use log pay/scaling; do not auto-delete", "Skewed pay and long tenures are real senior staff, not errors."))
    # 8 time / source structure
    if "source" in df:
        vc = df.source.value_counts().to_dict()
        F.append(_row("Mixed row origins and time span", "source, observation_date", int(vc.get("historical_extension", 0)),
                      f"{vc}; observation_date spans {df.observation_date.min().date()} to {df.observation_date.max().date()}; only master rows have attendance/payroll history",
                      "High", "Exclude source/date as inputs; report coverage of monthly features; ablate with/without them",
                      "Model could learn 'which batch' rather than 'who leaves'; monthly features exist for 16% of rows."))
    # 8b label already visible at observation time
    if {"resignation_date", "observation_date"} <= set(df.columns):
        early = df["resignation_date"].notna() & (df["resignation_date"] <= df["observation_date"])
        if early.sum():
            F.append(_row("Leaver had already resigned before the snapshot", "resignation_date vs observation_date", int(early.sum()),
                          f"{int(early.sum())} leavers submitted resignation on/before observation_date ({100*early.sum()/df[C.TARGET].sum():.1f}% of all leavers): "
                          "the outcome was already decided when the features were measured", "Medium",
                          "Keep rows (cannot be detected from safe columns); report a sensitivity run excluding them",
                          "Prediction at that date is not truly forward-looking; inflates how easy these leavers look."))
    # 9 payroll duplicates / monthly coverage
    if pay is not None:
        from .data_loading import dedupe_payroll
        _, nd = dedupe_payroll(pay)
        if nd:
            F.append(_row("Duplicate payroll employee-month rows", "payroll_monthly_features", nd,
                          f"{nd} rows identical to another row except payroll_item_id", "Medium", "Drop exact repeats before aggregating",
                          "A repeated month would double-count pay in a 3-month sum."))
    if att is not None and "exit_date" in df:
        ex = df.loc[df.exit_date.notna(), ["employee_id", "exit_date"]]
        m = att.merge(ex, on="employee_id"); post = int((m.month > m.exit_date).sum())
        F.append(_row("Monthly records after the observation date", "attendance/payroll months up to 2026-09", int((att.month > pd.Timestamp("2025-09-30")).sum()),
                      f"{int((att.month > pd.Timestamp('2025-09-30')).sum())} attendance rows are dated after the 30 Sep 2025 snapshot (future information); {post} are after an exit date",
                      "High", "Aggregate only complete months up to observation_date", "Using Oct 2025 - Sep 2026 attendance would reveal the future."))
    # 10 hidden structure in single columns
    sc = single_column_power(df, [c for c in C.NUMERIC_CORE + C.CATEGORICAL_CORE if c in df.columns])
    return pd.DataFrame(F), lk, sc


# ------------------------------- plots -------------------------------
def column_roles(df):
    """Drop / retain / transform decision for every column (input to the Day 2 pipeline)."""
    rows = []
    for c in df.columns:
        if c == C.TARGET: r, why = "TARGET", "label; never an input"
        elif c in C.ID_COLS: r, why = "DROP", "identifier, no predictive meaning"
        elif c in C.DATE_COLS: r, why = "DROP", "time stamp of snapshot; time proxy / not a driver"
        elif c in C.LEAKAGE_CONFIRMED: r, why = "DROP", "confirmed post-outcome leakage"
        elif c in C.METADATA_EXCLUDED: r, why = "DROP", "dataset-origin tag, suspected proxy"
        elif c in C.CATEGORICAL_CORE: r, why = "TRANSFORM", "impute + one-hot (unknown-safe)"
        elif c in ("annual_fixed_pay", "total_experience_years", "tenure_months", "wfh_share_3m", "attendance_rate_3m"):
            r, why = "TRANSFORM", "clean impossible values, then impute + scale" + (" (+ dept-relative pay)" if c == "annual_fixed_pay" else "")
        elif c in C.NUMERIC_CORE: r, why = "RETAIN", "legitimate predictor: impute + scale"
        else: r, why = "REVIEW", "not classified"
        rows.append({"column": c, "decision": r, "reason": why})
    return pd.DataFrame(rows)


def plot_class_balance(cb, title="Class balance: who left within 12 months"):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(["Stayed (0)", "Left (1)"], [cb["stayed"], cb["left"]], color=["#4C78A8", "#E45756"])
    for b, v in zip(bars, [cb["stayed"], cb["left"]]):
        ax.text(b.get_x() + b.get_width() / 2, v + 20, f"{v:,}\n({100*v/cb['rows']:.1f}%)", ha="center", fontsize=9)
    ax.set_ylim(0, cb["stayed"] * 1.18); ax.set_ylabel("Employees"); ax.set_title(title, fontsize=10)
    ax.text(0.5, 0.97, f"'Always stay' = {cb['majority_accuracy_pct']}% accurate, catches 0 leavers",
            transform=ax.transAxes, ha="center", va="top", fontsize=8, color="#444")
    return fig


def plot_missingness(ms):
    import matplotlib.pyplot as plt
    d = ms.sort_values("missing_pct"); colors = ["#E45756" if c in C.LEAKAGE_CONFIRMED else "#4C78A8" for c in d.column]
    fig, ax = plt.subplots(figsize=(7, 4)); ax.barh(d.column, d.missing_pct, color=colors)
    ax.set_xlabel("% missing"); ax.set_title("Missing values (red = post-outcome leakage columns, blank by design for stayers)", fontsize=9)
    return fig


def plot_outliers(df, cols=("annual_fixed_pay", "tenure_months", "commute_distance_km", "avg_overtime_hours_per_month_3m", "training_hours_12m", "late_arrivals_3m")):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, c in zip(axes.ravel(), cols):
        ax.boxplot(df[c].dropna(), vert=False, flierprops=dict(marker=".", markersize=3, alpha=.5)); ax.set_title(c, fontsize=9); ax.set_yticks([])
    fig.suptitle("Outlier screening (IQR whiskers; dots = candidates to INVESTIGATE, not delete)", fontsize=10); fig.tight_layout()
    return fig


def plot_leakage(lk_power, single):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    d = lk_power.sort_values("filled_in_flag_auc"); axes[0].barh(d.column, d.filled_in_flag_auc, color="#E45756"); axes[0].axvline(.9, ls="--", c="k")
    axes[0].set_title("Leakage suspects: 'is it filled in?' predicts target\n(AUC; >0.9 = answer is hidden in the column)", fontsize=9); axes[0].set_xlim(.5, 1.02)
    s = single.head(12).sort_values("single_column_auc"); axes[1].barh(s.column, s.single_column_auc, color="#4C78A8"); axes[1].axvline(.9, ls="--", c="k")
    axes[1].set_xlim(.5, 1.02); axes[1].set_title("Legitimate predictors: best single-column AUC\n(all weak => no hidden leak)", fontsize=9)
    fig.tight_layout(); return fig
