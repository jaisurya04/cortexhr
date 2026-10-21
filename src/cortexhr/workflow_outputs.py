"""Script-side outputs for the 12-stage workflow (EDA, cleaning, confusion matrices, tuning, interpretation).
Everything the notebooks show is also written to reports/ by the scripts, so scripts and notebooks are equivalent."""
from __future__ import annotations
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression
from . import config as C, data_loading as D, audit as A, features as F, modeling as M, evaluation as E, reporting as R, workflow as W

MK = lambda est, monthly=True: Pipeline([("features", F.build_pipeline(monthly)), ("model", est)])
LRB = lambda **kw: LogisticRegression(class_weight="balanced", max_iter=2000, random_state=C.SEED, **kw)
KEYS = ["roc_auc", "pr_auc", "precision", "recall", "f1", "accuracy", "top_precision", "top_recall"]


def _row(y, s, thr=0.5):
    r = E.binary_report(y, s, threshold=thr); return {k: round(r[k], 4) for k in KEYS}


def _interpret(prefix, Xtr, ytr, folds, monthly=True):
    tr, va = folds[0]; m = MK(LRB(), monthly).fit(Xtr.iloc[tr], ytr.iloc[tr])
    R.save_table(W.logreg_coefficients(m, 20), f"{prefix}_coefficients")
    R.save_table(W.permutation_table(m, Xtr.iloc[va], ytr.iloc[va], n_repeats=3, top=15), f"{prefix}_permutation_importance")


def audit_extras(path=None):
    """Task 1 stages 4-11."""
    raw = D.load_attrition(path); dedup, _ = D.prepare_attrition(raw); clean = D.clean_raw(dedup)
    R.save_fig(W.plot_eda(raw), "eda_attrition_by_group")
    R.save_table(W.numeric_vs_target(raw, list(C.NUMERIC_CORE)).reset_index().rename(columns={"index": "column"}), "eda_numeric_vs_target")
    R.save_table(pd.concat([W.rate_by(raw, c).rename(columns={c: "group"}).astype({"group": str}).assign(variable=c) for c in ["department_id", "job_level", "work_mode", "employment_type"]]), "eda_rate_by_group")
    R.save_table(W.cleaning_report(raw, clean).reset_index().rename(columns={"index": "check"}), "cleaning_report")
    Xs, ys = F.make_xy(clean, False); Xtr, Xte, ytr, yte = M.make_split(Xs, ys); folds = M.make_folds(Xtr, ytr)
    oof_safe = cross_val_predict(MK(LRB(), False), Xtr, ytr, cv=folds, method="predict_proba")[:, 1]
    oof_dummy = cross_val_predict(MK(DummyClassifier(strategy="most_frequent"), False), Xtr, ytr, cv=folds, method="predict_proba")[:, 1]
    Zl = pd.DataFrame({"exit_date_filled": clean.loc[Xtr.index, "exit_date"].notna().astype(int).to_numpy(), "fnf_filled": clean.loc[Xtr.index, "fnf_settlement_amount"].notna().astype(int).to_numpy()})
    oof_leaky = cross_val_predict(LogisticRegression(max_iter=500), Zl, ytr.to_numpy(), cv=folds, method="predict_proba")[:, 1]
    R.save_table(pd.DataFrame({"Dummy (always stays)": _row(ytr, oof_dummy), "Safe logistic regression": _row(ytr, oof_safe), "LEAKY (reads exit columns)": _row(ytr, oof_leaky)}).T.reset_index().rename(columns={"index": "model"}), "audit_proof_models")
    R.save_fig(W.plot_confusion_grid(ytr, {"Dummy": oof_dummy, "Safe LR": oof_safe, "LEAKY demo": oof_leaky}), "audit_confusion_matrices")
    R.save_table(W.tune_grid(lambda C_: MK(LRB(C=C_), False), {f"C={c}": {"C_": c} for c in [0.01, 0.03, 0.1, 0.3, 1, 3, 10]}, Xtr, ytr, folds), "audit_tuning_lr_C")
    _interpret("audit_interpretation", Xtr, ytr, folds, False)


def pipeline_outputs(path=None):
    """Task 2: join report, 3-row smoke test, training-only proof, window sweep, with/without monthly features."""
    import joblib
    raw = D.load_attrition(path); att = D.load_sheet(C.SHEET_ATT, path); pay = D.load_sheet(C.SHEET_PAY, path)
    prepared, _ = D.prepare_attrition(raw); joined, rep = D.join_monthly_features(prepared, att, pay)
    from .runs import load_model_frame
    S = load_model_frame(True, path); Xtr, ytr, Xte, folds = S["Xtr"], S["ytr"], S["Xte"], S["folds"]
    R.save_json(S["join_report"], "join_report")
    R.save_table(pd.DataFrame(F.FEATURE_DOCS, columns=["feature", "source", "definition", "temporal_cutoff"]), "feature_definitions")
    matched = joined[joined.att_months_covered.notna()]
    R.save_table(W.numeric_vs_target(matched, C.MONTHLY_ALL).reset_index().rename(columns={"index": "feature"}), "eda_monthly_features_vs_target")
    pipe = F.build_pipeline(True).fit(Xtr, ytr); d = pipe.named_steps["derive"]
    raw3 = F.representative_rows(S["df"]); Z = pipe.transform(raw3[F.input_columns(True)])
    out = Z.copy(); out.insert(0, "row_id", raw3["row_id"].to_numpy()); R.save_table(out, "pipeline_3row_smoke_test_output")
    t = raw3[F.input_columns(True)].copy(); t.iloc[0, t.columns.get_loc("role_id")] = "R999"; t.iloc[1, t.columns.get_loc("department_id")] = np.nan; t.iloc[2, t.columns.get_loc("comp_ratio")] = np.nan
    checks = {"3_row_output_shape": str(Z.shape), "all_values_finite": bool(np.isfinite(Z.to_numpy()).all()),
              "target_in_output": any("left_within" in c for c in Z.columns), "unseen_category_and_missing_handled": bool(np.isfinite(pipe.transform(t).to_numpy()).all()),
              "leakage_column_in_output": any(any(b in c for b in C.LEAKAGE_CONFIRMED) for c in Z.columns),
              "dept_pay_groups_learned_from_training_rows": len(d.grp_median_), "n_training_rows_used_to_fit": len(Xtr), "n_test_rows_used_to_fit": 0,
              "global_median_pay_from_training_only": d.global_median_}
    R.save_json(checks, "pipeline_smoke_test_checks")
    joblib.dump(pipe, C.project_root() / "artifacts" / "models" / "feature_pipeline.joblib")
    R.save_json({"fitted_on": "training split only", "n_training_rows": len(Xtr), "input_columns": list(S["X"].columns), "output_feature_names": list(Z.columns), "versions": R.package_versions()}, "feature_pipeline")
    oof_full = cross_val_predict(MK(LRB(), True), Xtr, ytr, cv=folds, method="predict_proba")[:, 1]
    Xc, _ = F.make_xy(S["df"], False); oof_core = cross_val_predict(MK(LRB(), False), Xc.loc[Xtr.index], ytr, cv=folds, method="predict_proba")[:, 1]
    R.save_table(pd.DataFrame({"with attendance+payroll features": _row(ytr, oof_full), "attrition-table features only": _row(ytr, oof_core)}).T.reset_index().rename(columns={"index": "feature_set"}), "pipeline_feature_set_comparison")
    R.save_fig(W.plot_confusion_grid(ytr, {"Pipeline + balanced LR (OOF)": oof_full}), "pipeline_confusion_matrix")
    rows = []
    for w in [1, 3, 6]:
        jw, _ = D.join_monthly_features(prepared, att, pay, window=w); Xw, yw = F.make_xy(jw, True); idx = jw.att_months_covered.notna().to_numpy(); Xm, ym = Xw[idx], yw[idx]
        ff = list(StratifiedKFold(5, shuffle=True, random_state=C.SEED).split(Xm, ym)); s = cross_val_predict(MK(LRB(), True), Xm, ym, cv=ff, method="predict_proba")[:, 1]
        rows.append({"window_months": w, "rows": len(Xm), "leavers": int(ym.sum()), "cv_roc_auc": round(roc_auc_score(ym, s), 3)})
    R.save_table(pd.DataFrame(rows), "pipeline_tuning_window_months")
    fit_full = MK(LRB(), True).fit(Xtr, ytr); cf = W.logreg_coefficients(fit_full, 300); R.save_table(cf.head(20), "pipeline_coefficients")
    R.save_table(cf[cf.feature.isin(F.NUMERIC_DERIVED)], "pipeline_derived_feature_coefficients")
    dv = pipe.named_steps["derive"].transform(pipe.named_steps["clean"].transform(Xtr)); q = pd.qcut(dv.pay_vs_dept_median, 4, duplicates="drop")
    R.save_table(pd.crosstab(q.astype(str), ytr.to_numpy(), normalize="index").round(3).rename(columns={0: "stayed", 1: "left"}).reset_index(), "new_feature_pay_vs_dept_quartiles")
    return checks


def baseline_extras(o):
    """Task 3 stages 4, 9 (confusion matrices), 10 (tuning), 11 (interpretation)."""
    S = o["state"]; Xtr, ytr, folds, df = S["Xtr"], S["ytr"], S["folds"], S["df"]
    R.save_table(W.numeric_vs_target(df, [c for c in C.NUMERIC_CORE if c in df.columns]).reset_index().rename(columns={"index": "column"}), "eda_modeling_numeric_vs_target")
    R.save_fig(W.plot_confusion_grid(ytr, o["oof"]), "baseline_confusion_matrices")
    cm = []
    for n, s in o["oof"].items():
        r = E.binary_report(ytr, s); cm.append({"model": n, "tn": r["tn"], "fp": r["fp"], "fn": r["fn"], "tp": r["tp"]})
    R.save_table(pd.DataFrame(cm), "baseline_confusion_counts")
    t_lr = W.tune_grid(lambda C_: MK(LRB(C=C_), True), {f"LR C={c}": {"C_": c} for c in [0.01, 0.03, 0.1, 0.3, 1, 3]}, Xtr, ytr, folds)
    t_tr = W.tune_grid(lambda d: MK(DecisionTreeClassifier(max_depth=d, class_weight="balanced", random_state=C.SEED), True), {f"Tree depth={d}": {"d": d} for d in [2, 3, 4, 5, 6, 8]}, Xtr, ytr, folds)
    R.save_table(pd.concat([t_lr, t_tr]).sort_values("cv_roc_auc", ascending=False).reset_index(drop=True), "baseline_tuning")
    _interpret("baseline_interpretation", Xtr, ytr, folds, True)


def evaluation_extras(ev):
    """Task 4 stages 4, 9 (confusion matrices), 10 (threshold + calibrator choice), 11 (band profile, interpretation)."""
    S = ev["state"]; Xtr, ytr, Xte, yte, folds = S["Xtr"], S["ytr"], S["Xte"], S["yte"], S["folds"]; oof, p = ev["oof"], ev["p_test"]
    R.save_fig(W.plot_confusion_grid(ytr, {"Train out-of-fold": oof}), "evaluation_confusion_train_oof")
    R.save_fig(W.plot_confusion_grid(yte, {"Held-out test": p}), "evaluation_confusion_test")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for v, lab in [(0, "stayed"), (1, "left")]: ax.hist(oof[ytr.to_numpy() == v], bins=30, alpha=.55, density=True, label=lab)
    ax.set_xlabel("out-of-fold score"); ax.legend(); ax.set_title("Score distribution by actual outcome"); R.save_fig(fig, "evaluation_score_distribution")
    ths = np.round(np.arange(0.2, 0.81, 0.05), 2)
    tt = pd.DataFrame([{"threshold": t, **{k: _row(ytr, oof, t)[k] for k in ("precision", "recall", "f1")}, "flagged_pct": round(100 * (oof >= t).mean(), 1)} for t in ths])
    R.save_table(tt, "evaluation_threshold_sweep_train_oof"); best = float(tt.loc[tt.f1.idxmax(), "threshold"])
    R.save_table(pd.DataFrame([{"chosen_threshold_from_train_oof": best, **_row(yte, p, best)}]), "evaluation_threshold_applied_to_test")
    platt, iso = np.zeros(len(oof)), np.zeros(len(oof))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=C.SEED).split(oof, ytr):
        platt[b] = E.Recalibrator().fit(oof[a], ytr.iloc[a]).predict(oof[b]); iso[b] = IsotonicRegression(out_of_bounds="clip").fit(oof[a], ytr.iloc[a]).predict(oof[b])
    R.save_table(pd.DataFrame({"calibrator": ["raw scores", "Platt (chosen)", "Isotonic"], "cv_brier_on_train_oof": [brier_score_loss(ytr, oof), brier_score_loss(ytr, platt), brier_score_loss(ytr, iso)]}).round(4), "evaluation_calibrator_comparison")
    bands = ev["bands"]["Held-out test"].to_numpy()
    prof = Xte[["tenure_months", "comp_ratio", "months_since_last_raise", "last_engagement_score", "last_performance_rating", "attendance_rate_3m"]].groupby(bands).mean().reindex(["High", "Medium", "Low"])
    R.save_table(prof.round(2).reset_index().rename(columns={"index": "band"}), "evaluation_band_profile")
    R.save_table(W.logreg_coefficients(ev["final_model"], 20), "evaluation_coefficients")
    tr, va = folds[0]; m = MK(LRB(), True).fit(Xtr.iloc[tr], ytr.iloc[tr]); R.save_table(W.permutation_table(m, Xtr.iloc[va], ytr.iloc[va], n_repeats=3, top=15), "evaluation_permutation_importance")


def attendance_extras(o, path=None):
    """Task 5 stages 3, 4, 7, 10, 11 + confusion matrices."""
    from . import attendance as T
    from sklearn.ensemble import IsolationForest
    att = D.load_sheet(C.SHEET_ATT, path); daily = D.load_sheet("attendance_daily_examples", path); d = T.add_history_features(att); feats = T.RAW_FEATURES + T.HISTORY_FEATURES
    audit = {"missing values": int(att.isna().sum().sum()), "duplicate (employee, month)": int(att.duplicated(["employee_id", "month"]).sum()),
             "present_days > scheduled_days": int((att.present_days > att.scheduled_days).sum()), "longest streak > unpaid absence days": int((att.longest_unpaid_streak_days > att.unpaid_absence_days).sum()),
             "wfh_days > present_days": int((att.wfh_days > att.present_days).sum()), "late_arrivals > present_days": int((att.late_arrivals > att.present_days).sum()),
             "avg_hours_worked > 12": int((att.avg_hours_worked > 12).sum()), "avg_hours < 3 while present": int(((att.avg_hours_worked < 3) & (att.present_days > 0)).sum()), "zero present days": int((att.present_days == 0).sum())}
    R.save_table(pd.Series(audit, name="rows").rename_axis("check").reset_index(), "attendance_audit_checks")
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6)); att.groupby("month").attendance_rate.mean().plot(ax=ax[0], title="mean attendance rate by month")
    ax[1].hist(att.avg_hours_worked, bins=50); ax[1].set_title("avg_hours_worked"); ax[2].hist(att.late_arrivals, bins=range(0, int(att.late_arrivals.max()) + 2)); ax[2].set_title("late_arrivals"); fig.tight_layout(); R.save_fig(fig, "attendance_eda")
    months = sorted(d.month.unique()); cut = months[11]; fit_df, later = d[d.month <= cut], d[d.month > cut]
    iso_t = IsolationForest(n_estimators=300, contamination=0.03, random_state=C.SEED).fit(fit_df[feats]); thr = np.quantile(-iso_t.score_samples(fit_df[feats]), 0.97)
    R.save_table(pd.DataFrame([{"fit_months": f"{fit_df.month.min().date()} to {cut.date()}", "scored_months": f"{later.month.min().date()} to {later.month.max().date()}",
                                "flag_rate_pct_on_later_months": round(100 * (-iso_t.score_samples(later[feats]) > thr).mean(), 2)}]), "attendance_temporal_split_stability")
    rows = []
    for c in [0.01, 0.02, 0.03, 0.05]:
        f = IsolationForest(n_estimators=300, contamination=c, random_state=C.SEED).fit(d[feats]); dpc = T.daily_proxy_eval(daily, contamination=c).set_index("detector").loc["Daily IsolationForest"]
        rows.append({"contamination": c, "monthly_rows_flagged_by_model": int((f.predict(d[feats]) == -1).sum()), "daily_sample_precision": round(dpc.precision, 3), "daily_sample_recall": round(dpc.recall, 3)})
    R.save_table(pd.DataFrame(rows), "attendance_tuning_contamination")
    fl = o["flags"].set_index("attendance_feature_id").flagged; dd = d.set_index("attendance_feature_id").assign(flagged=fl)
    comp = dd.groupby("flagged")[feats].mean().T.rename(columns={False: "normal", True: "flagged"}); comp["ratio"] = (comp.flagged / comp.normal.replace(0, np.nan)).round(2)
    R.save_table(comp.round(3).sort_values("ratio", key=lambda s: (s - 1).abs(), ascending=False).reset_index().rename(columns={"index": "feature"}), "attendance_interpretation_flagged_vs_normal")
    dp = o["daily_proxy"].set_index("detector").loc["Daily combined"]; tp, fp, fn = int(dp.caught), int(dp.flagged - dp.caught), int(dp.injected - dp.caught)
    R.save_fig(W.plot_confusion_counts(len(daily) - tp - fp - fn, fp, fn, tp, "Daily sample: combined detector"), "attendance_confusion_daily_sample")
    if o.get("measured") is not None:
        lab = T.load_labels(o["labels_path"]); key = "attendance_feature_id"; truth = d[key].isin(set(lab[key])); f = o["flags"].flagged.to_numpy(); tr = truth.to_numpy()
        R.save_fig(W.plot_confusion_counts(int((~f & ~tr).sum()), int((f & ~tr).sum()), int((~f & tr).sum()), int((f & tr).sum()), "Monthly combined detector vs hidden labels"), "attendance_confusion_monthly")
