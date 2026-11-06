"""Task 13 end-to-end: tune, compare on the same folds, calibrate, evaluate once on the test set, save versioned artifacts."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from cortexhr import config as C
from . import v2 as V
from ..common.config import load_config
from ..common.io import read_csv, save_json, save_table, utc_now
from ..common.paths import MODELS, METADATA, VAULT, task_dir

MODEL_VERSION = "attrition-v2"


def vault_attributes() -> pd.DataFrame | None:
    p = VAULT / "unlock_CX-15_protected_attributes.csv"
    return pd.read_csv(p) if p.exists() else None


def group_report(df: pd.DataFrame, group_col: str, score="p", y="y", flag_frac=0.10) -> pd.DataFrame:
    """Per-group base rate, mean predicted risk, share flagged in the top 10%, recall of leavers and AUC (measurement only)."""
    cut = df[score].quantile(1 - flag_frac); rows = []
    for g, d in df.groupby(group_col):
        auc = roc_auc_score(d[y], d[score]) if d[y].nunique() == 2 and len(d) >= 30 else np.nan
        flagged = d[score] >= cut
        rows.append({group_col: g, "n": len(d), "actual_leave_rate": d[y].mean(), "mean_predicted_risk": d[score].mean(),
                     "flagged_top10_share": flagged.mean(), "recall_of_leavers_in_top10": (flagged & (d[y] == 1)).sum() / max(1, (d[y] == 1).sum()), "auc": auc})
    return pd.DataFrame(rows)



def run_lightgbm_audit(write: bool = True) -> dict:
    """Focused official-booster audit using the exact Task 3 folds, without rerunning HGB tuning."""
    if not V.optional_libs().get("lightgbm"):
        raise RuntimeError("LightGBM is not installed; install requirements-optional.txt or lightgbm before running this audit")
    d = V.load_training(); X, y, folds, Xte, yte = d["Xtr"], d["ytr"], d["folds"], d["Xte"], d["yte"]
    rows, fold_tables = [], []
    for i, params in enumerate(V.lightgbm_tuning_grid(), 1):
        name = f"LightGBM trial {i}"; ft = V.fold_table(V.make_lightgbm(**params), X, y, folds, name)
        fold_tables.append(ft)
        rows.append({"experiment_id": name, "model": "LightGBM", "params": json.dumps(params, sort_keys=True),
                     "validation": "same five Task 3 folds; training portion only",
                     "roc_auc_mean": ft.roc_auc.mean(), "roc_auc_sd": ft.roc_auc.std(ddof=1),
                     "pr_auc_mean": ft.pr_auc.mean(), "pr_auc_sd": ft.pr_auc.std(ddof=1), "brier_mean": ft.brier.mean()})
    tuning = pd.DataFrame(rows).sort_values(["pr_auc_mean", "roc_auc_mean"], ascending=False).reset_index(drop=True)
    tuning["decision"] = np.where(tuning.index == 0, "selected for independent test evaluation", "not selected")
    tuning["reason"] = np.where(tuning.index == 0, "highest mean CV PR-AUC among the predeclared candidates", "lower mean CV PR-AUC than selected candidate")
    params = json.loads(tuning.iloc[0].params); candidate = V.make_lightgbm(**params)
    model = CalibratedClassifierCV(clone(candidate), method="sigmoid", cv=3, ensemble=False).fit(X, y)
    p = model.predict_proba(Xte)[:, 1]
    from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
    test = pd.DataFrame([{"model": "LightGBM tuned (sigmoid calibrated)", "roc_auc": roc_auc_score(yte, p),
        "pr_auc": average_precision_score(yte, p), "brier": brier_score_loss(yte, p),
        **V.threshold_metrics(yte, p, 0.20), **V.top_fraction_metrics(yte, p)}])
    folds_out = pd.concat(fold_tables, ignore_index=True); summary = V.summarise(folds_out)
    result = {"tuning_log": tuning, "fold_metrics": folds_out, "cv_summary": summary, "test_metrics": test, "model": model, "params": params}
    if write:
        td = task_dir(13); MODELS.mkdir(parents=True, exist_ok=True); METADATA.mkdir(parents=True, exist_ok=True)
        artifact = MODELS / "attrition_lightgbm_candidate.joblib"; joblib.dump(model, artifact)
        meta = {"model_version": "attrition-lightgbm-candidate-v1", "model_family": "LightGBMClassifier + sigmoid calibration",
                "params": params, "lightgbm_version": V.optional_libs()["lightgbm"], "seed": C.SEED,
                "cv": "same five stratified Task 3 folds on training partition", "train_rows": len(X), "test_rows": len(Xte),
                "features": list(X.columns), "artifact_sha256": V.file_sha256(artifact),
                "test_metrics": test.round(5).to_dict("records"),
                "note": "Independent candidate artifact for evaluation; not silently substituted for the selected production attrition-v2 model."}
        save_json(meta, METADATA / "attrition_lightgbm_candidate_metadata.json")
        save_table(tuning, td / "lightgbm_tuning_log.csv"); save_table(folds_out, td / "lightgbm_fold_metrics.csv")
        save_table(summary, td / "lightgbm_cv_summary.csv"); save_table(test, td / "lightgbm_test_metrics.csv")
        baseline_cv_path, baseline_test_path = td / "cv_summary.csv", td / "test_metrics.csv"
        if baseline_cv_path.exists():
            baseline_cv = pd.read_csv(baseline_cv_path)
            save_table(pd.concat([baseline_cv, summary], ignore_index=True, sort=False), td / "cv_summary_with_lightgbm.csv")
        if baseline_test_path.exists():
            baseline_test = pd.read_csv(baseline_test_path)
            save_table(pd.concat([baseline_test, test], ignore_index=True, sort=False), td / "test_metrics_with_lightgbm.csv")
    return result

def run_task13(write: bool = True, verbose: bool = False) -> dict:
    out: dict = {}
    d = V.load_training(); X, y, folds, Xte, yte = d["Xtr"], d["ytr"], d["folds"], d["Xte"], d["yte"]; out["frame"] = d
    bench = read_csv("baseline_benchmarks"); out["benchmarks"] = bench
    # 1) tuning (test set untouched)
    log, tune_folds = V.run_tuning(X, y, folds); params = V.best_params(log); out["tuning_log"], out["best_params"] = log, params
    # 2) same-fold comparison
    lr = V.baseline_models()["Logistic Regression (balanced)"]; rf = V.baseline_models()["Random Forest (300, balanced)"]
    cands = {"Logistic Regression (balanced) [baseline]": lr, "Random Forest (300, balanced) [baseline]": rf,
             "HistGradientBoosting default": V.wrap(V.make_hgb()), "HistGradientBoosting tuned": V.wrap(V.make_hgb(**params)),
             "attrition-v2 candidate: LR + tuned HGB blend": V.make_v2(params)}
    ft = pd.concat([V.fold_table(m, X, y, folds, n) for n, m in cands.items()], ignore_index=True); out["fold_metrics"] = ft
    summ = V.summarise(ft); out["summary"] = summ
    base, v2n = "Logistic Regression (balanced) [baseline]", "attrition-v2 candidate: LR + tuned HGB blend"
    out["paired"] = pd.DataFrame([V.paired_difference(ft, a, base, m) for a in (v2n, "HistGradientBoosting tuned") for m in ("roc_auc", "pr_auc")])
    pv = V.paired_difference(ft, v2n, base, "roc_auc"); out["verdict"] = V.verdict(pv)
    sel = V.choose_simplest_within(summ, ["HistGradientBoosting tuned", v2n]) if False else summ.set_index("model").loc[["HistGradientBoosting tuned", v2n]]["pr_auc_mean"].idxmax()
    out["selected_candidate"] = sel
    # 3) calibration, chosen by out-of-fold Brier score
    model = V.make_v2(params); cal_rows = []; oof = {}
    for m in load_config()["attrition"]["calibration_methods"]:
        p = V.calibrated_oof(model, X, y, folds, m); oof[m] = p
        cal_rows.append({"method": m, "brier": brier_score_loss(y, p), "ece": V.expected_calibration_error(y, p), "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p), "mean_probability": p.mean(), "base_rate": y.mean()})
    cal = pd.DataFrame(cal_rows); out["calibration"] = cal; chosen = cal.sort_values("brier").iloc[0]["method"]; out["calibration_method"] = chosen; out["oof"] = oof
    # 4) one-time test evaluation (models fit on the training part only)
    v2_cal = CalibratedClassifierCV(clone(model), method=chosen, cv=3, ensemble=False).fit(X, y)
    lr_cal = CalibratedClassifierCV(clone(lr), method=chosen, cv=3, ensemble=False).fit(X, y)
    p_v2, p_lr = v2_cal.predict_proba(Xte)[:, 1], lr_cal.predict_proba(Xte)[:, 1]
    rows = []
    for name, p in (("Logistic Regression baseline (calibrated)", p_lr), ("attrition-v2 (calibrated)", p_v2)):
        rows.append({"model": name, "roc_auc": roc_auc_score(yte, p), "pr_auc": average_precision_score(yte, p), "brier": brier_score_loss(yte, p), "ece": V.expected_calibration_error(yte, p),
                     **V.threshold_metrics(yte, p, 0.20), **V.top_fraction_metrics(yte, p)})
    out["test_metrics"] = pd.DataFrame(rows); out["test_boot"] = V.bootstrap_oof_auc_diff(yte, p_v2, p_lr)
    out["oof_boot"] = V.bootstrap_oof_auc_diff(y, oof[chosen], V.calibrated_oof(lr, X, y, folds, chosen))
    out["test_scores"] = pd.DataFrame({"y": yte.to_numpy(), "p_v2": p_v2, "p_lr": p_lr}, index=Xte.index)
    # 5) group measurement (restricted file, measurement only; never a model input)
    attrs = vault_attributes()
    if attrs is not None:
        a = attrs[attrs.record_type == "attrition_snapshot"].drop_duplicates("person_ref").set_index("person_ref")
        allp = pd.concat([pd.DataFrame({"p": oof[chosen], "y": y.to_numpy()}, index=X.index), pd.DataFrame({"p": p_v2, "y": yte.to_numpy()}, index=Xte.index)]).sort_index()
        allp["employee_id"] = d["df"].loc[allp.index, "employee_id"].to_numpy(); allp = allp.join(a[["gender", "age_band"]], on="employee_id")
        out["group_gender"], out["group_age"] = group_report(allp.dropna(subset=["gender"]), "gender"), group_report(allp.dropna(subset=["age_band"]), "age_band")
        out["group_frame"] = allp
    # 6) save artifacts (never overwrite Phase 1/2 models: different folder and name)
    if write:
        full_X, full_y = pd.concat([X, Xte]), pd.concat([y, yte])
        prod = CalibratedClassifierCV(clone(model), method=chosen, cv=5, ensemble=False).fit(full_X, full_y)
        MODELS.mkdir(parents=True, exist_ok=True)
        joblib.dump(v2_cal, MODELS / "attrition_v2_train_only.joblib"); joblib.dump(prod, MODELS / "attrition_v2.joblib")
        meta = {"model_version": MODEL_VERSION, "created_utc": utc_now(), "model_family": "LogisticRegression(balanced) + HistGradientBoosting (equal-weight probability blend), calibrated",
                "calibration_method": chosen, "hgb_params": params, "seed": C.SEED, "cv": "StratifiedKFold(5, shuffle, seed 42) on the 80% training part (identical to Task 3)",
                "train_rows": int(len(X)), "test_rows": int(len(Xte)), "production_fit_rows": int(len(full_X)), "features_raw": list(X.columns),
                "preprocessing": "cortexhr.features.build_pipeline(include_monthly=True) refit inside every fold and inside the final fit",
                "excluded_as_inputs": list(C.FORBIDDEN_AS_INPUT), "cv_summary": summ.set_index("model").loc[[base, v2n]].round(4).to_dict("index"),
                "test_metrics": out["test_metrics"].round(4).to_dict("records"), "verdict": out["verdict"], "environment": V.environment(),
                "artifact_sha256": {"attrition_v2.joblib": V.file_sha256(MODELS / "attrition_v2.joblib"), "attrition_v2_train_only.joblib": V.file_sha256(MODELS / "attrition_v2_train_only.joblib")},
                "note": "Test-set metrics come from the model fitted on the training part only; attrition_v2.joblib is refit on all labelled rows for scoring."}
        save_json(meta, METADATA / "attrition_v2_metadata.json"); out["metadata"] = meta
        td = task_dir(13)
        for nm, df in (("tuning_log", log), ("fold_metrics", ft), ("cv_summary", summ), ("paired_differences", out["paired"]), ("calibration_comparison", cal), ("test_metrics", out["test_metrics"])):
            save_table(df, td / f"{nm}.csv")
        if attrs is not None:
            save_table(out["group_gender"], td / "group_gender.csv"); save_table(out["group_age"], td / "group_age.csv")
    return out
