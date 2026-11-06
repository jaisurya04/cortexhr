"""Task 15 end-to-end: baselines, linear and gradient-boosting models, bands, group gaps, artifacts."""
from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from . import model as S
from ..common.config import load_config
from ..common.io import save_json, save_table, utc_now
from ..common.paths import METADATA, MODELS, task_dir
from ..attrition.v2 import environment, file_sha256
from ..explanations.risk_report import load_helpers

MODEL_VERSION = "salary-band-v1"


def hgb_grid() -> list[dict]:
    return [dict(learning_rate=lr, max_iter=it, max_depth=d, min_samples_leaf=msl) for lr, it, d, msl in
            ((0.05, 200, 2, 20), (0.05, 300, 3, 20), (0.03, 400, 3, 40), (0.08, 150, 3, 20), (0.05, 300, 4, 30), (0.1, 100, 2, 20))]


def group_table(df: pd.DataFrame, group_col: str, cols: dict, helper, min_n: int) -> pd.DataFrame:
    """Mean of each column per group, plus the supplied group_gap() (best minus worst group mean) for each column."""
    rows = []
    sizes = df.groupby(group_col).size(); ok = sizes[sizes >= min_n].index; d = df[df[group_col].isin(ok)]
    for name, col in cols.items():
        means, gap = helper.group_gap(d[col].to_numpy(), d[group_col].to_numpy()); rows.append({"measure": name, "group_gap": gap, **{f"mean[{k}]": v for k, v in means.items()}})
    return pd.DataFrame(rows)


def bootstrap_diff(df, group_col, a, b, col, n_boot=2000, seed=42) -> dict:
    rng = np.random.default_rng(seed); x, y = df.loc[df[group_col] == a, col].to_numpy(), df.loc[df[group_col] == b, col].to_numpy()
    d = [rng.choice(x, len(x)).mean() - rng.choice(y, len(y)).mean() for _ in range(n_boot)]
    return {"difference": float(x.mean() - y.mean()), "ci_low": float(np.percentile(d, 2.5)), "ci_high": float(np.percentile(d, 97.5))}


def run_task15(write: bool = True) -> dict:
    cfg = load_config(); out: dict = {}; seed = cfg["seed"]; min_n = cfg["salary"]["min_group_size"]
    df = S.load_salary(); tr, te = S.split(df); out.update(df=df, train=tr, test=te)
    y_tr, y_te = tr[S.TARGET].to_numpy(float), te[S.TARGET].to_numpy(float)
    # ---- baselines and candidates, 5-fold out-of-fold predictions on the training part
    def oof_baseline(adjust):
        pred = np.zeros(len(tr))
        for a, b in S.folds(len(tr)):
            m = S.RoleMeanBaseline(adjust).fit(tr.iloc[a], y_tr[a]); pred[b] = m.predict(tr.iloc[b])
        return pred
    oof = {"baseline: average offered salary by role": oof_baseline(False), "baseline 2: role average adjusted for market index": oof_baseline(True)}
    gl = {"ridge": lambda: S.ridge(S.JOB_FEATURES), "ridge_with_gap": lambda: S.ridge(S.WITH_GAP_FEATURES)}
    oof["Ridge regression (job-related features)"] = S.oof_predict(gl["ridge"], S.JOB_FEATURES, tr)
    oof["Ridge regression + career_gap_months"] = S.oof_predict(gl["ridge_with_gap"], S.WITH_GAP_FEATURES, tr)
    # ---- gradient boosting: small tuned grid, selected by cross-validated MAE (training part only)
    tune = []
    for i, p in enumerate(hgb_grid()):
        o = S.oof_predict(lambda p=p: S.hgb(S.JOB_FEATURES, **p), S.JOB_FEATURES, tr); tune.append({"experiment_id": f"S{i:02d}", "params": str(p), **S.metrics(y_tr, o)}); oof[f"__hgb{i}"] = o
    tl = pd.DataFrame(tune); best = int(tl.MAE_rupees.idxmin()); best_p = hgb_grid()[best]; tl["decision"] = np.where(tl.index == best, "selected (lowest CV MAE)", "not selected"); out["tuning_log"] = tl
    oof["Gradient boosting (job-related features, tuned)"] = oof.pop(f"__hgb{best}"); [oof.pop(k) for k in [k for k in oof if k.startswith("__hgb")]]
    oof["Gradient boosting + career_gap_months"] = S.oof_predict(lambda: S.hgb(S.WITH_GAP_FEATURES, **best_p), S.WITH_GAP_FEATURES, tr)
    out["oof_train"] = oof
    out["cv_metrics"] = pd.DataFrame({k: S.metrics(y_tr, v) for k, v in oof.items()}).T
    # ---- final choice among job-related models (simpler wins unless the other is clearly better: > 2% lower MAE)
    r, g = out["cv_metrics"].loc["Ridge regression (job-related features)"], out["cv_metrics"].loc["Gradient boosting (job-related features, tuned)"]
    final_name = "Gradient boosting (job-related features, tuned)" if g.MAE_rupees < 0.98 * r.MAE_rupees else "Ridge regression (job-related features)"; out["final_model"] = final_name
    makers = {"Ridge regression (job-related features)": (lambda: S.ridge(S.JOB_FEATURES), S.JOB_FEATURES), "Gradient boosting (job-related features, tuned)": (lambda: S.hgb(S.JOB_FEATURES, **best_p), S.JOB_FEATURES),
              "Ridge regression + career_gap_months": (lambda: S.ridge(S.WITH_GAP_FEATURES), S.WITH_GAP_FEATURES), "Gradient boosting + career_gap_months": (lambda: S.hgb(S.WITH_GAP_FEATURES, **best_p), S.WITH_GAP_FEATURES)}
    fitted = {k: m().fit(tr[f], np.log(y_tr)) for k, (m, f) in makers.items()}; out["fitted"] = fitted
    base1 = S.RoleMeanBaseline(False).fit(tr, y_tr); base2 = S.RoleMeanBaseline(True).fit(tr, y_tr)
    tp = {"baseline: average offered salary by role": base1.predict(te), "baseline 2: role average adjusted for market index": base2.predict(te)}
    for k, (m, f) in makers.items(): tp[k] = np.exp(fitted[k].predict(te[f]))
    out["test_pred"] = tp; out["test_metrics"] = pd.DataFrame({k: S.metrics(y_te, v) for k, v in tp.items()}).T
    # ---- recommended band: spread of out-of-fold log error of the final model
    mult = S.band_quantiles(y_tr, oof[final_name]); out["band_multipliers"] = mult
    bands = S.recommend_band(fitted[final_name], te, mult); bands["actual_offer"] = y_te; bands["inside_band"] = (y_te >= bands.recommended_low) & (y_te <= bands.recommended_high)
    out["bands_test"] = pd.concat([te[["hire_id", "role_id", "job_level", "hire_year"]], bands], axis=1); out["band_coverage"] = float(bands.inside_band.mean())
    # ---- fairness: out-of-fold predictions for training rows + train-fit predictions for test rows, so every row has an out-of-sample prediction
    va = S.vault_gender(); out["vault"] = va is not None
    if va is not None:
        H = load_helpers(); full = pd.concat([tr.assign(_part="train"), te.assign(_part="test")], ignore_index=True)
        for name, key in (("A: with career_gap_months", "Gradient boosting + career_gap_months" if final_name.startswith("Gradient") else "Ridge regression + career_gap_months"),
                          ("B: job-related only (chosen)", final_name)):
            full[f"pred_{name[0]}"] = np.concatenate([oof[key], tp[key]])
        norm = full.band_mid_2026 * full.market_index
        for c in ("A", "B"): full[f"ratio_pred_{c}"] = full[f"pred_{c}"] / norm; full[f"resid_ratio_{c}"] = full[S.TARGET] / full[f"pred_{c}"]
        full["ratio_actual"] = full[S.TARGET] / norm; full = full.join(va, on="hire_id"); full["gap_bin"] = pd.cut(full.career_gap_months, [-1, 0, 6, 12, 24, 10 ** 4], labels=["none", "1-6 months", "7-12 months", "13-24 months", "over 24 months"])
        out["full"] = full
        cols = {"actual offer / band mid-point (history)": "ratio_actual", "model A predicted ratio": "ratio_pred_A", "model B predicted ratio (chosen)": "ratio_pred_B", "model A actual/predicted": "resid_ratio_A", "model B actual/predicted": "resid_ratio_B"}
        out["gender_gaps"], out["gap_bin_gaps"], out["age_gaps"] = (group_table(full.dropna(subset=[c]), c, cols, H, min_n) for c in ("gender", "gap_bin", "age_band"))
        out["gender_means"] = full.groupby("gender")[list(cols.values())].mean().rename(columns={v: k for k, v in cols.items()}); out["gap_bin_means"] = full.groupby("gap_bin", observed=True)[list(cols.values())].mean().rename(columns={v: k for k, v in cols.items()})
        out["gender_boot"] = {c: bootstrap_diff(full, "gender", "Female", "Male", c) for c in ("ratio_actual", "ratio_pred_A", "ratio_pred_B")}
        out["gender_vs_gap"] = full.groupby("gender").career_gap_months.agg(["mean", "median", lambda s: (s > 0).mean()]).rename(columns={"<lambda_0>": "share_with_gap"})
    if write:
        MODELS.mkdir(parents=True, exist_ok=True); full_fit = makers[final_name][0]().fit(df[makers[final_name][1]], np.log(df[S.TARGET].to_numpy(float)))
        path = MODELS / "salary_band_v1.joblib"; joblib.dump({"model": full_fit, "features": makers[final_name][1], "band_multipliers": mult, "version": MODEL_VERSION}, path)
        meta = {"model_version": MODEL_VERSION, "created_utc": utc_now(), "model": final_name, "features": makers[final_name][1], "never_inputs": S.NEVER_INPUTS, "excluded_job_unrelated": ["career_gap_months"],
                "band_multipliers_log": mult, "band_coverage_test": out["band_coverage"], "seed": seed, "train_rows": len(tr), "test_rows": len(te), "production_fit_rows": len(df), "environment": environment(),
                "cv_metrics": out["cv_metrics"].round(4).to_dict("index"), "test_metrics": out["test_metrics"].round(4).to_dict("index"), "sha256": file_sha256(path),
                "note": "Advice for a manager and HR, never an offer. The target is historical offers and may contain past bias."}
        save_json(meta, METADATA / "salary_band_v1_metadata.json"); out["metadata"] = meta; td = task_dir(15)
        save_table(out["cv_metrics"].reset_index(names="model"), td / "cv_metrics.csv"); save_table(out["test_metrics"].reset_index(names="model"), td / "test_metrics.csv"); save_table(tl, td / "tuning_log.csv"); save_table(out["bands_test"], td / "band_recommendations_test.csv")
        if va is not None:
            for k in ("gender_gaps", "gap_bin_gaps", "age_gaps"): save_table(out[k], td / f"{k}.csv")
            save_table(out["gender_means"].reset_index(), td / "gender_means.csv"); save_table(out["gap_bin_means"].reset_index(), td / "career_gap_means.csv")
    return out
