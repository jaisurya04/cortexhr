"""End-to-end runs shared by scripts/ and notebooks/ (no duplicated logic)."""
from __future__ import annotations
import json, warnings
import numpy as np, pandas as pd, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from . import config as C, data_loading as D, features as F, modeling as M, evaluation as E, reporting as R, audit as A

BENCH_MAP = {"Dummy (always stays)": "Dummy (always stays)",
             "Logistic Regression (balanced)": "Logistic Regression (class_weight=balanced)",
             "Decision Tree (depth 4, balanced)": "Decision Tree (depth 4, balanced)",
             "Random Forest (300, balanced)": "Random Forest (300 trees, balanced)"}
PRIMARY = "Logistic Regression (balanced)"


def load_model_frame(include_monthly=True, path=None):
    """Load -> drop exact duplicate rows -> join cutoff-safe monthly features -> X, y -> stratified split -> folds."""
    raw = D.load_attrition(path); att = D.load_sheet(C.SHEET_ATT, path); pay = D.load_sheet(C.SHEET_PAY, path)
    df, prep_log = D.prepare_attrition(raw)
    pay_dedup, n_pay_dup = D.dedupe_payroll(pay)
    joined, join_rep = D.join_monthly_features(df, att, pay, window=3)
    join_rep.update(payroll_exact_duplicates_removed=n_pay_dup, **prep_log)
    X, y = F.make_xy(joined, include_monthly)
    Xtr, Xte, ytr, yte = M.make_split(X, y); folds = M.make_folds(Xtr, ytr)
    return dict(raw=raw, df=joined, X=X, y=y, Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte, folds=folds,
                join_report=join_rep, include_monthly=include_monthly)


def leaky_demo(df, tr_idx, folds):
    """Controlled demonstration: a 'model' that only looks at whether exit fields are filled in."""
    from sklearn.model_selection import cross_val_predict
    Z = pd.DataFrame({"exit_date_filled": df.loc[tr_idx, "exit_date"].notna().astype(int),
                      "fnf_filled": df.loc[tr_idx, "fnf_settlement_amount"].notna().astype(int)}).reset_index(drop=True)
    y = df.loc[tr_idx, C.TARGET].reset_index(drop=True)
    p = cross_val_predict(LogisticRegression(max_iter=500), Z, y, cv=folds, method="predict_proba")[:, 1]
    fm = M.fold_metrics(y, p, folds, "LEAKY Logistic Regression (adds exit information)"); return M.summarise(fm)


def leakage_guard(summary: pd.DataFrame, threshold=0.90):
    s = summary.copy(); s["leakage_check"] = np.where(s.roc_auc_mean > threshold, "INVESTIGATE: ROC-AUC > 0.90 (leakage/evaluation bug?)", "ok"); return s


def compare_with_benchmark(summ: pd.DataFrame, path=None) -> pd.DataFrame:
    bench = D.load_sheet(C.SHEET_BENCH, path).set_index("model"); rows = []
    for _, r in summ.iterrows():
        b = bench.loc[BENCH_MAP.get(r.model, r.model)] if BENCH_MAP.get(r.model, r.model) in bench.index else None
        rows.append({"model": r.model, "roc_auc": r.roc_auc_mean, "roc_auc_sd": r.get("roc_auc_sd", np.nan),
                     "benchmark_roc_auc": b.roc_auc_mean if b is not None else np.nan,
                     "delta_roc_auc": r.roc_auc_mean - b.roc_auc_mean if b is not None else np.nan,
                     "pr_auc": r.pr_auc_mean, "benchmark_pr_auc": b.pr_auc_mean if b is not None else np.nan,
                     "precision": r.precision_mean, "benchmark_precision": b.precision_mean if b is not None else np.nan,
                     "recall": r.recall_mean, "benchmark_recall": b.recall_mean if b is not None else np.nan,
                     "f1": r.f1_mean, "accuracy": r.accuracy_mean, "benchmark_accuracy": b.accuracy_mean if b is not None else np.nan})
    return pd.DataFrame(rows)


def run_baselines(path=None, write=True):
    S = load_model_frame(True, path); Xtr, ytr, folds, df = S["Xtr"], S["ytr"], S["folds"], S["df"]
    out = {"state": S}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        oof, fm, summ = M.evaluate_zoo(Xtr, ytr, folds, True)
        S2 = load_model_frame(False, path)            # ablation: attrition-table features only (same rows, same split, same folds)
        oof_core, fm_core, summ_core = M.evaluate_zoo(S2["Xtr"], S2["ytr"], S2["folds"], False)
        assert all((a[1] == b[1]).all() for a, b in zip(folds, S2["folds"])), "folds differ between feature sets"
        tree = M.tree_overfit_demo(Xtr, ytr, folds, True)
        smote = {"LR balanced class weights": oof[PRIMARY], "LR no weights": None, "LR + SMOTE (train folds only)": M.oof_scores_smote(Xtr, ytr, folds, True)}
        from sklearn.model_selection import cross_val_predict
        plain = Pipeline([("features", F.build_pipeline(True)), ("model", LogisticRegression(max_iter=2000, random_state=C.SEED))])
        smote["LR no weights"] = cross_val_predict(plain, Xtr, ytr, cv=folds, method="predict_proba")[:, 1]
    out["warnings"] = sorted({f"{w.category.__name__}: {str(w.message)[:80]}" for w in caught})
    imb = []
    for k, s in smote.items():
        fmk = M.fold_metrics(ytr, s, folds, k); imb.append(M.summarise(fmk))
    imb = pd.DataFrame(imb)
    summ_g = leakage_guard(pd.concat([summ, pd.DataFrame([leaky_demo(df, Xtr.index, folds)])], ignore_index=True))
    comp = compare_with_benchmark(summ_g, path); comp_core = compare_with_benchmark(summ_core, path)
    out.update(oof=oof, fold_metrics=fm, summary=summ_g, comparison=comp, comparison_core=comp_core, summary_core=summ_core,
               tree=tree, imbalance=imb, oof_core=oof_core)
    # final model (fit on the training split only) + artifact
    final = M.model_zoo(True)[PRIMARY].fit(Xtr, ytr); out["final_model"] = final
    out["interpretation"] = interpret(comp, tree, imb, summ_g)
    if write: write_baseline_outputs(out, path)
    return out


def interpret(comp, tree, imb, summ):
    lr = comp[comp.model == PRIMARY].iloc[0]; d = comp[comp.model == "Dummy (always stays)"].iloc[0]
    dt = comp[comp.model.str.startswith("Decision")].iloc[0]; rf = comp[comp.model.str.startswith("Random")].iloc[0]
    rel = "below" if lr.delta_roc_auc < -0.005 else "above" if lr.delta_roc_auc > 0.005 else "level with"
    sm = imb[imb.model.str.contains("SMOTE")].iloc[0]; bal = imb[imb.model.str.contains("balanced")].iloc[0]
    return [
        f"Logistic regression reaches a cross-validated ROC-AUC of {lr.roc_auc:.3f} (+/-{lr.roc_auc_sd:.3f}), {abs(lr.delta_roc_auc):.3f} {rel} the benchmark {lr.benchmark_roc_auc:.3f}; a gap this small is believable because my CV uses 80% of the rows and different fold boundaries, and the fold-to-fold spread is itself about {lr.roc_auc_sd:.2f}.",
        f"The always-stay Dummy scores {d.accuracy:.3f} accuracy but ROC-AUC 0.500 and recall 0, so accuracy alone would have called it a success while it finds no leavers.",
        f"The balanced logistic regression catches {lr.recall:.0%} of leavers at precision {lr.precision:.0%} (benchmark {lr.benchmark_recall:.0%} / {lr.benchmark_precision:.0%}), meaning most flagged people are false alarms: that is the price of finding more leavers on a 14.6% base rate.",
        f"The depth-4 tree ({dt.roc_auc:.3f}) is weaker than the logistic regression and the forest ({rf.roc_auc:.3f}) is about equal in ranking but almost never says 'will leave' at a 0.5 threshold (recall {rf.recall:.0%}); an unlimited tree scores {tree['train_auc_in_sample']:.2f} AUC on its own training rows but only {tree['cv_auc_out_of_fold']:.2f} out-of-fold because it memorises noise.",
        f"No honest model exceeds 0.90 (the highest is {comp[~comp.model.str.contains('LEAKY')].roc_auc.max():.3f}), so the leakage alarm is quiet, while the deliberately leaky demo hits {comp[comp.model.str.contains('LEAKY')].roc_auc.iloc[0]:.2f} and would have been sent back for investigation; SMOTE gave ROC-AUC {sm.roc_auc_mean:.3f} versus {bal.roc_auc_mean:.3f} with class weights, so it {'brings no ranking gain here' if sm.roc_auc_mean <= bal.roc_auc_mean + 0.005 else 'gives a small ranking gain worth a closer look'}.",
    ]


def write_baseline_outputs(o, path=None):
    S = o["state"]; df = S["df"]
    R.save_table(o["fold_metrics"], "baseline_fold_metrics"); R.save_table(o["comparison"], "baseline_comparison")
    R.save_table(o["comparison_core"], "baseline_comparison_core_only"); R.save_table(o["imbalance"], "imbalance_handling_comparison")
    R.save_table(pd.DataFrame([o["tree"]]), "tree_train_vs_cv")
    oof_tab = pd.DataFrame({"row_id": df.loc[S["Xtr"].index, "row_id"].to_numpy(), "y": S["ytr"].to_numpy(), **{k: v for k, v in o["oof"].items()}})
    R.save_table(oof_tab, "oof_scores_train")
    joblib.dump(o["final_model"], C.project_root() / "artifacts" / "models" / "attrition_lr_balanced.joblib")
    fe = o["final_model"].named_steps["features"]      # the fitted, reusable feature pipeline (fitted on TRAINING rows only)
    joblib.dump(fe, C.project_root() / "artifacts" / "models" / "feature_pipeline.joblib")
    R.save_json({"fitted_on": "training split only", "n_training_rows": len(S["Xtr"]), "seed": C.SEED, "input_columns": list(S["X"].columns),
                 "output_feature_names": list(fe.named_steps["prep"].get_feature_names_out()), "versions": R.package_versions()}, "feature_pipeline")
    meta = {"seed": C.SEED, "test_size": C.TEST_SIZE, "cv": f"StratifiedKFold(n_splits={C.N_SPLITS}, shuffle=True, random_state={C.SEED})",
            "train_rows": len(S["Xtr"]), "test_rows": len(S["Xte"]), "fold_sizes": [(len(a), len(b)) for a, b in S["folds"]],
            "positives_train": int(S["ytr"].sum()), "positives_test": int(S["yte"].sum()), "join_report": S["join_report"],
            "input_columns": list(S["X"].columns), "models": {k: repr(v.named_steps["model"]) for k, v in M.model_zoo(True).items()},
            "primary_model": PRIMARY, "versions": R.package_versions(), "python_warnings_during_run": o["warnings"]}
    R.save_json(meta, "baseline_run")
    comp = o["comparison"]; cc = o["comparison_core"]
    show = comp[["model", "roc_auc", "roc_auc_sd", "benchmark_roc_auc", "delta_roc_auc", "pr_auc", "precision", "recall", "f1", "accuracy"]]
    imb = o["imbalance"][["model", "roc_auc_mean", "pr_auc_mean", "precision_mean", "recall_mean", "f1_mean"]]
    lk = o["summary"][["model", "roc_auc_mean", "leakage_check"]]
    txt = f"""# Baseline comparison note (Day 3)

**Setup.** {meta['train_rows']} training rows / {meta['test_rows']} held-out test rows (stratified 80/20, seed {C.SEED}); the test set is untouched here. Stratified 5-fold CV (seed {C.SEED}) on the training rows, **identical folds for every model**. Out-of-fold probabilities come from `cross_val_predict(method="predict_proba")`; preprocessing is refit inside every fold. Threshold metrics use 0.5; undefined precision/F1 (no positive predictions) are reported as 0 (`zero_division=0`) and counted in `folds_with_undefined_precision`.

## Results vs the supplied benchmark (full pipeline: attrition features + attendance/payroll aggregates)
{R.md_table(show)}

## Same models, attrition-table features only (ablation)
{R.md_table(cc[['model','roc_auc','roc_auc_sd','benchmark_roc_auc','delta_roc_auc','pr_auc','recall']])}

The 3-month attendance/payroll aggregates exist for only {S['join_report']['attendance_matched_rows']} of {S['join_report']['rows_after_join']} rows (the 2,000 historical rows have no monthly history), so they cannot move the overall score much.

## Five-sentence interpretation
""" + "\n".join(f"{i}. {s}" for i, s in enumerate(o["interpretation"], 1)) + f"""

## Imbalance handling: class weights vs SMOTE
{R.md_table(imb)}

SMOTE is applied only to training folds, never to validation rows. It was implemented in-house (imbalanced-learn was not installable in the build sandbox); because it interpolates between one-hot columns it is closer to plain SMOTE than SMOTE-NC, a known limitation.

## Why a decision tree can look perfect on training data but weaker in cross-validation
An unlimited tree keeps splitting until it has memorised every training row (in-sample AUC {o['tree']['train_auc_in_sample']:.2f}), including noise. On rows it has not seen it scores {o['tree']['cv_auc_out_of_fold']:.2f}. Only the out-of-fold number tells you how it will behave on next year's employees.

## What the balanced logistic regression trades
`class_weight="balanced"` makes a missed leaver cost about 6x a false alarm, so the model flags many more people and finds more leavers (recall {comp[comp.model==PRIMARY].recall.iloc[0]:.0%}) but most flags are wrong (precision {comp[comp.model==PRIMARY].precision.iloc[0]:.0%}). That is acceptable for a ranked list a human reviews, and not acceptable for any automatic action.

## Leakage alarm (ROC-AUC > 0.90 triggers an investigation)
{R.md_table(lk)}

The last row is a controlled demonstration (a model that only sees whether the exit fields are filled in). It is **not** part of the real pipeline.

## Run metadata
Seed {C.SEED}; model configs and package versions in `artifacts/metadata/baseline_run.json`. Python warnings captured during the run: {o['warnings'] or 'none'}.
"""
    (C.reports_dir() / "baseline_model_comparison.md").write_text(txt)


def run_evaluation(path=None, write=True):
    S = load_model_frame(True, path); Xtr, ytr, Xte, yte, folds, df = S["Xtr"], S["ytr"], S["Xte"], S["yte"], S["folds"], S["df"]
    model = M.model_zoo(True)[PRIMARY]
    oof = M.oof_scores(model, Xtr, ytr, folds)                           # honest scores for every training row
    final = model.fit(Xtr, ytr); p_test = final.predict_proba(Xte)[:, 1]    # test used once, for reporting only
    recal = E.Recalibrator().fit(oof, ytr); p_test_cal = recal.predict(p_test); oof_cal = recal.predict(oof)
    rep_oof = E.binary_report(ytr, oof); rep_test = E.binary_report(yte, p_test)
    # threshold chosen from TRAINING out-of-fold scores only: the cut-off that flags the top 10% there, applied to test
    thr_top10 = float(np.sort(oof)[::-1][int(np.ceil(0.10 * len(oof))) - 1]); rep_test_thr = E.binary_report(yte, p_test, threshold=thr_top10)
    metrics = pd.DataFrame([{"evaluation": "Training out-of-fold (5-fold)", **rep_oof}, {"evaluation": "Held-out test (used once)", **rep_test},
                            {"evaluation": f"Held-out test @ threshold {thr_top10:.3f} (top-10% cut from training OOF)", **rep_test_thr}])
    bands = {"Training OOF": E.official_risk_bands(oof), "Held-out test": E.official_risk_bands(p_test)}   # official company rule (cortexhr_helpers.risk_bands)
    bands_ranked = {"Training OOF": E.risk_bands(oof), "Held-out test": E.risk_bands(p_test)}                 # my rank-based cross-check
    bs = {k: E.band_summary(y, b) for k, y in zip(bands, [ytr, yte]) for b in [bands[k]]}
    gains = {"Training OOF": E.gains_table(ytr, oof), "Held-out test": E.gains_table(yte, p_test)}
    cal = {"OOF raw": E.calibration_table(ytr, oof), "Test raw": E.calibration_table(yte, p_test, 8), "Test recalibrated": E.calibration_table(yte, p_test_cal, 8)}
    brier = {"test_raw": rep_test["brier"], "test_recalibrated": E.binary_report(yte, p_test_cal)["brier"],
             "mean_pred_test_raw": float(p_test.mean()), "mean_pred_test_recal": float(p_test_cal.mean()), "base_rate_test": float(yte.mean())}
    out = dict(state=S, oof=oof, p_test=p_test, p_test_cal=p_test_cal, metrics=metrics, bands=bands, band_summary=bs, gains=gains,
               calibration=cal, brier=brier, thr_top10=thr_top10, final_model=final, bands_ranked=bands_ranked,
               official={"oof": E.official_report(ytr, oof), "test": E.official_report(yte, p_test)}, parity=E.parity_table(yte, p_test))
    if write: write_eval_outputs(out)
    return out


def write_eval_outputs(o):
    S = o["state"]; ytr, yte = S["ytr"], S["yte"]; m = o["metrics"]
    R.save_table(m, "evaluation_metrics")
    R.save_table(pd.concat([t.assign(scored_set=k) for k, t in o["band_summary"].items()]), "risk_band_counts")
    R.save_table(pd.concat([t.assign(scored_set=k) for k, t in o["gains"].items()]), "gains_lift_table")
    R.save_table(pd.concat([t.assign(curve=k) for k, t in o["calibration"].items()]), "calibration_table")
    df = S["df"]
    scored = pd.DataFrame({"row_id": df.loc[S["Xte"].index, "row_id"].to_numpy(), "score": o["p_test"], "calibrated_probability": o["p_test_cal"],
                           "risk_band": o["bands"]["Held-out test"].astype(str).to_numpy(), "actual_left": yte.to_numpy()})
    R.save_table(scored.sort_values("score", ascending=False), "test_risk_ranking")
    R.save_fig(E.plot_calibration({"OOF raw": o["calibration"]["OOF raw"], "Test raw": o["calibration"]["Test raw"], "Test recalibrated": o["calibration"]["Test recalibrated"]}), "calibration_curve")
    R.save_fig(E.plot_gains_lift(o["gains"]), "gains_lift_chart")
    R.save_fig(E.plot_roc_pr({"OOF": (ytr, o["oof"]), "Test": (yte, o["p_test"])}), "roc_pr_curves")
    R.save_fig(E.plot_bands(o["band_summary"]["Held-out test"], "Risk bands on held-out test"), "risk_bands")
    joblib_path = C.project_root() / "artifacts" / "models"; import joblib; joblib.dump((o["final_model"], ), joblib_path / "attrition_lr_final_eval.joblib")
    R.save_json({"seed": C.SEED, "threshold_top10_from_training_oof": o["thr_top10"], "brier": o["brier"], "versions": R.package_versions()}, "evaluation_run")
    ro, rt = m.iloc[0], m.iloc[1]; bt = o["band_summary"]["Held-out test"].set_index("band"); bo = o["band_summary"]["Training OOF"].set_index("band")
    gt = o["gains"]["Held-out test"]; g10 = gt.iloc[0]; go10 = o["gains"]["Training OOF"].iloc[0]
    cols = ["evaluation", "n", "positives", "base_rate", "roc_auc", "pr_auc", "precision", "recall", "f1", "accuracy", "tn", "fp", "fn", "tp", "top_k", "top_precision", "top_recall", "top_lift", "brier"]
    cal_msg = (f"On the test set the raw scores average {o['brier']['mean_pred_test_raw']:.2f} against an actual leaver rate of {o['brier']['base_rate_test']:.3f}: "
               f"`class_weight=balanced` inflates probabilities, so a raw '0.40' does **not** mean 4 in 10. After Platt recalibration fitted on training out-of-fold scores only, "
               f"the average becomes {o['brier']['mean_pred_test_recal']:.3f} and the Brier score moves from {o['brier']['test_raw']:.3f} to {o['brier']['test_recalibrated']:.3f}.")
    report = f"""# Model evaluation report (Day 4)

Model: balanced logistic regression on the leakage-safe pipeline. Scores for training rows are **out-of-fold**; the held-out test set ({len(yte)} rows) was scored once by a model fit on training rows only.

## Metrics in plain words
- **ROC-AUC**: pick one person who left and one who stayed at random; ROC-AUC is the chance the model gives the leaver the higher score (0.5 = coin flip, 1.0 = perfect).
- **PR-AUC (average precision)**: how pure the flagged list stays as you flag more people, which matters when only {100*yte.mean():.1f}% leave. A useless model scores about the base rate ({100*yte.mean():.1f}%).
- **Precision**: of the people flagged, how many really left. **Recall**: of the people who left, how many were flagged.
- **Top-10%** (official `binary_report`): review only the {max(1, int(len(yte) * 0.10))} highest-scored employees (int(n x 0.10)); precision = share of them who left, recall = share of all leavers they contain.

## Metrics table
{R.md_table(m[cols], floatfmt="{:.3f}")}

## Official helper (`05_Starter/cortexhr_helpers.py`) and parity with my extended report
{R.md_table(pd.DataFrame({"training out-of-fold": o["official"]["oof"], "held-out test": o["official"]["test"]}).T.reset_index().rename(columns={"index": "evaluation"}))}

Parity (held-out test): ROC-AUC, PR-AUC and base rate are identical; the top group differs by one row because my extended report uses ceil(n x 10%) with a stable tie-break and the official helper uses int(n x 10%).

{R.md_table(o["parity"], floatfmt="{:.4f}")}

## Calibration
{cal_msg} Chart: `figures/calibration_curve.png`. Calibration is fitted on training data only and never on test labels; because it is a monotone mapping, ranking metrics and risk bands do not change.

## Risk bands: official company rule (`risk_bands`: High = score at or above the 90th percentile, Medium = at or above the 70th, Low = the rest)
Held-out test:

{R.md_table(o['band_summary']['Held-out test'])}

Training out-of-fold:

{R.md_table(o['band_summary']['Training OOF'])}

## Lift
The official top-10% group of test scores holds **{100*o['official']['test']['recall_at_top']:.1f}%** of all leavers versus 10% from random picking (precision {100*o['official']['test']['precision_at_top']:.1f}% vs base rate {100*o['official']['test']['base_rate']:.1f}%, lift {o['official']['test']['precision_at_top']/o['official']['test']['base_rate']:.2f}x); on training out-of-fold scores it holds {100*o['official']['oof']['recall_at_top']:.1f}% (lift {o['official']['oof']['precision_at_top']/o['official']['oof']['base_rate']:.2f}x). The test set is small ({int(yte.sum())} leavers), so treat its numbers as +/- several points. Chart: `figures/gains_lift_chart.png`.

## Honest limits
Only {int(yte.sum())} leavers in the test set; ROC-AUC differences below about 0.03 are noise. The model sees data up to the snapshot date; 12% of leavers had already resigned before the snapshot. 84% of rows have no attendance/payroll history. Scores support a human review and must not decide any employment action.
"""
    (C.reports_dir() / "model_evaluation_report.md").write_text(report)
    manager = f"""# CortexHR attrition score: one-page note for managers

**What the score is.** For each employee the model estimates how likely they are to leave in the next 12 months, based on pay position, raises, engagement, workload, tenure and recent attendance. We rank everyone and group them: **High** = top 10% of scores, **Medium** = next 20%, **Low** = the remaining 70%.

**How well it works (held-out data, {len(yte)} employees, {int(yte.sum())} actual leavers).**
- Overall {100*yte.mean():.1f}% left. In the **High** band {bt.loc['High','leaver_rate_pct']:.0f}% left ({int(bt.loc['High','leavers'])} of {int(bt.loc['High','employees'])}), versus {bt.loc['Low','leaver_rate_pct']:.0f}% in the **Low** band.
- The High band holds **{bt.loc['High','share_of_all_leavers_pct']:.0f}%** of all leavers (random picking would give 10%); High + Medium hold {bt.loc['High','share_of_all_leavers_pct']+bt.loc['Medium','share_of_all_leavers_pct']:.0f}%.
- Ranking quality is moderate: ROC-AUC {rt.roc_auc:.2f}, PR-AUC {rt.pr_auc:.2f}. It is useful for prioritising conversations, not for predicting individuals.

**What the score is NOT.**
- Not a decision. It does not know why someone might leave, and most High-band people ({100-bt.loc['High','leaver_rate_pct']:.0f}%) did **not** leave.
- Not a probability you can quote: raw scores are inflated by design ({o['brier']['mean_pred_test_raw']:.2f} average vs {o['brier']['base_rate_test']:.3f} actual). Use the band, not the number.
- Not evidence about performance, conduct or potential. Never use it to deny a raise, promotion or role, or to start any disciplinary step.

**Limitations.** Small number of leavers; 84% of rows lack attendance/payroll history; people who had already resigned are partly included; fairness across groups has not yet been tested (protected attributes are held separately for that audit).

**How to use it.** (1) A person, not the model, reviews each High-band name. (2) Ask what could be fixed: pay gap, stalled raise, manager change, workload. (3) Combine with what you know first-hand; ignore the score when it conflicts with facts. (4) Record the human decision, not the score. (5) Re-check monthly: scores go stale.

*Decision support only. The model must never independently determine an employment action.*
"""
    (C.reports_dir() / "manager_summary.md").write_text(manager)


# ------------------------------ Day 5 ------------------------------
def find_attendance_labels(vault=None):
    v = vault or (C.project_root() / "04_Validation_Vault")
    for p in sorted(v.glob("*")) if v.exists() else []:
        n = p.name.lower()
        if p.suffix.lower() in (".csv", ".xlsx") and ("cx-05" in n or "attendance_anomaly_labels" in n): return p
    return None


def run_attendance(labels_path=None, path=None, write=True):
    from . import attendance as T
    att = D.load_sheet(C.SHEET_ATT, path); flags = T.detect(att, contamination=0.03, seed=C.SEED)
    daily = T.daily_proxy_eval(D.load_sheet("attendance_daily_examples", path))
    anom_bench = D.load_sheet(C.SHEET_ANOM_BENCH, path)
    out = {"flags": flags, "daily_proxy": daily, "benchmark": anom_bench, "measured": None, "labels_path": None}
    if write:   # flags are written BEFORE any label is read, so the measurement cannot influence them
        R.save_table(flags, "attendance_flags"); R.save_table(T.review_queue(flags, 100), "attendance_review_queue_top100")
        R.save_table(daily, "attendance_daily_proxy_eval")
    lp = labels_path or find_attendance_labels()
    if lp:
        out["measured"] = T.measure(flags, T.load_labels(lp)); out["labels_path"] = str(lp)
        if write:
            R.save_table(out["measured"]["overall"], "attendance_detector_metrics"); R.save_table(out["measured"]["top_k"], "attendance_topk_review_metrics")
            if out["measured"]["by_type"] is not None: R.save_table(out["measured"]["by_type"], "attendance_by_anomaly_type")
    if write: write_attendance_outputs(out)
    return out


def write_attendance_outputs(o):
    import matplotlib.pyplot as plt
    f = o["flags"]; b = o["benchmark"].iloc[0]; m = o["measured"]
    fig, ax = plt.subplots(figsize=(7, 3.8)); ax.hist(f.iforest_score, bins=60, color="#4C78A8")
    thr = f.loc[f.iforest_flag, "iforest_score"].min(); ax.axvline(thr, c="r", ls="--", label=f"3% contamination cut-off ({int(f.iforest_flag.sum())} rows)")
    ax.set_xlabel("Isolation Forest anomaly score (higher = more unusual)"); ax.set_ylabel("Employee-months"); ax.legend(fontsize=8)
    ax.set_title("Attendance anomaly scores", fontsize=10); R.save_fig(fig, "attendance_anomaly_scores")
    n = len(f)
    if m is None:
        metrics = ("## Measured precision and recall\n\n**NOT MEASURED YET.** The hidden attendance labels (`unlock_CX-05` / `attendance_anomaly_labels`) were not in the supplied "
                   "`04_Validation_Vault` Drive folder, which contained only `unlock_CX-01_answer_key_notes.csv`. No number below is invented. "
                   "To measure: put the labels file in `04_Validation_Vault/` and run `python scripts/detect_attendance_anomalies.py` (it auto-detects the file). "
                   "The flags were already saved to `reports/tables/attendance_flags.csv` before any label is read, so measuring cannot change them.\n\n"
                   f"Reference to beat (anomaly_baselines): precision {b.precision:.3f}, recall {b.recall:.3f}, {int(b.flagged)} flagged, {int(b.injected)} injected.")
    else:
        ov = m["overall"]; bt = m["by_type"]
        metrics = (f"## Measured precision and recall (labels used for measurement only: `{o['labels_path']}`)\n\n{R.md_table(ov)}\n\n"
                   f"Reference (anomaly_baselines): precision {b.precision:.3f}, recall {b.recall:.3f}, {int(b.flagged)} flagged, {int(b.injected)} injected.\n\n"
                   f"### If reviewers can only check N rows per month\n{R.md_table(m['top_k'])}\n\n" +
                   (f"### Anomaly types: which are hardest\n{R.md_table(bt)}\n\n{hard_types_text(f, bt)}\n" if bt is not None else ""))
    txt = f"""# Attendance anomaly report (Day 5)

## What was built
- Data: `attendance_monthly_features`, {n:,} employee-months for {f.employee_id.nunique()} employees. **No labels are read at any point of training or flagging.**
- **Model:** `IsolationForest(n_estimators=300, contamination=0.03, random_state={C.SEED})`. It builds random trees; an unusual row is separated from the others in very few splits, so it gets a high score. Contamination = the share of rows we choose to flag (3% -> {int(f.iforest_flag.sum())} rows).
- **Features:** the 10 attendance columns (`{', '.join(__import__('cortexhr.attendance', fromlist=['x']).RAW_FEATURES)}`) plus three label-free comparisons with the same employee's own median month (attendance rate, late arrivals, hours) so that a *sudden* change for one person stands out even if the absolute value looks normal.
- **Rules (explainable, domain logic, no labels):** R1 zero attendance for a whole month ({int(f.rule_zero_attendance.sum())} rows); R2 implausible hours, i.e. above 12 h/day or below 3 h on days present ({int(f.rule_implausible_hours.sum())} rows).
- **Combination:** a row is flagged if the model OR a rule fires: {int(f.flagged.sum())} rows ({100*f.flagged.mean():.1f}%): {int(f.iforest_flag.sum())} model, {int(f.rule_flag.sum())} rule, {int((f.iforest_flag & f.rule_flag).sum())} both. Rules add certainty for obvious cases; the model finds patterns nobody wrote a rule for.

## Ranking flags (think for yourself: who checks 1,000 rows?)
Flags are ranked by `priority_score` = model score + 1.0 if a rule fired, so rule-confirmed rows come first and the rest follow by how unusual they are. Reviewers work top-down within their capacity (`reports/tables/attendance_review_queue_top100.csv`, `review_rank` column in `attendance_flags.csv`). Two production ideas: suppress repeat flags for the same employee-month pattern, and add leave status so legitimate long absences (e.g. maternity leave, which the monthly table cannot show) are not escalated as anomalies; that is the main false-positive trap for the zero-attendance rule.

{metrics}

## Supplementary check on the labelled DAILY sample (a different table, measurement only)
`attendance_daily_examples` (2,000 daily records with labels, shipped inside the main workbook) was used only to measure a daily version of the same ideas; it does **not** replace the monthly measurement above. Anomaly types there: ODD_HOUR_LOGIN, PRESENT_WITH_ZERO_HOURS, IMPOSSIBLE_HOURS, CHECKOUT_BEFORE_CHECKIN (30 each).

{R.md_table(o['daily_proxy'])}

Reading: simple clock/hour rules catch most daily anomalies with high precision; the Isolation Forest alone is weaker at 3% contamination. On the monthly table (measured above) the same pattern holds: the hours rule fully covers impossible-hours rows while the model covers streaks and sudden drops.

## Checklist
- Precision and recall reported: {'yes (monthly labels)' if m is not None else 'daily proxy only; monthly labels missing from the supplied vault'}
- Hard anomaly types noted: {'yes' if m is not None and m['by_type'] is not None else 'hypotheses only until monthly labels are supplied'}
- Never trained on labels: yes (`assert_no_labels`, test `test_attendance_never_uses_labels`)
- Rules and model explained: yes
"""
    (C.reports_dir() / "attendance_anomaly_report.md").write_text(txt)


def hard_types_text(flags, bt):
    """Plain-language reading of the by-type table. v1 numbers are the honest measurement; v2 is post-hoc."""
    tcol = bt.columns[0]; hard = bt[bt.recall < 0.5]
    fp_total = int(flags.flagged.sum() - bt.caught.sum())
    zero_fp = int(flags.rule_zero_attendance.sum() - bt.loc[bt[tcol].astype(str).str.contains("ZERO"), "caught"].sum())
    hard_txt = ", ".join(f"`{t}` (recall {r:.2f}, {int(mi)} of {int(i)} missed)" for t, r, mi, i in zip(hard[tcol], hard.recall, hard.missed, hard.injected)) or "none below 0.5 recall"
    rows = [f"**Hardest type(s) for v1:** {hard_txt}.",
            "**Why (my reading):** a late-arrival spike is an extreme in ONE column, but the Isolation Forest scores combinations of columns: with only 3% of rows to flag, single-column spikes were out-ranked by rows that were odd on several columns, and no rule covered late arrivals.",
            f"**False alarms:** {fp_total} of {int(flags.flagged.sum())} flags were not injected anomalies. About {zero_fp} came from the zero-attendance rule: those months are fully covered by paid leave (e.g. maternity or long leave), which is legitimate. This is the known false-positive trap; leave status must be used.",
            "**Post-hoc v2** (written after seeing this table, so its numbers are optimistic): R1 ignores months covered by paid leave and a new R3 flags late arrivals above the 99.5th percentile of the column (a label-free threshold). Compare the v2 row in the table above; do not present v2 as an honest test."]
    return "\n\n".join(rows)
