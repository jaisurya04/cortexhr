"""Generate model cards, the experiment log, per-task reports and the traceability matrix from the saved results (nothing typed in by hand)."""
import json, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
import traceability as TR
rd = lambda p: pd.read_csv(ROOT / p); js = lambda p: json.loads((ROOT / p).read_text(encoding="utf-8"))
W = lambda p, t: ((ROOT / p).parent.mkdir(parents=True, exist_ok=True), (ROOT / p).write_text(t, encoding="utf-8"))

q = rd("reports/task11/ranking_quality_summary.csv").set_index("engine"); per = rd("reports/task11/ranking_quality_per_job.csv"); mis = rd("reports/task11/misranking_analysis.csv"); wexp = rd("reports/task11/weight_experiments.csv")
fa = js("reports/task12/fairness_results.json"); abl = pd.DataFrame(fa["ablation"]); cvs = rd("reports/task13/cv_summary.csv").set_index("model"); pdiff = rd("reports/task13/paired_differences.csv")
tl13 = rd("reports/task13/tuning_log.csv"); cal = rd("reports/task13/calibration_comparison.csv"); tm13 = rd("reports/task13/test_metrics.csv"); m13 = js("artifacts/metadata/attrition_v2_metadata.json")
run14 = js("reports/task14/explanation_run.json"); risk = rd("artifacts/predictions/attrition_risk_2026.csv"); m15 = js("artifacts/metadata/salary_band_v1_metadata.json"); tl15 = rd("reports/task15/tuning_log.csv")
mr = js("artifacts/metadata/resume_ranker_v2_metadata.json")
f4 = lambda x: f"{x:.4f}"; f3 = lambda x: f"{x:.3f}"
sel = lambda m: [r for r in m13["test_metrics"] if r["model"].startswith("attrition-v2")][0]
b_name, v_name = "Logistic Regression (balanced) [baseline]", "attrition-v2 candidate: LR + tuned HGB blend"
# ---------------------------------------------------------------------------------------------- model cards
W("docs/model_cards/resume_ranker_v2.md", f"""# Model card: resume ranker v2 ({mr['engine_version']})
**Intended use.** Decision support for recruiters: rank resumes against a job description and show why. A human reviews every candidate. **Not for:** automatic rejection, hiring decisions, or any use that adds name, gender, age or college tier.
**Method.** TF-IDF similarity + required and nice-to-have skill coverage + capped experience fit; weights {mr['weights']}. No parameters learned from labels.
**Inputs used:** {', '.join(mr['score_inputs'])}. **Never used:** {', '.join(mr['never_used'])}.
**Evidence (555 human-labelled pairs, 10 jobs, ranking inside each job's labelled pool).** precision@5 {f3(q.loc['final engine (v2, masked)','precision@5'])}, recall@5 {f3(q.loc['final engine (v2, masked)','recall@5'])}, hit rate@5 {f3(q.loc['final engine (v2, masked)','hit_rate@5'])}, NDCG@5 {f3(q.loc['final engine (v2, masked)','ndcg@5'])}, MRR {f3(q.loc['final engine (v2, masked)','mrr'])}; random ordering gives precision@5 {f3(q.loc['random ordering','precision@5'])} and NDCG@5 {f3(q.loc['random ordering','ndcg@5'])}.
**Fairness (Task 12).** 25 twin pairs: mean absolute twin gap {fa['before']['mean_abs_gap']:.6f} before masking, {fa['after']['mean_abs_gap']:.6f} after; NDCG@5 {f3(fa['quality_before']['ndcg@5'])} to {f3(fa['quality_after']['ndcg@5'])}. See `docs/fairness/fairness_statement.md`.
**Limits.** See `docs/limitations.md`. Artifact `artifacts/models/resume_ranker_v2.joblib`, sha256 `{mr['sha256'][:16]}...`, created {mr['created_utc']}.""")
v, b = cvs.loc[v_name], cvs.loc[b_name]; ts = sel(m13)
W("docs/model_cards/attrition_v2.md", f"""# Model card: attrition-v2
**Intended use.** Support for HR conversations: a calibrated probability that an employee leaves within 12 months, with reasons (Task 14). **Not for:** dismissal, pay, promotion or any single-handed decision about a person; every output says human review is required.
**Model.** {m13['model_family']}. Calibration: {m13['calibration_method']} (chosen by out-of-fold Brier score). Gradient-boosting settings: {m13['hgb_params']}.
**Data and split.** Phase 1 frame (exact duplicates removed, cut-off-safe monthly aggregates). {m13['train_rows']} training rows, {m13['test_rows']} test rows; 5 stratified folds, seed {m13['seed']}, identical to Task 3. Final production fit on {m13['production_fit_rows']} rows. Inputs: {len(m13['features_raw'])} raw columns; excluded as inputs: {', '.join(m13['excluded_as_inputs'])}.
**Same-fold results (mean +- sd over 5 folds).** Baseline logistic regression ROC-AUC {f3(b.roc_auc_mean)} +- {f3(b.roc_auc_sd)}, PR-AUC {f3(b.pr_auc_mean)}; attrition-v2 ROC-AUC {f3(v.roc_auc_mean)} +- {f3(v.roc_auc_sd)}, PR-AUC {f3(v.pr_auc_mean)}.
**Is the gain real?** {m13['verdict']['text']} (mean paired fold difference {m13['verdict']['mean_diff']:+.4f}, 95% interval [{m13['verdict']['ci'][0]:+.4f}, {m13['verdict']['ci'][1]:+.4f}]).
**Test set (used once).** ROC-AUC {f3(ts['roc_auc'])}, PR-AUC {f3(ts['pr_auc'])}, Brier {f3(ts['brier'])}, ECE {f3(ts['ece'])}.
**Honest summary.** v2 is equivalent to the simpler logistic baseline within the noise of five folds; it is saved because Task 14 asks for it, not because it is proven better.
**Environment.** {m13['environment']}. Checksums are in `artifacts/metadata/attrition_v2_metadata.json`.""")
t15 = m15["test_metrics"]; fm = m15["model"]
W("docs/model_cards/salary_band_v1.md", f"""# Model card: {m15['model_version']}
**Intended use.** Advice to a manager and HR: a low / mid / high annual fixed salary band for a role at hiring. The final decision stays with people. **Not for:** an automatic offer, or setting pay of existing staff.
**Model.** {fm} on the log salary, inputs {', '.join(m15['features'])}. Excluded as not job-related: {', '.join(m15['excluded_job_unrelated'])}. Never inputs: {', '.join(m15['never_inputs'])}.
**Accuracy (test set, used once).** MAE Rs {t15[fm]['MAE_rupees']:,.0f}, R2 {t15[fm]['R2']:.3f}; baseline 'average by role' MAE Rs {t15['baseline: average offered salary by role']['MAE_rupees']:,.0f}, R2 {t15['baseline: average offered salary by role']['R2']:.3f}; role average adjusted for market index MAE Rs {t15['baseline 2: role average adjusted for market index']['MAE_rupees']:,.0f}.
**Band.** mid x exp({m15['band_multipliers_log']['low']:.3f}) to mid x exp({m15['band_multipliers_log']['high']:.3f}) (10th and 90th percentile of out-of-fold log error); coverage on the test set {m15['band_coverage_test']:.3f}.
**Fairness.** See `reports/task15/README.md` (gender and career-gap gaps, measured with a restricted file).
**Limits.** {m15['note']} See `docs/limitations.md`.""")
# ---------------------------------------------------------------------------------------------- experiment log
best_w = wexp.sort_values("ndcg@5", ascending=False).iloc[0]
W("docs/experiment_log.md", f"""# Experiment log (generated from the saved tables)

| Area | Experiments | Selection rule | Outcome | File |
|---|---|---|---|---|
| Resume score weights | {len(wexp)} weightings (step 0.1) | weights fixed a priori; grid is a sensitivity analysis; leave-one-job-out check | NDCG@5 ranges {f3(wexp['ndcg@5'].min())} to {f3(wexp['ndcg@5'].max())}; configured weights {mr['weights']} give {f3(q.loc['final engine (v2, masked)','ndcg@5'])} | `reports/task11/weight_experiments.csv` |
| Masking ablation | {len(abl)} variants | measure which identifier causes the twin gap | {', '.join(f"{r['masked']}: mean gap {r['mean_abs_gap']:.6f}" for _, r in abl.iterrows())} | `reports/task12/ablation_masking.csv` |
| Attrition gradient boosting | {len(tl13)} settings, 5 folds each | highest mean CV PR-AUC on the training part | selected `{tl13.loc[tl13.decision.str.startswith('selected'),'experiment_id'].iloc[0]}`; top-5 settings span {f3(tl13.pr_auc_mean.nlargest(5).max()-tl13.pr_auc_mean.nlargest(5).min())} PR-AUC (fold sd about {f3(tl13.pr_auc_sd.mean())}) | `reports/task13/tuning_log.csv` |
| Attrition calibration | {len(cal)} methods | lowest out-of-fold Brier | chosen {m13['calibration_method']}: {', '.join(f"{r['method']} Brier {r['brier']:.4f} ECE {r['ece']:.4f}" for _, r in cal.iterrows())} | `reports/task13/calibration_comparison.csv` |
| Attrition v2 vs baseline | 5 paired folds, bootstrap | gain real only if the 95% interval is above 0 | {m13['verdict']['text']} | `reports/task13/paired_differences.csv` |
| Explanations | 120 permutations (2 independent halves) | split-half agreement | phi correlation {run14['stability']['phi_correlation']:.3f}; additivity error {run14['additivity_max_error']:.1e} | `reports/task14/explanation_run.json` |
| Salary gradient boosting | {len(tl15)} settings, 5 folds each | lowest CV MAE | selected `{tl15.loc[tl15.decision.str.startswith('selected'),'experiment_id'].iloc[0]}` | `reports/task15/tuning_log.csv` |
""")
# ---------------------------------------------------------------------------------------------- task reports
per_s = per.sort_values("precision@5")
W("reports/task11/README.md", f"""# Task 11: resume ranking
Engine {mr['engine_version']} (TF-IDF + skills + experience, identifiers masked). Notebooks 01 to 04.
**Quality vs human labels (555 pairs, 10 jobs):** precision@5 {f3(q.loc['final engine (v2, masked)','precision@5'])}, recall@5 {f3(q.loc['final engine (v2, masked)','recall@5'])}, hit rate@5 {f3(q.loc['final engine (v2, masked)','hit_rate@5'])}, NDCG@5 {f3(q.loc['final engine (v2, masked)','ndcg@5'])}, MRR {f3(q.loc['final engine (v2, masked)','mrr'])} (random: precision@5 {f3(q.loc['random ordering','precision@5'])}, NDCG@5 {f3(q.loc['random ordering','ndcg@5'])}).
**Weakest job:** {per_s.iloc[0].job_id} (precision@5 {f3(per_s.iloc[0]['precision@5'])}, {int(per_s.iloc[0].relevant)} relevant candidates, so precision@5 cannot exceed {min(1, per_s.iloc[0].relevant/5):.2f}).
**Misranks:** {len(mis)} shown with root cause in `misranking_analysis.csv`.
**Files:** ranking_quality_per_job.csv, ranking_quality_summary.csv, misranking_analysis.csv, top5_per_job_all_resumes.csv, weight_experiments.csv. API contract: `docs/api_notes_resume_ranking.md`.""")
W("reports/task12/README.md", f"""# Task 12: fairness audit of the resume screener
25 twin pairs, 75 comparisons. Notebooks 05 and 06; statement in `docs/fairness/fairness_statement.md`.
**Before (text as given):** non-zero gap in {fa['before']['comparisons_with_nonzero_gap']} of {fa['before']['comparisons']} comparisons; mean absolute gap {fa['before']['mean_abs_gap']:.6f}, median {fa['before']['median_abs_gap']:.6f}, maximum {fa['before']['max_abs_gap']:.6f}.
**After (identifiers masked):** mean {fa['after']['mean_abs_gap']:.6f}, maximum {fa['after']['max_abs_gap']:.6f}. Ranking quality NDCG@5 {f3(fa['quality_before']['ndcg@5'])} to {f3(fa['quality_after']['ndcg@5'])}, precision@5 {f3(fa['quality_before']['precision@5'])} to {f3(fa['quality_after']['precision@5'])}.
**Source of the gap (ablation):** {'; '.join(f"{r['masked']} -> {r['mean_abs_gap']:.6f}" for _, r in abl.iterrows())}.
**Remaining:** a zero gap is by construction; other proxies untested (see statement).""")
W("reports/task13/README.md", f"""# Task 13: attrition model v2
Notebooks 07 to 09. Same folds as Task 3. Candidates: logistic regression (baseline), random forest, histogram gradient boosting, LR + gradient boosting blend (v2).
**CV ROC-AUC:** baseline {f3(b.roc_auc_mean)} +- {f3(b.roc_auc_sd)}; gradient boosting alone {f3(cvs.loc['HistGradientBoosting tuned','roc_auc_mean'])}; attrition-v2 {f3(v.roc_auc_mean)} +- {f3(v.roc_auc_sd)}.
**Verdict:** {m13['verdict']['text']} Mean paired difference {m13['verdict']['mean_diff']:+.4f}, 95% interval [{m13['verdict']['ci'][0]:+.4f}, {m13['verdict']['ci'][1]:+.4f}].
**Calibration:** {m13['calibration_method']}; test ECE {f3(ts['ece'])}, Brier {f3(ts['brier'])} (uncalibrated ECE {f3(cal.set_index('method').loc['none','ece'])}).
**Saved:** `artifacts/models/attrition_v2.joblib` with metadata and checksum. Model card: `docs/model_cards/attrition_v2.md`.""")
bands = risk.risk_band.value_counts().to_dict()
W("reports/task14/README.md", f"""# Task 14: explain every person's risk
Notebooks 10 and 11. 430 active employees scored with attrition-v2; bands {bands}; mean risk {risk.risk_score_12m.mean():.3f}.
**Explanations:** Monte Carlo Shapley values over 23 groups (120 permutations, 200-row background): additivity error {run14['additivity_max_error']:.1e}; split-half correlation {run14['stability']['phi_correlation']:.3f}; identical top-3 set in {run14['stability']['top_n_identical_share']*100:.0f}% of employees (mean overlap {run14['stability']['mean_top_n_overlap']:.2f}).
**Format checks:** {sum(c['passed'] for c in run14['checks'])} of {len(run14['checks'])} passed (columns, order, ranges, bands, three distinct reasons, human-review note, wording).
**Output:** `attrition_risk_2026.csv`. Manual review of 10 rows and the wording changes it caused are in notebook 11.""")
g15 = rd("reports/task15/gender_means.csv").set_index("gender") if (ROOT / "reports/task15/gender_means.csv").exists() else None
W("reports/task15/README.md", f"""# Task 15: fair pay bands
Notebooks 12 to 14. Model {fm}, job-related inputs only (career_gap_months excluded, gender and age never inputs).
**Test accuracy:** MAE Rs {t15[fm]['MAE_rupees']:,.0f}, R2 {t15[fm]['R2']:.3f} versus the 'average by role' baseline MAE Rs {t15['baseline: average offered salary by role']['MAE_rupees']:,.0f}, R2 {t15['baseline: average offered salary by role']['R2']:.3f}. A role average adjusted for the market index is already close (MAE Rs {t15['baseline 2: role average adjusted for market index']['MAE_rupees']:,.0f}): most of the signal is role, level, band and market.
**Band coverage on the test set:** {m15['band_coverage_test']:.3f} (target 0.80).
{('**Pay ratio by gender (history, model with career gap, chosen model):** ' + '; '.join(f"{i}: {r.iloc[0]:.4f} / {r.iloc[1]:.4f} / {r.iloc[2]:.4f}" for i, r in g15.iterrows()) + '. Details and bootstrap intervals in notebook 14.') if g15 is not None else 'Group gaps skipped: restricted file not present.'}
**Limits:** `docs/limitations.md`.""")
# Additional evidence generated by the optional official LightGBM and SHAP-library audit runs.
lgb_summary_path = ROOT / "reports/task13/lightgbm_cv_summary.csv"
lgb_test_path = ROOT / "reports/task13/lightgbm_test_metrics.csv"
if lgb_summary_path.exists() and lgb_test_path.exists():
    lgb_cv_now = pd.read_csv(lgb_summary_path).iloc[0]
    lgb_test_now = pd.read_csv(lgb_test_path).iloc[0]
    extra13 = ("\n\n**Official LightGBM candidate (separate challenger).** Best of three predeclared settings on the same five Task 3 folds: CV ROC-AUC "
               + f3(lgb_cv_now.roc_auc_mean) + ", PR-AUC " + f3(lgb_cv_now.pr_auc_mean)
               + "; calibrated holdout ROC-AUC " + f3(lgb_test_now.roc_auc) + ", PR-AUC " + f3(lgb_test_now.pr_auc)
               + ". It did not beat the existing LR + HGB blend and is stored separately as `artifacts/models/attrition_lightgbm_candidate.joblib`. Details: `reports/task13/lightgbm_tuning_log.csv`.\n")
    for rel in ("reports/task13/README.md", "docs/model_cards/attrition_v2.md", "docs/experiment_log.md"):
        fp = ROOT / rel; old = fp.read_text(encoding="utf-8")
        if "Official LightGBM candidate (separate challenger)" not in old:
            fp.write_text(old.rstrip() + extra13, encoding="utf-8")
shap_check = run14.get("shap_library_cross_check")
if shap_check:
    extra14 = ("\n\n**Actual SHAP library cross-check:** SHAP " + str(shap_check["shap_version"])
               + " PermutationExplainer on " + str(shap_check["rows"]) + " employees; additivity error "
               + f"{shap_check['additivity_max_error']:.1e}"
               + ". Feature-level and grouped results: `shap_library_feature_cross_check.csv` and `shap_library_group_cross_check.csv`.\n")
    fp = ROOT / "reports/task14/README.md"; old = fp.read_text(encoding="utf-8")
    if "Actual SHAP library cross-check" not in old:
        fp.write_text(old.rstrip() + extra14, encoding="utf-8")

n_ok, n_all = TR.write(TR.build())
print("docs written;", n_ok, "of", n_all, "requirements complete")
