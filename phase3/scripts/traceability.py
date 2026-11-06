"""Requirement traceability: every Phase 3 instruction -> notebook, code, output, test. A row is 'Complete' only if every referenced file and test exists."""
from __future__ import annotations
import ast, csv, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def R(i, task, req, nbs, code, out, tests, note=""):
    return dict(id=i, task=task, requirement=req, notebooks=nbs, code=code, outputs=out, tests=tests, note=note)


S = "src/cortexhr/phase3/"
T = "reports/task%d/"
TRACE = [
 # ------------------------------------------------------------------ Task 11
 R("11.01", 11, "Read 5 resumes and 3 job descriptions; list what makes a good match", ["02"], [], [], ["test_data_validation::test_phase3_data_files_have_expected_shapes"]),
 R("11.02", 11, "Data audit: dimensions, types, missing, duplicates, text quality, label distribution, id links, label consistency", ["01"], [S + "common/validation.py"], [], ["test_data_validation::test_keys_are_unique_and_links_are_intact", "test_data_validation::test_labels_are_internally_consistent"]),
 R("11.03", 11, "Text pipeline: normalise, tokenise, stop words, lemmatise, keep skill phrases, handle malformed text", ["03"], [S + "resume/text.py"], [], ["test_resume_ranking::test_masking_removes_name_email_and_tier"], "NLTK/spaCy not installed: conservative rule-based folding is used and documented"),
 R("11.04", 11, "Extract skills (exact and synonym matches) from resumes and jobs; matched, missing, overlap", ["02"], [S + "resume/skills.py"], [], ["test_resume_ranking::test_skill_extraction_synonyms_and_span_consumption", "test_resume_ranking::test_skill_extraction_quality_against_structured_lists"]),
 R("11.05", 11, "Extract years of experience; handle missing, invalid, overlapping; not the only factor", ["02"], [S + "resume/experience.py"], [], ["test_resume_ranking::test_experience_extraction_matches_the_structured_years_for_every_resume", "test_resume_ranking::test_experience_edge_cases"]),
 R("11.06", 11, "Score with TF-IDF + cosine + skill overlap (+ experience); documented formula and weights; normalised components", ["03"], [S + "resume/scoring.py", "docs/methodology.md"], [], ["test_resume_ranking::test_ranking_is_deterministic_and_returns_top_k", "test_resume_ranking::test_weights_must_sum_to_one"]),
 R("11.07", 11, "Test different weights and record experiments", ["03"], [S + "resume/analysis.py"], [T % 11 + "weight_experiments.csv"], ["test_outputs::test_every_task_has_its_evidence_files"]),
 R("11.08", 11, "No name, gender, age or college tier in the score", ["04"], [S + "resume/scoring.py", S + "resume/api.py"], [], ["test_resume_ranking::test_ranking_inputs_exclude_protected_fields", "test_resume_ranking::test_changing_only_the_name_does_not_change_the_masked_score"]),
 R("11.09", 11, "Rank candidates per job; deterministic ties; top 5 with score breakdown, matched and missing skills, experience evidence, human-review recommendation", ["04"], [S + "resume/scoring.py"], [T % 11 + "top5_per_job_all_resumes.csv"], ["test_resume_ranking::test_ranking_is_deterministic_and_returns_top_k", "test_resume_fairness::test_ties_are_not_always_broken_in_favour_of_the_original"]),
 R("11.10", 11, "Compare with resume_job_labels: precision@5, recall@5, hit rate@5, NDCG@5, MRR; per job and aggregate; ranking quality table", ["04"], [S + "resume/metrics.py"], [T % 11 + "ranking_quality_per_job.csv", T % 11 + "ranking_quality_summary.csv"], ["test_resume_ranking::test_metric_functions_on_a_toy_ranking", "test_outputs::test_ranking_summary_file_agrees_with_a_fresh_computation", "test_resume_ranking::test_ranking_quality_meets_the_success_criteria_fixed_in_advance"]),
 R("11.11", 11, "Five genuine misranked examples with root cause and improvement (real number if fewer)", ["04"], [S + "resume/analysis.py"], [T % 11 + "misranking_analysis.csv"], ["test_outputs::test_five_misranks_or_the_real_number_are_reported"]),
 R("11.12", 11, "Ranking endpoint: inputs, outputs, validation, errors, example", ["04"], [S + "resume/api.py", "docs/api_notes_resume_ranking.md"], [], ["test_resume_ranking::test_api_contract_good_and_bad_requests"]),
 R("11.13", 11, "Explain the score to a recruiter; would I rank on years alone?", ["02", "04"], [], [], ["test_resume_ranking::test_ranking_inputs_exclude_protected_fields"]),
 R("11.14", 11, "Visualisations and Task 11 report", ["04"], [], ["reports/figures/task11_ranking_quality_per_job.png", T % 11 + "README.md"], ["test_outputs::test_every_task_has_its_evidence_files"]),
 # ------------------------------------------------------------------ Task 12
 R("12.01", 12, "Validate the 25 twin pairs: ids, changed attribute, other content identical, missing values", ["05"], [S + "fairness/twins.py"], [], ["test_resume_fairness::test_twin_pairs_are_valid"]),
 R("12.02", 12, "Use unlock_CX-12 for the protected attributes; never guess hidden attributes", ["05", "06"], [S + "fairness/twins.py"], [], ["test_outputs::test_fairness_statement_matches_the_saved_results"], "authorized restricted file supplied for this audit; 185/185 resume IDs matched; notebooks 05 and 06 re-executed; file remains excluded from ZIP"),
 R("12.03", 12, "Score both twins; gap per pair, average, median, maximum, non-zero count; sign and units defined", ["05"], [S + "fairness/twins.py"], [T % 12 + "twin_pair_results.csv"], ["test_resume_fairness::test_unmasked_screener_leaks_through_the_free_text"]),
 R("12.04", 12, "Investigate sources: names, e-mail, tier, career gap, education wording, proxies", ["05", "06"], [S + "resume/text.py"], [T % 12 + "ablation_masking.csv"], ["test_resume_fairness::test_name_and_email_are_the_source_of_the_gap"]),
 R("12.05", 12, "Reduce the gap by masking; rerun all 25 pairs; before/after average, median, maximum, number affected, % improvement", ["06"], [S + "resume/scoring.py"], [T % 12 + "before_after_gap.csv"], ["test_resume_fairness::test_masked_screener_has_no_twin_gap"]),
 R("12.06", 12, "Ranking quality still reported and not collapsed", ["06"], [S + "resume/metrics.py"], [T % 12 + "ranking_quality_before_after.csv"], ["test_resume_fairness::test_ranking_quality_does_not_collapse_after_the_fix"]),
 R("12.07", 12, "Honest fairness statement: tested, found, changed, remains; equal accuracy is not equal treatment; trade-off view", ["06"], ["docs/fairness/fairness_statement.md"], ["docs/fairness/fairness_statement.md"], ["test_outputs::test_fairness_statement_matches_the_saved_results"]),
 R("12.08", 12, "Group gap with the supplied group_gap() helper", ["06"], ["05_Starter/cortexhr_helpers.py"], [T % 12 + "group_gaps_full_pool.csv"], ["test_resume_fairness::test_group_gap_helper_signature_and_values"]),
 R("12.09", 12, "Fairness charts, per-pair results, Task 12 report", ["06"], [], ["reports/figures/task12_twin_gap_before_after.png", T % 12 + "README.md"], ["test_outputs::test_every_task_has_its_evidence_files"]),
 # ------------------------------------------------------------------ Task 13
 R("13.01", 13, "Inspect baseline pipeline, benchmarks, features, leakage audit, folds; record exact baseline", ["07"], [S + "attrition/v2.py"], [T % 13 + "cv_summary.csv"], ["test_attrition_v2::test_same_folds_as_phase_one_and_no_leakage_columns"]),
 R("13.02", 13, "Recheck target, leakage, future information, duplicates, missing, invalid, protected features, contamination", ["07"], [S + "attrition/v2.py"], [], ["test_attrition_v2::test_same_folds_as_phase_one_and_no_leakage_columns", "test_data_validation::test_scoring_inputs_have_no_forbidden_or_protected_columns"]),
 R("13.03", 13, "Reuse the Task 2 pipeline and Task 3 folds; preprocessing fitted inside folds", ["07", "08"], [S + "attrition/v2.py"], [], ["test_attrition_v2::test_same_folds_as_phase_one_and_no_leakage_columns"]),
 R("13.04", 13, "Train baseline and gradient boosting (LightGBM / XGBoost where supported, else documented substitute)", ["08"], [S + "attrition/v2.py"], [T % 13 + "fold_metrics.csv", T % 13 + "lightgbm_tuning_log.csv", T % 13 + "lightgbm_fold_metrics.csv", T % 13 + "lightgbm_cv_summary.csv", T % 13 + "lightgbm_test_metrics.csv", T % 13 + "cv_summary_with_lightgbm.csv", T % 13 + "test_metrics_with_lightgbm.csv"], ["test_attrition_v2::test_blend_classifier_averages_probabilities", "test_attrition_v2::test_lightgbm_candidate_is_real_and_uses_phase1_pipeline_when_installed", "test_attrition_v2::test_lightgbm_audit_artifacts_and_metrics_are_saved_consistently"], "When installed, a real LightGBM candidate is tuned and evaluated on the same Task 3 folds; HGB remains a documented comparator/fallback"),
 R("13.05", 13, "Tune a few parameters with cross-validation; keep a log", ["08"], [S + "attrition/v2.py"], [T % 13 + "tuning_log.csv"], ["test_attrition_v2::test_gradient_boosting_comparison_is_reported_honestly"]),
 R("13.06", 13, "Same-fold comparison table with benchmarks and own baseline; ROC-AUC, PR-AUC, precision, recall, F1, confusion matrix, balanced accuracy, Brier, CV mean and SD", ["08", "09"], [S + "attrition/v2.py"], [T % 13 + "cv_summary.csv", T % 13 + "comparison_with_benchmarks.csv", T % 13 + "test_metrics.csv"], ["test_attrition_v2::test_metric_helpers_on_toy_values"]),
 R("13.07", 13, "Calibration: none, sigmoid, isotonic; Brier and ECE; never on the test set", ["09"], [S + "attrition/train.py"], [T % 13 + "calibration_comparison.csv", "reports/figures/task13_calibration.png"], ["test_attrition_v2::test_calibration_meets_the_success_criterion"]),
 R("13.08", 13, "Honest assessment: is the gain real, practically meaningful, worth the complexity; would I ship v2", ["09"], [S + "attrition/v2.py"], [T % 13 + "paired_differences.csv"], ["test_attrition_v2::test_verdict_rule_is_conservative"]),
 R("13.09", 13, "Save model as attrition-v2 with version, config, features, metrics, timestamp, dependency versions; do not overwrite Phase 1/2", ["09"], [S + "attrition/train.py"], ["artifacts/models/attrition_v2.joblib", "artifacts/metadata/attrition_v2_metadata.json"], ["test_attrition_v2::test_saved_model_artifacts_and_metadata", "test_attrition_v2::test_model_scores_the_430_employees_in_range_and_is_reproducible"]),
 R("13.10", 13, "Model card, experiment log, visualisations, Task 13 report", ["09"], ["docs/model_cards/attrition_v2.md", "docs/experiment_log.md"], ["reports/figures/task13_permutation_importance.png", T % 13 + "README.md"], ["test_outputs::test_every_task_has_its_evidence_files"]),
 # ------------------------------------------------------------------ Task 14
 R("14.01", 14, "Audit the 430-row scoring file: schema, missingness, duplicates, unseen categories, invalid values, no leakage; never train on it", ["10"], [S + "attrition/v2.py"], [], ["test_data_validation::test_scoring_file_matches_the_sample_format_order", "test_data_validation::test_scoring_inputs_have_no_forbidden_or_protected_columns"]),
 R("14.02", 14, "Score with attrition-v2; assign bands with the supplied risk_bands()", ["10", "11"], [S + "explanations/risk_report.py"], ["artifacts/predictions/attrition_risk_2026.csv"], ["test_risk_scoring::test_scores_and_bands_follow_the_company_rule", "test_risk_scoring::test_risk_bands_helper_behaviour"]),
 R("14.03", 14, "Top 3 drivers per employee with SHAP-style values: global importance, summary plot, local explanations, positive and negative", ["10"], [S + "explanations/shapley.py", S + "explanations/groups.py"], ["reports/figures/task14_shap_global_importance.png", "reports/figures/task14_shap_summary.png", T % 14 + "group_contributions.csv", T % 14 + "shap_library_feature_cross_check.csv", T % 14 + "shap_library_group_cross_check.csv"], ["test_shap_explanations::test_shapley_matches_the_exact_solution_for_a_linear_model", "test_shap_explanations::test_efficiency_property_is_exact", "test_shap_explanations::test_saved_explanation_run_is_additive_and_stable_enough", "test_shap_explanations::test_real_shap_permutation_cross_check_runs_when_library_is_available", "test_shap_explanations::test_saved_shap_library_cross_check_is_reproducible"], "actual SHAP PermutationExplainer cross-check is executed when the optional package is installed; grouped Monte Carlo Shapley remains the primary HR-group explainer"),
 R("14.04", 14, "Driver-to-sentence lookup in plain English; no certainty, no blame, no sensitive assumptions", ["10", "11"], [S + "explanations/reasons.py"], [], ["test_shap_explanations::test_sentences_are_plain_english_and_safe", "test_shap_explanations::test_sensitive_groups_never_reveal_values", "test_risk_scoring::test_reasons_do_not_state_certainty_or_protected_traits"]),
 R("14.05", 14, "Export 430 rows in the exact sample format: columns, order, types, ranges, ids, human-review wording", ["11"], [S + "explanations/risk_report.py"], [T % 14 + "attrition_risk_2026.csv"], ["test_risk_scoring::test_risk_report_passes_every_format_check", "test_risk_scoring::test_columns_rows_and_order_equal_the_sample", "test_risk_scoring::test_every_row_has_three_distinct_plain_reasons_and_the_human_review_note"]),
 R("14.06", 14, "Read 10 rows aloud; fix confusing or unfair wording; tenure wording for new joiners", ["11"], [S + "explanations/reasons.py"], [T % 14 + "manual_review_10_rows.csv"], ["test_outputs::test_every_task_has_its_evidence_files"]),
 R("14.07", 14, "Sample of 10 rows, SHAP summary chart, Task 14 report", ["10", "11"], [], [T % 14 + "manual_review_10_rows.csv", "reports/figures/task14_shap_summary.png", T % 14 + "README.md"], ["test_outputs::test_every_task_has_its_evidence_files"]),
 # ------------------------------------------------------------------ Task 15
 R("15.01", 15, "Open salary_training; choose job-related features; never protected ones", ["12"], [S + "salary/model.py"], [], ["test_salary_model::test_chosen_features_are_job_related_only", "test_salary_model::test_data_has_no_protected_columns_and_the_target_is_positive"]),
 R("15.02", 15, "Linear model, then gradient boosting; compare with the simple 'average by role' baseline", ["13"], [S + "salary/run.py"], [T % 15 + "cv_metrics.csv", T % 15 + "test_metrics.csv"], ["test_salary_model::test_role_mean_baseline_logic", "test_salary_model::test_model_beats_the_simple_baseline_on_the_test_set"]),
 R("15.03", 15, "Report MAE and R2 for each", ["13"], [S + "salary/model.py"], [T % 15 + "test_metrics.csv"], ["test_salary_model::test_metric_function"]),
 R("15.04", 15, "Use unlock_CX-15 to test gaps by gender and career gap with group_gap()", ["14"], [S + "salary/run.py"], [T % 15 + "gender_gaps.csv", T % 15 + "career_gap_means.csv", "reports/figures/task15_group_gaps.png"], ["test_salary_model::test_fairness_tables_exist_when_the_restricted_file_was_used"], "authorized restricted file supplied for this audit; 2,934/2,934 salary IDs matched; Task 15 rerun and notebooks 12-14 re-executed; file remains excluded from ZIP"),
 R("15.05", 15, "Output a recommended band (low, mid, high), not one number", ["14"], [S + "salary/model.py"], [T % 15 + "band_recommendations_test.csv"], ["test_salary_model::test_saved_model_makes_ordered_bands", "test_salary_model::test_band_output_is_a_range_not_a_single_number_and_has_no_protected_column", "test_salary_model::test_band_coverage_is_close_to_the_claimed_80_percent"]),
 R("15.06", 15, "Write what the gap means and what was changed; should the model copy past pay gaps", ["14"], [S + "salary/run.py"], [], ["test_salary_model::test_chosen_features_are_job_related_only"]),
 R("15.07", 15, "Limits written down; model saved with version and metadata; Task 15 report", ["14"], ["docs/limitations.md", "docs/model_cards/salary_band_v1.md"], ["artifacts/models/salary_band_v1.joblib", "artifacts/metadata/salary_band_v1_metadata.json", T % 15 + "README.md"], ["test_salary_model::test_metadata_and_artifact_checksum"]),
 # ------------------------------------------------------------------ Cross-cutting
 R("X.01", "all", "Notebooks follow the 38-stage workflow; N/A stages explained; every notebook executed with real outputs", ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15"], ["scripts/nbkit.py"], [], ["test_outputs::test_each_task_covers_all_38_workflow_stages", "test_outputs::test_all_notebooks_were_executed_without_errors_and_have_outputs", "test_outputs::test_na_stages_explain_why"]),
 R("X.02", "all", "Config in one file, seeds fixed, no hidden constants", [], ["config/phase3_config.yaml"], [], ["test_data_validation::test_config_is_consistent"]),
 R("X.03", "all", "Phase 1 and Phase 2 untouched; no Phase 1/2 model overwritten", [], [], [], ["test_attrition_v2::test_saved_model_artifacts_and_metadata"]),
 R("X.04", "all", "Documentation: architecture, methodology, features, limitations, model cards, experiment log, quality gate, traceability", [], ["docs/architecture.md", "docs/methodology.md", "docs/feature_definitions.md", "docs/limitations.md", "docs/experiment_log.md", "docs/phase3_quality_gate.md"], [], ["test_outputs::test_every_task_has_its_evidence_files"]),
]


def test_names() -> set[str]:
    found = set()
    for f in (ROOT / "tests").glob("test_*.py"):
        for n in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(n, ast.FunctionDef) and n.name.startswith("test_"): found.add(f"{f.stem}::{n.name}")
    return found


def build() -> list[dict]:
    tests = test_names(); rows = []
    for r in TRACE:
        problems = []
        nbs = []
        for p in r["notebooks"]:
            m = sorted((ROOT / "notebooks").glob(f"{p}_*.ipynb")); nbs += [x.name for x in m]
            if not m: problems.append(f"notebook {p} missing")
        for f in r["code"] + r["outputs"]:
            if not (ROOT / f).exists(): problems.append(f"missing file {f}")
        for t in r["tests"]:
            if t not in tests: problems.append(f"missing test {t}")
        status = "Complete" if not problems else "MISSING: " + "; ".join(problems)
        if not problems and r["note"]: status = "Complete with documented limitation"
        rows.append({**{k: ("; ".join(v) if isinstance(v, list) else v) for k, v in r.items() if k != "notebooks"}, "notebooks": "; ".join(nbs), "status": status})
    return rows


def write(rows: list[dict]):
    d = ROOT / "docs"; d.mkdir(exist_ok=True)
    cols = ["id", "task", "requirement", "notebooks", "code", "outputs", "tests", "status", "note"]
    with open(d / "requirements_traceability.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); [w.writerow({c: r[c] for c in cols}) for r in rows]
    n_ok = sum(r["status"].startswith("Complete") for r in rows)
    md = ["# Requirements traceability", "", f"{n_ok} of {len(rows)} tracked requirements are complete (a row is complete only if every notebook, code file, output file and test it names exists). Generated by `python scripts/make_docs.py`.", "",
          "| ID | Requirement | Notebooks | Code | Outputs | Tests | Status |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['id']} | {r['requirement']} | {r['notebooks']} | {r['code']} | {r['outputs']} | {r['tests'].replace('; ', '<br>')} | {r['status']}{(' (' + r['note'] + ')') if r['note'] else ''} |")
    (d / "requirements_traceability.md").write_text("\n".join(md), encoding="utf-8")
    return n_ok, len(rows)
