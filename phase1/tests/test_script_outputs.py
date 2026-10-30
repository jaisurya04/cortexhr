from _common import *
import json

REQUIRED = {
 "Task 1 (run_audit.py)": ["data_audit_report.md", "tables/audit_table.csv", "tables/leakage_review.csv", "tables/cleaning_report.csv", "tables/eda_numeric_vs_target.csv", "tables/audit_proof_models.csv",
                           "tables/audit_tuning_lr_C.csv", "tables/audit_interpretation_coefficients.csv", "tables/audit_interpretation_permutation_importance.csv",
                           "figures/audit_table.png", "figures/class_balance.png", "figures/eda_attrition_by_group.png", "figures/audit_confusion_matrices.png", "vault_comparison_cx01.md"],
 "Task 2 (run_pipeline.py)": ["tables/pipeline_3row_smoke_test_output.csv", "tables/feature_definitions.csv", "tables/pipeline_tuning_window_months.csv", "tables/pipeline_feature_set_comparison.csv",
                              "tables/new_feature_pay_vs_dept_quartiles.csv", "figures/pipeline_confusion_matrix.png"],
 "Task 3 (train_baselines.py)": ["baseline_model_comparison.md", "tables/baseline_comparison.csv", "tables/baseline_confusion_counts.csv", "tables/baseline_tuning.csv", "tables/imbalance_handling_comparison.csv",
                                 "tables/baseline_interpretation_coefficients.csv", "figures/baseline_confusion_matrices.png"],
 "Task 4 (evaluate_models.py)": ["model_evaluation_report.md", "manager_summary.md", "tables/evaluation_metrics.csv", "tables/risk_band_counts.csv", "tables/gains_lift_table.csv", "tables/calibration_table.csv",
                                 "tables/evaluation_threshold_sweep_train_oof.csv", "tables/evaluation_calibrator_comparison.csv", "tables/evaluation_band_profile.csv",
                                 "figures/calibration_curve.png", "figures/gains_lift_chart.png", "figures/evaluation_confusion_test.png"],
 "Task 5 (detect_attendance_anomalies.py)": ["attendance_anomaly_report.md", "tables/attendance_flags.csv", "tables/attendance_review_queue_top100.csv", "tables/attendance_audit_checks.csv",
                                             "tables/attendance_tuning_contamination.csv", "tables/attendance_temporal_split_stability.csv", "tables/attendance_interpretation_flagged_vs_normal.csv",
                                             "figures/attendance_eda.png", "figures/attendance_confusion_daily_sample.png"],
}

def test_every_task_script_wrote_its_outputs():
    missing = [f"{t}: {f}" for t, fs in REQUIRED.items() for f in fs if not (C.reports_dir() / f).exists()]
    assert not missing, missing

def test_pipeline_smoke_checks_file_is_clean():
    c = json.loads((C.project_root() / "artifacts" / "metadata" / "pipeline_smoke_test_checks.json").read_text())
    assert c["all_values_finite"] and not c["target_in_output"] and not c["leakage_column_in_output"] and c["n_test_rows_used_to_fit"] == 0


VAULT_DEPENDENT = ["tables/attendance_detector_metrics.csv", "tables/attendance_by_anomaly_type.csv", "tables/vault_cx01_rowlevel_comparison.csv"]

def test_vault_measurements_present_when_the_vault_is_present():
    v = C.project_root() / "04_Validation_Vault"
    if not (v / "unlock_CX-05_attendance_anomaly_labels.csv").exists(): return          # restricted files are not shipped
    missing = [f for f in VAULT_DEPENDENT if not (C.reports_dir() / f).exists()]; assert not missing, missing
