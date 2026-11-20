# Phase 4 test report

Run at 2026-11-17 15:56 UTC with `python tests/run_all.py` on Python 3.13.18, pandas 3.0.5, scikit-learn 1.9.1.

Libraries in this environment: fastapi NOT installed, pydantic installed, statsmodels NOT installed, streamlit NOT installed, httpx NOT installed, pytest installed, playwright installed, flask installed, shap NOT installed.

**PASS** = executed and passed. **SKIP** = NOT executed (reason shown); skipped tests are not claimed as passed. The last lines prove that no Phase 1-3 file changed during the run.

```
SKIP test_api_fastapi::test_each_model_endpoint_returns_a_real_prediction (0.3s)  - fastapi is not installed in this environment
SKIP test_api_fastapi::test_health_open_and_models_protected_when_key_set (0.0s)  - fastapi is not installed in this environment
SKIP test_api_fastapi::test_openapi_lists_every_v1_route (0.0s)  - fastapi is not installed in this environment
PASS test_api_fastapi::test_pydantic_schemas_reject_impossible_values (0.1s)
SKIP test_api_fastapi::test_validation_errors_use_the_common_json_shape (0.0s)  - fastapi is not installed in this environment
PASS test_api_http::test_client_friendly_messages_for_each_failure_mode (1.3s)
PASS test_api_http::test_every_documented_get_endpoint_returns_json (0.6s)
PASS test_api_http::test_smoke_test_script_fails_loudly_when_api_is_down (0.1s)
PASS test_api_http::test_smoke_test_script_passes_every_model (0.8s)
PASS test_dashboard::test_charts_render_and_hover_tooltip_works (0.9s)
PASS test_dashboard::test_desktop_layout_kpis_and_no_console_errors (0.9s)
PASS test_dashboard::test_mobile_layout_has_no_horizontal_overflow_and_popup_fits (1.7s)
PASS test_dashboard::test_no_demo_values_in_the_generated_page (0.0s)
PASS test_dashboard::test_period_search_menus_and_modals_work (1.3s)
PASS test_dashboard::test_welcome_popup_opens_closes_and_is_remembered (2.0s)
PASS test_forecast_core::test_ar1_cumulative_sd_matches_simulation (0.0s)
PASS test_forecast_core::test_backtest_never_sees_the_future (0.0s)
PASS test_forecast_core::test_backtest_tables_are_honest_and_complete (0.3s)
PASS test_forecast_core::test_data_validation_rejects_broken_history (0.0s)
PASS test_forecast_core::test_eda_describes_trend_seasons_and_unusual_months (0.4s)
PASS test_forecast_core::test_empirical_interval_needs_enough_errors_and_is_symmetric (0.0s)
PASS test_forecast_core::test_events_classification_and_matrix (0.0s)
PASS test_forecast_core::test_forecast_only_refuses_when_it_would_ignore_the_scenario (0.0s)
PASS test_forecast_core::test_history_matches_sample_series_ids (0.0s)
PASS test_forecast_core::test_horizon_validation (0.0s)
PASS test_forecast_core::test_improved_model_status_is_reported_not_hidden (0.1s)
PASS test_forecast_core::test_metrics_hand_computed (0.0s)
PASS test_forecast_core::test_pipeline_is_reproducible (0.5s)
PASS test_forecast_core::test_pipeline_output_matches_sample_forecast_layout (0.2s)
PASS test_forecast_core::test_rolling_errors_counts (0.0s)
PASS test_forecast_core::test_seasonal_mean_formula (0.0s)
PASS test_forecast_core::test_seasonal_naive_drift_formula (0.0s)
SKIP test_forecast_core::test_statsmodels_event_regressor_changes_forecast (0.0s)  - statsmodels is not installed in this environment
SKIP test_forecast_core::test_statsmodels_level_and_count_models (0.0s)  - statsmodels is not installed in this environment
PASS test_monitoring::test_drift_on_real_data_finds_the_collapsed_raise_feature (0.0s)
PASS test_monitoring::test_feature_selection_uses_importance_and_numeric_shared_columns (0.0s)
PASS test_monitoring::test_gender_gap_is_reported_as_not_checked_without_the_vault (0.2s)
PASS test_monitoring::test_monitor_failed_check_is_visible_not_silent (0.3s)
PASS test_monitoring::test_monitor_is_reproducible (1.1s)
PASS test_monitoring::test_monitor_persist_appends_audit_history (0.4s)
PASS test_monitoring::test_monitor_report_structure_and_alert_plan (0.4s)
PASS test_monitoring::test_other_status_rules (0.0s)
PASS test_monitoring::test_pay_gap_reproduces_phase3_saved_numbers (0.3s)
PASS test_monitoring::test_psi_helper_behaviour_on_synthetic_data (0.0s)
PASS test_monitoring::test_psi_status_boundaries (0.0s)
PASS test_monitoring::test_single_comparable_group_is_never_reported_ok (0.2s)
PASS test_monitoring::test_twin_resume_check_runs_and_reports_limit (0.2s)
PASS test_phase_integrity::test_no_phase4_source_writes_into_phase1_to_3 (0.0s)
PASS test_phase_integrity::test_phase4_has_no_files_outside_its_folder_and_repo_phases_match_baseline (0.2s)
PASS test_phase_integrity::test_requirements_are_exactly_pinned (0.0s)
PASS test_phase_integrity::test_secrets_are_not_in_the_repository (0.0s)
PASS test_requirements_traceability::test_every_row_points_to_real_files_and_real_tests (0.0s)
PASS test_services_models::test_attrition_list_bands_equal_phase3_counts (0.0s)
PASS test_services_models::test_attrition_predict_matches_phase3_report (0.0s)
PASS test_services_models::test_chatbot_answers_policy_question_and_rejects_blank (0.0s)
PASS test_services_models::test_concurrent_monitoring_refresh_runs_once_and_never_returns_a_broken_file (0.5s)
PASS test_services_models::test_forecast_endpoint_matches_saved_forecast_file (0.0s)
PASS test_services_models::test_health_and_models_shape (0.0s)
PASS test_services_models::test_jsonable_handles_numpy_nan_and_timestamps (0.0s)
PASS test_services_models::test_monitoring_handler_returns_status_and_history (0.0s)
PASS test_services_models::test_overview_numbers_come_from_the_data_not_the_screenshot (0.0s)
PASS test_services_models::test_payband_low_mid_high_and_validation (0.0s)
PASS test_services_models::test_payroll_and_attendance_use_the_real_detectors (0.3s)
PASS test_services_models::test_payroll_invalid_row_is_a_clean_error_not_a_crash (0.0s)
PASS test_services_models::test_performance_endpoint_has_every_evaluation_table (0.0s)
PASS test_services_models::test_registry_loads_every_model_and_is_honest_about_fallbacks (0.0s)
PASS test_services_models::test_resume_ranking_sorted_and_protected_fields_refused (0.0s)

Phase 1-3 files checked: 724; changed during this run: 0 
61 passed, 0 failed, 6 skipped (skipped = not executed) in 17s
```

Exit code: 0 (0 = no failures and no Phase 1-3 file changed).
