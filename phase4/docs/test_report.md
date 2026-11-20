# Phase 4 test report

Run at 2026-11-18 08:06 UTC with `python tests/run_all.py` on Python 3.13.18, pandas 3.0.5, scikit-learn 1.9.1.

Libraries in this environment: fastapi NOT installed, pydantic installed, statsmodels NOT installed, streamlit NOT installed, httpx NOT installed, pytest installed, playwright installed, flask installed, shap NOT installed.

**PASS** = executed and passed. **SKIP** = NOT executed (reason shown); skipped tests are not claimed as passed. The last lines prove that no Phase 1-3 file changed during the run.

```
PASS test_ai_label::test_console_has_exactly_the_five_official_tabs_and_attendance_is_reachable (0.0s)
PASS test_ai_label::test_console_pages_never_draw_results_without_the_label_wrappers (0.1s)
PASS test_ai_label::test_dashboard_panels_with_model_output_show_the_label_in_the_browser (1.2s)
PASS test_ai_label::test_every_ai_api_response_carries_the_exact_label (0.3s)
PASS test_ai_label::test_every_wrapper_draws_the_label_before_the_content (0.0s)
SKIP test_ai_label::test_real_streamlit_console_shows_the_label_on_every_tab (0.1s)  - streamlit is not installed in this environment
SKIP test_ai_label::test_scenario_forecast_carries_the_label (0.0s)  - what-if forecast needs the improved (statsmodels) model: model_unavailable
PASS test_ai_label::test_wrappers_put_the_label_first_even_when_drawing_the_content_fails (0.0s)
SKIP test_api_fastapi::test_each_model_endpoint_returns_a_real_prediction (0.0s)  - fastapi is not installed in this environment
SKIP test_api_fastapi::test_health_open_and_models_protected_when_key_set (0.0s)  - fastapi is not installed in this environment
SKIP test_api_fastapi::test_openapi_lists_every_v1_route (0.0s)  - fastapi is not installed in this environment
PASS test_api_fastapi::test_pydantic_schemas_reject_impossible_values (0.1s)
SKIP test_api_fastapi::test_validation_errors_use_the_common_json_shape (0.0s)  - fastapi is not installed in this environment
PASS test_api_http::test_client_friendly_messages_for_each_failure_mode (0.5s)
PASS test_api_http::test_every_documented_get_endpoint_returns_json (0.6s)
PASS test_api_http::test_smoke_test_script_fails_loudly_when_api_is_down (0.1s)
PASS test_api_http::test_smoke_test_script_passes_every_model (0.7s)
PASS test_dashboard::test_charts_render_and_hover_tooltip_works (0.7s)
PASS test_dashboard::test_desktop_layout_kpis_and_no_console_errors (0.7s)
PASS test_dashboard::test_mobile_layout_has_no_horizontal_overflow_and_popup_fits (1.6s)
PASS test_dashboard::test_no_demo_values_in_the_generated_page (0.0s)
PASS test_dashboard::test_period_search_menus_and_modals_work (1.3s)
PASS test_dashboard::test_welcome_popup_opens_closes_and_is_remembered (2.0s)
PASS test_deliverables::test_architecture_picture_exists_and_is_a_png (0.0s)
PASS test_deliverables::test_demo_script_and_recording_checklist_exist (0.0s)
PASS test_deliverables::test_final_challenge_is_workflow_only_and_judgement_is_a_labelled_draft (0.0s)
PASS test_deliverables::test_final_checklist_and_user_action_list_exist (0.0s)
PASS test_deliverables::test_final_report_exists_and_does_not_claim_public_release (0.0s)
PASS test_deliverables::test_forecast_chart_and_backtest_table_evidence_exist (0.0s)
PASS test_deliverables::test_hosting_table_has_every_required_column_date_and_source_urls (0.0s)
PASS test_deliverables::test_load_test_doc_separates_local_from_public_measurements (0.0s)
PASS test_deliverables::test_readme_lists_every_v1_route_defined_in_the_api_source (0.0s)
PASS test_deliverables::test_reference_screenshot_comparison_evidence_exists_and_states_its_limits (0.0s)
PASS test_deliverables::test_release_notes_has_included_not_included_and_limits (0.0s)
PASS test_deliverables::test_requirements_are_pinned (0.0s)
PASS test_deliverables::test_tag_instructions_give_exact_commands_and_do_not_claim_a_tag (0.0s)
PASS test_deliverables::test_workflow_template_is_not_active_in_the_repository (0.0s)
PASS test_forecast_core::test_ar1_cumulative_sd_matches_simulation (0.0s)
PASS test_forecast_core::test_backtest_never_sees_the_future (0.0s)
PASS test_forecast_core::test_backtest_tables_are_honest_and_complete (0.3s)
PASS test_forecast_core::test_data_validation_rejects_broken_history (0.0s)
PASS test_forecast_core::test_eda_describes_trend_seasons_and_unusual_months (0.3s)
PASS test_forecast_core::test_empirical_interval_needs_enough_errors_and_is_symmetric (0.0s)
PASS test_forecast_core::test_events_classification_and_matrix (0.0s)
PASS test_forecast_core::test_forecast_only_refuses_when_it_would_ignore_the_scenario (0.0s)
PASS test_forecast_core::test_history_matches_sample_series_ids (0.0s)
PASS test_forecast_core::test_horizon_validation (0.0s)
PASS test_forecast_core::test_improved_model_status_is_reported_not_hidden (0.1s)
PASS test_forecast_core::test_metrics_hand_computed (0.0s)
PASS test_forecast_core::test_pipeline_is_reproducible (0.4s)
PASS test_forecast_core::test_pipeline_output_matches_sample_forecast_layout (0.2s)
PASS test_forecast_core::test_rolling_errors_counts (0.0s)
PASS test_forecast_core::test_seasonal_mean_formula (0.0s)
PASS test_forecast_core::test_seasonal_naive_drift_formula (0.0s)
SKIP test_forecast_core::test_statsmodels_event_regressor_changes_forecast (0.0s)  - statsmodels is not installed in this environment
SKIP test_forecast_core::test_statsmodels_level_and_count_models (0.0s)  - statsmodels is not installed in this environment
PASS test_model_cards::test_cards_on_disk_equal_cards_regenerated_from_the_evaluation_files (0.1s)
PASS test_model_cards::test_every_registry_model_has_a_card_in_the_template_layout (0.0s)
PASS test_model_cards::test_hand_typed_numbers_in_cards_are_rederived_from_their_sources (0.0s)
PASS test_model_cards::test_live_performance_numbers_appear_in_the_cards (0.0s)
PASS test_monitoring::test_drift_on_real_data_finds_the_collapsed_raise_feature (0.0s)
PASS test_monitoring::test_feature_selection_uses_importance_and_numeric_shared_columns (0.0s)
PASS test_monitoring::test_gender_gap_is_reported_as_not_checked_without_the_vault (0.2s)
PASS test_monitoring::test_monitor_failed_check_is_visible_not_silent (0.2s)
PASS test_monitoring::test_monitor_is_reproducible (0.8s)
PASS test_monitoring::test_monitor_persist_appends_audit_history (0.4s)
PASS test_monitoring::test_monitor_report_structure_and_alert_plan (0.4s)
PASS test_monitoring::test_other_status_rules (0.0s)
PASS test_monitoring::test_pay_gap_reproduces_phase3_saved_numbers (0.2s)
PASS test_monitoring::test_psi_helper_behaviour_on_synthetic_data (0.0s)
PASS test_monitoring::test_psi_status_boundaries (0.0s)
PASS test_monitoring::test_single_comparable_group_is_never_reported_ok (0.2s)
PASS test_monitoring::test_twin_resume_check_runs_and_reports_limit (0.2s)
PASS test_phase_integrity::test_no_phase4_source_writes_into_phase1_to_3 (0.0s)
PASS test_phase_integrity::test_phase4_has_no_files_outside_its_folder_and_repo_phases_match_baseline (0.2s)
PASS test_phase_integrity::test_requirements_are_exactly_pinned (0.0s)
PASS test_phase_integrity::test_secrets_are_not_in_the_repository (0.0s)
PASS test_requirements_traceability::test_every_row_points_to_real_files_and_real_tests (0.0s)
PASS test_requirements_traceability::test_no_row_is_pass_unless_its_test_passed_in_the_last_run (0.0s)
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

==============================================================================
PASSED   86   (executed and passed)
FAILED   0
SKIPPED  0   (not executed; never counted as passed)
BLOCKED  8   (a dependency could not be installed or a model was unavailable in this environment)
           - test_ai_label::test_real_streamlit_console_shows_the_label_on_every_tab | streamlit is not installed in this environment
           - test_ai_label::test_scenario_forecast_carries_the_label | what-if forecast needs the improved (statsmodels) model: model_unavailable
           - test_api_fastapi::test_each_model_endpoint_returns_a_real_prediction | fastapi is not installed in this environment
           - test_api_fastapi::test_health_open_and_models_protected_when_key_set | fastapi is not installed in this environment
           - test_api_fastapi::test_openapi_lists_every_v1_route | fastapi is not installed in this environment
           - test_api_fastapi::test_validation_errors_use_the_common_json_shape | fastapi is not installed in this environment
           - test_forecast_core::test_statsmodels_event_regressor_changes_forecast | statsmodels is not installed in this environment
           - test_forecast_core::test_statsmodels_level_and_count_models | statsmodels is not installed in this environment
USER ACTIONS REMAINING  23   (need your account, phone, vault or judgement; see docs/requirements_traceability.csv)
           - D17-5: Deploy with every setting as an environment variable. No secret goes in the repository.
           - D17-6: Run the smoke test against the live address from your phone.
           - D17-E: Evidence: Live API link; smoke test output; hosting choice and reason.
           - D18-5: Schedule it, for example GitHub Actions on a timer or a cron job.
           - D18-E: Evidence: PSI table; screenshot of the scheduled run.
           - D19-5: Deploy the console to a public address.
           - D19-6: Test on a phone and ask a stranger to use it without help.
           - D19-TH: Think for yourself: Which tab would a real HR manager use first? Does your console make th
           - D19-E: Evidence: Console link; screenshots of 3 tabs.
           - D20-4b: Tag v1.0.0.
           - D20-5: Finish your consolidated report and record your 5-minute demo.
           - D20-6: Final Challenge: improve one thing on its own branch, with no guide, and describe what you
           - D20-TH: Think for yourself: Which model would you not release, if you were the manager? Write down
           - D20-E: Evidence: Link to the v1.0.0 tag; model cards; Final Challenge notes.
           - D17-C1: /v1/health and every model endpoint work on the live address.
           - D17-C2: Smoke test passes against the live URL.
           - D18-C3: Schedule is running (a real successful scheduled run).
           - D18-C4: Live endpoint returns the latest status.
           - D19-C1: A stranger can use it without help.
           - D19-C4: Console and API are both live.
           - D20-C4: Demo recorded and link tested.
           - D20-B2: v1.0.0 tag prepared (exact commands + evidence needed) but NOT created.
           - D20-B3: Final Challenge: workflow, safe starting point and record template only; not done by the a
Phase 1-3 files checked: 724; changed during this run: 0    (16s)
RELEASE GATE: NOT READY - see FAILED / BLOCKED / USER ACTIONS above
==============================================================================
```

Exit code: 2 (0 = no failures and no Phase 1-3 file changed).
