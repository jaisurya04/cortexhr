"""Requirement traceability matrix in the required 8-field format.   python scripts/build_traceability.py

Statuses are COMPUTED from outputs/last_test_run.json (written by tests/run_all.py), never typed in:
  PASS                  the referenced test executed and passed AND the row is fully covered by it
  PARTIAL               implemented, but the test covers only part of it, or no executed test covers it
  FAIL                  the referenced test failed (or is missing from the last run)
  BLOCKED               the referenced test could not run here (library not installable / model unavailable)
  USER ACTION REQUIRED  needs your account, phone, vault, recording or judgement
A source file merely existing never gives PASS; a skipped test is never PASS.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from _traceability_rows import INTERP, ROWS  # noqa: E402

DEL = "tests/test_deliverables.py::"; AL = "tests/test_ai_label.py::"; MC = "tests/test_model_cards.py::"; FC = "tests/test_forecast_core.py::"
DASH = "tests/test_dashboard.py::"; MON = "tests/test_monitoring.py::"; INT = "tests/test_phase_integrity.py::"; SV = "tests/test_services_models.py::"

# id -> (kind, test override or None, required correction)   kind: auto | partial | user
OVR = {
 "D16-T": ("partial", None, "Run scripts/run_forecast.py with statsmodels installed so the improved model is compared with the baseline."),
 "D16-3": ("auto", None, "Install pinned statsmodels (requirements.txt) and run scripts/run_forecast.py; SARIMAX was never executed in the build sandbox."),
 "D16-4": ("partial", None, "Events reach the model as regressors only when statsmodels runs; run the two statsmodels tests on your machine."),
 "D16-5": ("partial", None, "Baseline backtest is executed; the improved model's backtest error exists only after SARIMAX runs."),
 "D16-6": ("partial", None, "Saved forecast is the baseline. Rolling-origin interval coverage is ~68% (target 80%): do not claim 80% until re-run and re-measured."),
 "D17-T": ("partial", None, "Serve the real FastAPI (uvicorn) and deploy; the FastAPI tests are blocked here."),
 "D17-1": ("auto", DEL + "test_readme_lists_every_v1_route_defined_in_the_api_source", ""),
 "D17-2a": ("partial", None, "Install FastAPI/Pydantic and run tests/test_api_fastapi.py."),
 "D17-3": ("partial", None, "Run smoke_test.py against uvicorn (real FastAPI) and then the live URL; so far it ran against the Flask stand-in."),
 "D17-4": ("auto", DEL + "test_hosting_table_has_every_required_column_date_and_source_urls", "Re-open the source links before you decide (pages were not re-fetched on the last day)."),
 "D17-TH": ("partial", None, "Repeat scripts/load_test.py --users 100 against the deployed API; local stand-in numbers are not public-service numbers."),
 "D18-3": ("partial", None, "Career-gap pay gap and twin-resume re-run; the gender pay gap needs the restricted vault (reported NOT checked)."),
 "D18-6": ("partial", None, "Route needs real FastAPI; redeploy and check the live URL."),
 "D19-T": ("partial", AL + "test_console_has_exactly_the_five_official_tabs_and_attendance_is_reachable", "Run the real Streamlit console against a live API (AppTest/uvicorn test is blocked here)."),
 "D19-1": ("partial", AL + "test_console_has_exactly_the_five_official_tabs_and_attendance_is_reachable", "Verify the five tabs in a running Streamlit app (source-level check only so far)."),
 "D19-2": ("partial", None, "Run the console with CORTEXHR_API_URL pointing at a running API; client executed over HTTP, Streamlit not."),
 "D19-3": ("partial", AL + "test_every_ai_api_response_carries_the_exact_label", "API labels and wrapper enforcement are tested; rendering in real Streamlit is unverified."),
 "D19-4": ("partial", None, "Friendly messages are tested in the API client; verify the on-screen messages in real Streamlit."),
 "D20-T": ("partial", MC + "test_every_registry_model_has_a_card_in_the_template_layout", "Remaining Day 20 items (recording, tag, challenge) are yours."),
 "D20-1": ("auto", MC + "test_every_registry_model_has_a_card_in_the_template_layout", ""),
 "D20-2": ("auto", DEL + "test_architecture_picture_exists_and_is_a_png", ""),
 "D20-3": ("partial", MC + "test_live_performance_numbers_appear_in_the_cards", "Phase 4 evaluations re-run; Phase 1-3 evaluations are read from saved files (no retraining allowed). Re-run forecast with statsmodels."),
 "D20-4a": ("auto", DEL + "test_release_notes_has_included_not_included_and_limits", ""),
 "B-1": ("auto", None, ""),
 "D16-TH": ("partial", None, "A reproducibility test does not prove the written answer; read docs/forecasting_report.md section 5 and confirm the answer in your words."),
 "D17-L": ("partial", None, "Learning goals are covered by the code; no single test proves them."),
 "B-3": ("partial", None, "Chart is browser-tested; the Forecast tab shell needs real Streamlit."),
 "B-5": ("partial", None, "Behaviour proven on the stand-in (same handlers); confirm on real FastAPI."),
 "B-9": ("partial", DASH + "test_desktop_layout_kpis_and_no_console_errors", "Layout measured and screenshot compared (MAE 20.1, SSIM 0.56) but NOT pixel-identical: fonts, 3-D glyphs, waves and avatar differ."),
 "B-12": ("partial", None, "No logo file exists in the repo; assets are crops of the screenshot. Supply your original logo file."),
 "B-13": ("auto", AL + "test_every_ai_api_response_carries_the_exact_label", ""),
 "B-14": ("partial", None, "8 tests are blocked here (FastAPI, statsmodels, Streamlit); run them on your machine."),
 "B-15": ("auto", DEL + "test_readme_lists_every_v1_route_defined_in_the_api_source", ""),
 "B-16": ("auto", DEL + "test_tag_instructions_give_exact_commands_and_do_not_claim_a_tag", ""),
}
# extra granular rows: (id, source, requirement, implementation, test, evidence, kind, correction)
EX = [
 ("D16-C1", "Day 16 checklist", "Backtest error is reported for baseline AND model.", "forecasting/backtest.py", FC + "test_backtest_tables_are_honest_and_complete", "outputs/forecast/model_comparison.csv", "partial", "Model (SARIMAX) rows are missing until statsmodels runs."),
 ("D16-C2", "Day 16 checklist", "Interval is shown on a chart.", "console/charts.py", DASH + "test_charts_render_and_hover_tooltip_works", "outputs/console/forecast_chart.png", "auto", ""),
 ("D16-C3", "Day 16 checklist", "Events are handled.", "forecasting/events.py", FC + "test_events_classification_and_matrix", "outputs/forecast/events_applied.csv", "partial", "Effect on the model is test-blocked (statsmodels)."),
 ("D16-C4", "Day 16 checklist", "Output columns match the sample_forecast columns exactly (extras in a separate file).", "forecasting/pipeline.py", FC + "test_pipeline_output_matches_sample_forecast_layout", "outputs/forecast/forecast_next_6_months.csv; outputs/forecast/forecast_detailed_both_models.csv", "auto", ""),
 ("D16-B1", "Day 16 brief", "No leakage: holdout model never sees the future.", "forecasting/backtest.py", FC + "test_backtest_never_sees_the_future", "docs/forecasting_report.md", "auto", ""),
 ("D16-B2", "Day 16 brief", "Investigate interval coverage below 80% and calibrate honestly; never claim 80% without evidence.", "scripts/interval_calibration_study.py", FC + "test_empirical_interval_needs_enough_errors_and_is_symmetric", "outputs/forecast/interval_calibration_study.csv; docs/forecasting_report.md", "partial", "Walk-forward (adaptive conformal) calibration added; judge by rolling_origin_calibrated_walk_forward in model_comparison.csv; short overlapping history = indicative only."),
 ("D16-B3", "Day 16 brief", "Least-trusted month is computed from interval widths.", "forecasting/pipeline.py (explain)", FC + "test_pipeline_is_reproducible", "outputs/forecast/forecast_explanation.json", "partial", "Only reproducibility is tested; confirm the answer reads sensibly."),
 ("D17-C1", "Day 17 checklist", "/v1/health and every model endpoint work on the live address.", "scripts/smoke_test.py", "", "", "user", "Needs your deployment."),
 ("D17-C2", "Day 17 checklist", "Smoke test passes against the live URL.", "scripts/smoke_test.py", "", "", "user", "Run from your phone and save the output."),
 ("D17-C3", "Day 17 checklist", "No secret is in the repository.", "core/settings.py; .env.example", INT + "test_secrets_are_not_in_the_repository", "", "auto", ""),
 ("D17-C4", "Day 17 checklist", "Hosting choice and reason written down.", "docs/hosting_comparison.md", DEL + "test_hosting_table_has_every_required_column_date_and_source_urls", "docs/hosting_comparison.md", "auto", ""),
 ("D17-B1", "Day 17 brief", "Requirements are exactly pinned.", "requirements.txt", INT + "test_requirements_are_exactly_pinned", "requirements.txt", "auto", ""),
 ("D17-B2", "Day 17 brief", "Models are loaded once at startup; unavailable models give a clear message, not a crash.", "services/registry.py", SV + "test_registry_loads_every_model_and_is_honest_about_fallbacks", "/v1/models", "auto", ""),
 ("D17-B3", "Day 17 brief", "Auth, validation and friendly JSON errors on real FastAPI.", "api/app.py; api/schemas.py", "tests/test_api_fastapi.py::test_validation_errors_use_the_common_json_shape", "", "auto", "Blocked until FastAPI is installed."),
 ("D17-B4", "Day 17 brief", "Load test at 100 users; local/simulated distinguished from the real public service.", "scripts/load_test.py", DEL + "test_load_test_doc_separates_local_from_public_measurements", "docs/load_test_findings.md", "partial", "Public-service load test is yours."),
 ("D18-C1", "Day 18 checklist", "PSI table with a verdict per feature.", "monitoring/drift.py", MON + "test_drift_on_real_data_finds_the_collapsed_raise_feature", "outputs/monitoring/monitoring_report.md", "auto", ""),
 ("D18-C2", "Day 18 checklist", "Fairness checks re-run automatically (inside the scheduled monitor).", "monitoring/fairness.py; deploy/github-actions-monitor.yml", MON + "test_twin_resume_check_runs_and_reports_limit", "outputs/monitoring/monitoring_latest.json", "partial", "Runs in monitor.py; automatic scheduling is not active until you add the workflow."),
 ("D18-C3", "Day 18 checklist", "Schedule is running (a real successful scheduled run).", "deploy/github-actions-monitor.yml", "", "", "user", "Copy the workflow to .github/workflows/, enable Actions, wait for a run, screenshot it."),
 ("D18-C4", "Day 18 checklist", "Live endpoint returns the latest status.", "services/handlers.py (monitoring)", SV + "test_monitoring_handler_returns_status_and_history", "", "user", "Redeploy and open /v1/monitoring on the live URL."),
 ("D18-B1", "Day 18 brief", "Gender pay-gap is never claimed passed without the restricted vault.", "monitoring/fairness.py", MON + "test_gender_gap_is_reported_as_not_checked_without_the_vault", "outputs/monitoring/monitoring_report.md", "auto", ""),
 ("D18-B2", "Day 18 brief", "Eight features, status thresholds, history of runs, 4-step ALERT plan.", "monitoring/monitor.py", MON + "test_monitor_report_structure_and_alert_plan", "outputs/monitoring/monitoring_history.jsonl", "auto", ""),
 ("D19-C1", "Day 19 checklist", "A stranger can use it without help.", "docs/console_guide.md", "", "", "user", "Needs a real person; note what they got stuck on."),
 ("D19-C2", "Day 19 checklist", "Every output carries the label 'AI-generated, for human review' (impossible to omit).", "console/ai_render.py; console/app.py", AL + "test_console_pages_never_draw_results_without_the_label_wrappers", "", "partial", "Source-level enforcement tested; confirm on screen in real Streamlit."),
 ("D19-C3", "Day 19 checklist", "Errors show friendly messages.", "console/api_client.py", "tests/test_api_http.py::test_client_friendly_messages_for_each_failure_mode", "", "partial", "Verify on screen in real Streamlit."),
 ("D19-C4", "Day 19 checklist", "Console and API are both live.", "docs/deployment.md", "", "", "user", "Deploy both."),
 ("D19-B1", "Day 19 brief", "Exactly five tabs, each calling the live API; API URL from an environment variable.", "console/app.py", AL + "test_console_has_exactly_the_five_official_tabs_and_attendance_is_reachable", "", "partial", "Run real Streamlit."),
 ("D19-B2", "Day 19 brief", "Real Streamlit run verified on desktop and mobile; screenshots of at least 3 real tabs.", "tests/test_ai_label.py (AppTest)", AL + "test_real_streamlit_console_shows_the_label_on_every_tab", "", "auto", "Blocked: Streamlit not installable here. No real-tab screenshots exist."),
 ("D19-B3", "Day 19 brief", "Label present on dashboard panels showing model output (browser-rendered).", "console/dashboard.py", AL + "test_dashboard_panels_with_model_output_show_the_label_in_the_browser", "outputs/console/reference_match/dashboard_with_popup_1365x768.png", "auto", ""),
 ("D19-B4", "Day 19 brief", "Responsive on phones (no horizontal overflow).", "static/reference_skin.css", DASH + "test_mobile_layout_has_no_horizontal_overflow_and_popup_fits", "outputs/console/reference_match/dashboard_mobile_390x844.png", "auto", ""),
 ("D19-B5", "Day 19 brief", "Greeting by Asia/Kolkata time with the matching animation; welcome popup centered and dismissible.", "console/dashboard.py; dashboard.js", DASH + "test_welcome_popup_opens_closes_and_is_remembered", "outputs/console/reference_match/greeting_afternoon_14h.png; outputs/console/reference_match/greeting_evening_19h.png; outputs/console/reference_match/greeting_night_23h.png", "partial", "Popup behaviour is tested; the greeting/animation per hour was checked by screenshots (afternoon, evening, night) - morning not screenshotted."),
 ("D19-B6", "Day 19 brief", "No reference-screenshot demo numbers in live mode.", "console/design_reference.py", DASH + "test_no_demo_values_in_the_generated_page", "", "auto", ""),
 ("D20-C1", "Day 20 checklist", "Every model has a model card (original template headings).", "docs/model_cards/", MC + "test_every_registry_model_has_a_card_in_the_template_layout", "docs/model_cards/", "auto", ""),
 ("D20-C2", "Day 20 checklist", "Numbers in cards match the service (/v1/performance) and the evaluation files.", "scripts/build_docs.py", MC + "test_cards_on_disk_equal_cards_regenerated_from_the_evaluation_files", "docs/model_cards/", "partial", "Matches the handlers/files here; compare with your LIVE service after deploying."),
 ("D20-C3", "Day 20 checklist", "Release notes written (included / not included / known limits).", "RELEASE_NOTES.md", DEL + "test_release_notes_has_included_not_included_and_limits", "RELEASE_NOTES.md", "auto", ""),
 ("D20-C4", "Day 20 checklist", "Demo recorded and link tested.", "docs/demo_script_5min.md; docs/demo_recording_checklist.md", DEL + "test_demo_script_and_recording_checklist_exist", "", "user", "Script and checklist exist; the recording is yours."),
 ("D20-B1", "Day 20 brief", "Hand-typed numbers in cards are re-derived from their sources (no stale numbers).", "docs/model_cards/", MC + "test_hand_typed_numbers_in_cards_are_rederived_from_their_sources", "", "auto", ""),
 ("D20-B2", "Day 20 brief", "v1.0.0 tag prepared (exact commands + evidence needed) but NOT created.", "docs/release_tag_instructions.md", DEL + "test_tag_instructions_give_exact_commands_and_do_not_claim_a_tag", "docs/release_tag_instructions.md", "user", "Tag only after every condition passes."),
 ("D20-B3", "Day 20 brief", "Final Challenge: workflow, safe starting point and record template only; not done by the agent.", "docs/final_challenge_workflow.md", DEL + "test_final_challenge_is_workflow_only_and_judgement_is_a_labelled_draft", "", "user", "Your own unguided improvement."),
 ("D20-B4", "Day 20 brief", "Final submission checklist and user-action checklist.", "docs/final_acceptance_checklist.md; docs/user_action_checklist.md", DEL + "test_final_checklist_and_user_action_list_exist", "", "auto", ""),
 ("D20-B5", "Day 20 brief", "Consolidated final report with an honest release verdict.", "docs/final_report.md", DEL + "test_final_report_exists_and_does_not_claim_public_release", "docs/final_report.md", "auto", ""),
 ("B-P1", "Brief: protection", "Phase 1-3 files unchanged (compared with the pristine repository before and after).", "tests/run_all.py", INT + "test_phase4_has_no_files_outside_its_folder_and_repo_phases_match_baseline", "", "auto", "Needs CORTEXHR_PRISTINE_REPO for the byte comparison."),
 ("B-P2", "Brief: protection", "Phase 4 code never writes into Phase 1-3.", "core/legacy.py", INT + "test_no_phase4_source_writes_into_phase1_to_3", "", "auto", ""),
 ("B-T1", "Brief: testing", "Final test runner reports Passed / Failed / Skipped / Blocked / User actions separately.", "tests/run_all.py", "", "outputs/last_test_run.json", "partial", "Executed by running it; no test asserts its own output format."),
]
BLOCK_WORDS = ("not installed", "model_unavailable")


def load_results() -> dict:
    p = ROOT / "outputs" / "last_test_run.json"
    if not p.exists():
        raise SystemExit("run `python tests/run_all.py` first: statuses are computed from outputs/last_test_run.json")
    return json.loads(p.read_text(encoding="utf-8"))


def key(ref: str) -> str:
    f, fn = ref.split("::")
    return f.replace("tests/", "").replace(".py", "") + "::" + fn


def evaluate(kind: str, test: str, results: dict) -> tuple[str, str]:
    r = results.get(key(test)) if test else None
    if kind == "user":
        base = "USER ACTION REQUIRED"
        if r:
            return base, f"Supporting test {r['status']} ({test.split('::')[1]}); the action itself is not done."
        return base, "Not done by the build; needs you."
    if not test:
        return "PARTIAL", "No executed test covers this; the file/document was inspected only."
    if r is None:
        return "FAIL", f"{test} was not in the last test run (renamed or not executed)."
    if r["status"] == "FAIL":
        return "FAIL", "FAILED: " + r["note"]
    if r["status"] == "SKIP":
        return ("BLOCKED" if any(w in r["note"] for w in BLOCK_WORDS) else "PARTIAL"), "NOT EXECUTED (skipped): " + r["note"]
    return ("PASS" if kind == "auto" else "PARTIAL"), f"PASSED in {r['seconds']}s"


def main() -> None:
    data = load_results(); results = data["results"]
    out = []
    for (id_, src, req, impl, test, evid, old, note) in ROWS:
        kind0 = {"Done (executed)": "auto", "Done (written, NOT executed here)": "partial", "Your action": "user"}[old]
        kind, t, corr = OVR.get(id_, (kind0, None, ""))
        test = t or test
        if kind0 == "user":
            kind = "user"
        if not corr:
            corr = note
        out.append((id_, src, req, impl, test, evid, kind, corr))
    out += EX
    rows = []
    for id_, src, req, impl, test, evid, kind, corr in out:
        status, actual = evaluate(kind, test, results)
        if status == "PASS":
            corr = corr or "none needed"
        cmd = f"python tests/run_all.py -k {key(test)}" if test else ("(none - user action)" if kind == "user" else "(no automated test)")
        rows.append({"task_id": id_, "source": src, "requirement": req, "implementation": impl, "required_correction": corr or "see note in docs/final_report.md",
                     "validation_command": cmd, "actual_result": actual, "evidence": evid or ("none yet - needs your action" if status == "USER ACTION REQUIRED" else "none beyond the test result"), "status": status, "test": test})
    cols = ["task_id", "requirement", "implementation", "required_correction", "validation_command", "actual_result", "evidence", "status", "source", "test"]
    with (ROOT / "docs" / "requirements_traceability.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
    n: dict[str, int] = {}
    for r in rows:
        n[r["status"]] = n.get(r["status"], 0) + 1
    md = ["# Requirement traceability matrix (Phase 4)\n",
          f"Generated from `outputs/last_test_run.json` (test run finished {data['finished']}; {len(results)} tests). {len(rows)} requirements: " + ", ".join(f"**{k} {v}**" for k, v in sorted(n.items())) + ".\n",
          "Statuses: **PASS** = the referenced test executed and passed and covers the whole row; **PARTIAL** = implemented but only partly covered/unexecuted; **FAIL**; **BLOCKED** = the test could not run here (library not installable); **USER ACTION REQUIRED** = needs your account, phone, vault, recording or judgement. A file existing is never PASS; a skipped test is never PASS.\n",
          "| ID | Requirement | Implementation | Required correction | Validation command | Actual result | Evidence | Status |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append("| " + " | ".join(str(r[c] or "-").replace("|", "/").replace("\n", " ") for c in ("task_id", "requirement", "implementation", "required_correction", "validation_command", "actual_result", "evidence", "status")) + " |")
    md += ["", "## Interpretations and unclear text (identified, not guessed)", *[f"* {x}" for x in INTERP], ""]
    (ROOT / "docs" / "requirements_traceability.md").write_text("\n".join(md), encoding="utf-8")
    print(len(rows), n)


if __name__ == "__main__":
    main()
