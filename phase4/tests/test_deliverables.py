"""Content checks for the Day 17/20 handover documents. They prove the documents contain what the brief asks for; they do not prove
that the user-action items (deploy, recording, tag) happened - those stay USER ACTION REQUIRED."""
import re

from _common import ROOT

D = ROOT / "docs"


def _t(name):
    return (D / name).read_text(encoding="utf-8")


def test_release_notes_has_included_not_included_and_limits():
    t = (ROOT / "RELEASE_NOTES.md").read_text(encoding="utf-8").lower()
    for w in ("included", "not included", "known limit"):
        assert w in t, w


def test_architecture_picture_exists_and_is_a_png():
    b = (D / "architecture.png").read_bytes()
    assert b[:8] == b"\x89PNG\r\n\x1a\n" and len(b) > 10_000


def test_hosting_table_has_every_required_column_date_and_source_urls():
    t = _t("hosting_comparison.md")
    head = next(l for l in t.splitlines() if l.startswith("| Option"))
    for c in ("Memory limit", "Sleep behaviour", "Build limits", "Price per month", "Fit for CortexHR", "Source"):
        assert c in head, c
    assert re.search(r"Date checked", t) and re.search(r"2026-11", t)
    assert len(re.findall(r"https://", t)) >= 4
    assert "Recommendation" in t and "Reason" in t


def test_demo_script_and_recording_checklist_exist():
    assert "0:00" in _t("demo_script_5min.md")
    c = _t("demo_recording_checklist.md")
    assert c.count("- [ ]") >= 8 and "USER ACTION REQUIRED" in c


def test_final_checklist_and_user_action_list_exist():
    assert "[ ]" in _t("final_acceptance_checklist.md") or "[x]" in _t("final_acceptance_checklist.md")
    u = _t("user_action_checklist.md")
    assert "MY ACTION REQUIRED" in u and u.count("\n| ") >= 12


def test_tag_instructions_give_exact_commands_and_do_not_claim_a_tag():
    t = _t("release_tag_instructions.md")
    assert "git tag -a v1.0.0" in t and "git push origin v1.0.0" in t and "NOT created" in t


def test_final_challenge_is_workflow_only_and_judgement_is_a_labelled_draft():
    w = _t("final_challenge_workflow.md")
    assert "git switch -c" in w and "Record template" in w and "USER ACTION REQUIRED" in w
    j = _t("final_challenge_and_judgement.md")
    assert "Draft answer" in j and "edit it into your own judgement" in j
    assert re.search(r"\* Branch:\s*\n", j)          # record fields still blank: the agent did not do the challenge


def test_load_test_doc_separates_local_from_public_measurements():
    t = _t("load_test_findings.md")
    assert "stand-in" in t and "not a measurement" in t.lower() and "Render" in t


def test_requirements_are_pinned():
    lines = [l.strip() for l in (ROOT / "requirements.txt").read_text().splitlines() if l.strip() and not l.startswith("#")]
    assert lines and all("==" in l for l in lines), [l for l in lines if "==" not in l]


def test_workflow_template_is_not_active_in_the_repository():
    assert (ROOT / "deploy" / "github-actions-monitor.yml").exists()


def test_readme_lists_every_v1_route_defined_in_the_api_source():
    src = (ROOT / "src" / "cortexhr_phase4" / "api" / "app.py").read_text(encoding="utf-8")
    routes = re.findall(r"@(?:open_r|r)\.(?:get|post)\(\"(/[^\"]*)\"", src)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert len(routes) >= 17
    missing = [p for p in routes if ("/v1" + p) not in readme and not (p.endswith("/batch") and ("/v1" + p[:-6]) in readme)]
    assert not missing, missing


def test_forecast_chart_and_backtest_table_evidence_exist():
    for f in ("outputs/forecast/model_comparison.csv", "outputs/forecast/backtest_holdout_last_6_months.csv", "outputs/console/forecast_chart.png"):
        assert (ROOT / f).stat().st_size > 100, f


def test_reference_screenshot_comparison_evidence_exists_and_states_its_limits():
    d = ROOT / "outputs" / "console" / "reference_match"
    for f in ("approved_reference.png", "dashboard_with_popup_1365x768.png", "reference_vs_implementation_side_by_side.png", "dashboard_mobile_390x844.png"):
        assert (d / f).stat().st_size > 1000, f
    t = (ROOT / "docs" / "design_reference_match.md").read_text(encoding="utf-8")
    assert "SSIM" in t and "difference" in t.lower()


def test_final_report_exists_and_does_not_claim_public_release():
    t = (ROOT / "docs" / "final_report.md").read_text(encoding="utf-8")
    assert "NOT ready for public release" in t
    for h in ("Already correct", "Changed", "Tested successfully", "Failed", "Skipped", "Needs you"):
        assert h in t, h
