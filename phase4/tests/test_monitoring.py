"""Day 18: PSI, status transitions, fairness re-checks, monitor report, history."""
import json
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from _common import csv_dir

from cortexhr_phase4.core.settings import PHASE3, load_settings
from cortexhr_phase4.monitoring import drift, fairness, monitor, status


def test_psi_status_boundaries():
    assert status.psi_status(0.0) == status.OK and status.psi_status(0.0999) == status.OK
    assert status.psi_status(0.10) == status.WATCH and status.psi_status(0.25) == status.WATCH
    assert status.psi_status(0.2501) == status.ALERT and status.psi_status(3.8) == status.ALERT
    assert status.psi_status(float("nan")) == status.WATCH, "an unjudgeable PSI must never be reported as OK"


def test_other_status_rules():
    assert status.gap_status(0.0199) == status.OK and status.gap_status(0.02) == status.WATCH and status.gap_status(0.05) == status.ALERT
    assert status.twin_status(0.0, 0) == status.OK and status.twin_status(1e-9, 3) == status.WATCH and status.twin_status(0.002, 5) == status.ALERT
    assert status.worst([status.OK, status.WATCH, status.SKIPPED]) == status.WATCH
    assert status.worst([status.OK, status.ALERT, status.WATCH]) == status.ALERT
    assert status.worst([status.SKIPPED]) == status.WATCH, "nothing judged is not 'OK'"


def test_psi_helper_behaviour_on_synthetic_data():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"a": rng.normal(0, 1, 3000), "b": rng.normal(0, 1, 3000)})
    same = pd.DataFrame({"a": rng.normal(0, 1, 1000), "b": rng.normal(0, 1, 1000)})
    shifted = pd.DataFrame({"a": rng.normal(0.3, 1, 1000), "b": rng.normal(2.0, 1, 1000)})
    t_same, t_shift = drift.feature_drift(ref, same, ["a", "b"]), drift.feature_drift(ref, shifted, ["a", "b"])
    assert (t_same.psi < 0.10).all() and (t_same.status == status.OK).all()
    assert status.WATCH in set(t_shift.status) or status.ALERT in set(t_shift.status)
    assert t_shift.set_index("feature").loc["b", "status"] == status.ALERT
    assert t_shift.set_index("feature").loc["b", "psi"] > t_shift.set_index("feature").loc["a", "psi"]


def test_feature_selection_uses_importance_and_numeric_shared_columns():
    ref, cur = pd.read_csv(csv_dir() / drift.REFERENCE_FILE), pd.read_csv(csv_dir() / drift.CURRENT_FILE)
    imp = pd.read_csv(PHASE3 / "reports" / "task13" / "permutation_importance.csv")
    f = drift.select_features(imp, ref, cur)
    assert len(f) == 8 and len(set(f)) == 8
    assert all(pd.api.types.is_numeric_dtype(ref[c]) and c in cur.columns for c in f)
    ranked = list(imp.sort_values("auc_drop", ascending=False).feature)
    assert [ranked.index(c) for c in f] == sorted(ranked.index(c) for c in f), "must follow importance order"


def test_drift_on_real_data_finds_the_collapsed_raise_feature():
    d = drift.run_drift(csv_dir(), PHASE3 / "reports" / "task13" / "permutation_importance.csv")
    t = d["table"].set_index("feature")
    assert len(t) == 8 and t.loc["months_since_last_raise", "status"] == status.ALERT and t.loc["months_since_last_raise", "psi"] > 1
    assert t.loc["months_since_last_raise", "current_top_value_share_pct"] > 50 and d["status"] == status.ALERT


def test_twin_resume_check_runs_and_reports_limit():
    r = fairness.twin_resume_check(load_settings())
    assert r["summary"]["comparisons"] == 75 and r["status"] in (status.OK, status.WATCH, status.ALERT) and "by construction" in r["limit"]


def test_pay_gap_reproduces_phase3_saved_numbers():
    r = fairness.pay_gap_check(load_settings())
    g = r["groups"]["career_gap"]
    saved = pd.read_csv(PHASE3 / "reports" / "task15" / "gap_bin_gaps.csv").set_index("measure").group_gap
    assert abs(g["gap_history"] - saved["actual offer / band mid-point (history)"]) < 1e-3
    assert abs(g["gap_model"] - saved["model B predicted ratio (chosen)"]) < 1e-3
    assert abs(g["gap_model_with_career_gap"] - saved["model A predicted ratio"]) < 1e-3
    assert len(g["groups_compared"]) >= 2


def test_gender_gap_is_reported_as_not_checked_without_the_vault():
    s = replace(load_settings(), vault_dir=Path(tempfile.mkdtemp()))
    r = fairness.pay_gap_check(s)
    assert r["groups"]["gender"]["status"] == status.SKIPPED and "NOT checked" in r["groups"]["gender"]["reason"]


def test_single_comparable_group_is_never_reported_ok():
    old = status.MIN_GROUP_SIZE
    try:
        status.MIN_GROUP_SIZE = 10 ** 6
        r = fairness.pay_gap_check(load_settings())
    finally:
        status.MIN_GROUP_SIZE = old
    assert r["groups"]["career_gap"]["status"] == status.WATCH and "cannot be judged" in r["groups"]["career_gap"]["reason"]


def test_monitor_report_structure_and_alert_plan():
    rep = monitor.run_monitor(load_settings())
    assert rep["status"] in (status.OK, status.WATCH, status.ALERT) and set(rep["components"]) == {"drift", "twin_resume_gap", "pay_gap"}
    assert rep["thresholds"]["psi_watch"] == 0.10 and rep["thresholds"]["psi_alert"] == 0.25
    assert rep["coverage"] in ("complete", "partial") and isinstance(rep["explanation"], list) and rep["explanation"]
    if rep["status"] == status.ALERT:
        assert len(rep["next_steps"]) == 4
    assert monitor.EXIT_CODE == {"OK": 0, "WATCH": 1, "ALERT": 2}
    assert rep["status"] in monitor.render_text(rep)


def test_monitor_failed_check_is_visible_not_silent():
    import cortexhr_phase4.monitoring.fairness as fm
    orig = fm.twin_resume_check
    fm.twin_resume_check = lambda s: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        rep = monitor.run_monitor(load_settings())
    finally:
        fm.twin_resume_check = orig
    assert "twin_resume_gap" in rep["errors"] and rep["coverage"] == "partial" and rep["components"]["twin_resume_gap"]["status"] == status.WATCH


def test_monitor_persist_appends_audit_history():
    tmp = Path(tempfile.mkdtemp())
    s = replace(load_settings(), output_dir=tmp)
    rep = monitor.run_monitor(s)
    monitor.persist(rep, s)
    monitor.persist(rep, s)
    hist = monitor.read_history(s)
    assert len(hist) == 2 and hist[0]["status"] == rep["status"] and "created_utc" in hist[0] and hist[0]["max_psi"] > 0
    assert json.loads((tmp / "monitoring" / "monitoring_latest.json").read_text())["status"] == rep["status"]
    assert (tmp / "monitoring" / "monitoring_report.md").exists()


def test_monitor_is_reproducible():
    a, b = monitor.run_monitor(load_settings()), monitor.run_monitor(load_settings())
    ta, tb = a["components"]["drift"]["table"], b["components"]["drift"]["table"]
    assert ta == tb and a["status"] == b["status"]
