"""CortexHR Insights Console (Streamlit) - every feature talks to the live /v1 API.

Run from the phase4 folder (so .streamlit/config.toml is picked up)::

    cd phase4
    CORTEXHR_API_URL=http://127.0.0.1:8000 streamlit run src/cortexhr_phase4/console/app.py
"""
from __future__ import annotations

import sys
import json
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))          # phase4/src

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from cortexhr_phase4.console import ai_render as ai, charts, dashboard
from cortexhr_phase4.console.api_client import ApiClient, ApiResult
from cortexhr_phase4.core.settings import PHASE1, PHASE2, PHASE3, PHASE4_ROOT, load_settings

AI_LABEL = ai.AI_LABEL
INSIGHT_TABS = ["Risks", "Anomalies", "Assistant", "Resume Screener", "Forecast"]       # the five feature tabs of the official Day 19 task
SETTINGS = load_settings()
PAGES = {k: label for k, label, _ in dashboard.NAV}


# ------------------------------------------------------------------ plumbing
def client() -> ApiClient:
    api = ApiClient(SETTINGS.api_url, SETTINGS.api_key, SETTINGS.request_timeout_s)
    api.bearer_token = st.session_state.get("access_token")
    return api


@st.cache_data(ttl=20, show_spinner=False)
def cached_get(path: str, **params) -> ApiResult:
    return client().get(path, **params)


def fail(res: ApiResult) -> None:
    """Friendly message instead of a crash."""
    st.error(res.message(SETTINGS.api_url))


def pill(status: str) -> str:
    return f'<span class="cx-pill {status}">{status}</span>'


def to_df(records: list[dict], cols: list[str] | None = None) -> pd.DataFrame:
    df = pd.DataFrame(records)
    return df[[c for c in cols if c in df.columns]] if cols else df


# ------------------------------------------------------------------ pages
def page_dashboard() -> None:
    ov, emp = cached_get("/v1/overview"), cached_get("/v1/attrition/employees", limit=500)
    if not (ov.ok and emp.ok):
        fail(ov if not ov.ok else emp)
        return
    user = {"name": st.session_state.get("user_name") or "Guest", "role": st.session_state.get("user_role", "Admin")}
    ai.html_frame(dashboard.build(ov.data, emp.data["employees"], user, mode="embed", env=SETTINGS.env, show_welcome=st.session_state.get("show_welcome", SETTINGS.show_welcome)),
                    height=790, scrolling=False, label_inside=True)


def tab_attrition() -> None:
    st.subheader("Attrition Risk")
    c1, c2, c3 = st.columns([1, 1, 1])
    band = c1.selectbox("Risk band", ["All", "High", "Medium", "Low"], key="att_band")
    dep = c2.selectbox("Department", ["All"] + [d["id"] for d in (cached_get("/v1/overview").data.get("departments", []))], key="att_dep")
    n = c3.slider("Rows", 5, 100, 25, key="att_n")
    res = client().get("/v1/attrition/employees", band=None if band == "All" else band, department_id=None if dep == "All" else dep, limit=n)
    if not res.ok:
        return fail(res)
    st.caption(f"{res.data['count']} employees match, out of {res.data['population']} active. Mean 12-month risk {res.data['mean_risk'] * 100:.1f}%. Bands: {res.data['bands']}")
    if not res.data["employees"]:
        return st.info("No employees match these filters. Set Risk band and Department back to All to see everyone.")
    ai.dataframe(to_df(res.data["employees"], ["employee_id", "department_id", "risk_score_12m", "risk_band", "top_reason_1"]), hide_index=True)
    ids = [e["employee_id"] for e in res.data["employees"]]
    if ids:
        pick = st.selectbox("Explain one employee", ids, key="att_pick")
        one = client().post("/v1/attrition/predict", {"employee_id": pick})
        if not one.ok:
            return fail(one)
        d = one.data
        ai.card(f'<div class="cx-card"><h4>{d["employee_id"]} &middot; {d["department_id"]}</h4>Risk score <b>{d["risk_score_12m"] * 100:.1f}%</b> {pill({"High": "ALERT", "Medium": "WATCH", "Low": "OK"}[d["risk_band"]])} '
                    f'&nbsp;<small>source: {d["source"]}, model {d["model_version"]}</small><ul>' + "".join(f"<li>{r}</li>" for r in d["top_reasons"]) + f'</ul><span class="cx-ai">{AI_LABEL}</span></div>', unsafe_allow_html=True)
        st.caption(d["notice"])


def tab_payroll() -> None:
    st.subheader("Payroll Anomalies")
    st.caption("Score one payslip. The form starts with a real payslip from the dataset; change any number to see how the flag reacts.")
    base = {"payroll_item_id": "PI-000001", "run_id": "PR-2025-04", "employee_id": "EMP-0002", "month": "2025-04", "department_id": "D08", "role_id": "R047", "job_level": 4,
                    "overtime_eligible": "N", "monthly_fixed_gross": 144460.0, "payable_days": 30.0, "lop_days": 0.0, "overtime_hours_paid": 0.0, "overtime_pay": 0.0, "variable_pay": 34670.4,
                    "variable_pay_type": "Performance Bonus", "gross_pay": 179130.4, "pf_employee": 1800.0, "esi_employee": 0.0, "professional_tax": 200.0, "tds": 18625.79,
                    "other_deductions": 0.0, "total_deductions": 20625.79, "net_pay": 158504.61}
    with st.form("payroll_form"):
        cols = st.columns(3)
        row = dict(base)
        for i, k in enumerate(["monthly_fixed_gross", "payable_days", "lop_days", "overtime_hours_paid", "overtime_pay", "variable_pay", "gross_pay", "pf_employee", "esi_employee",
                               "professional_tax", "tds", "other_deductions", "total_deductions", "net_pay"]):
            row[k] = cols[i % 3].number_input(k.replace("_", " "), value=float(base[k]), key=f"pay_{k}")
        go = st.form_submit_button("Check payslip")
    if go:
        res = client().post("/v1/anomaly/payroll", row)
        if not res.ok:
            return fail(res)
        r = res.data["results"][0]
        ai.markdown(f'Result: {pill("ALERT" if r.get("flagged") == "Y" else "OK")} anomaly score <b>{r.get("anomaly_score")}</b>', unsafe_allow_html=True)
        ai.dataframe(pd.DataFrame([r]).T.rename(columns={0: "value"}))
        st.caption(res.data["notice"])


def tab_assistant() -> None:
    st.subheader("HR Assistant")
    st.caption("Ask about leave, payslips, attendance or policies, in your own words. Questions that need personal records use a demo user from the Phase 2 mock HR service.")
    st.session_state.setdefault("chat_session", uuid.uuid4().hex[:12])
    st.session_state.setdefault("chat_log", [])
    if not st.session_state["chat_log"]:
        st.info("Not sure what to ask? Try: *What is the notice period?* or *How many leave days do I have?*")
    for who, text in st.session_state["chat_log"]:
        if who == "error":
            st.error(text)                    # a problem message, not an AI answer
        else:
            ai.chat_message(who, text)        # every assistant bubble carries the label
    msg = st.chat_input("Ask the HR assistant, for example: What is the notice period?")
    if msg:
        st.session_state["chat_log"].append(("user", msg))
        res = client().post("/v1/chat", {"message": msg, "session_id": st.session_state["chat_session"]})
        if res.ok:
            reply = res.data.get("reply") or "I could not find an answer to that. Please rephrase, or ask HR directly."
            if res.data.get("confidence") is not None:
                reply += f"\n\n_intent: {res.data.get('intent')}, confidence {float(res.data['confidence']):.2f}_"
            st.session_state["chat_log"].append(("assistant", reply))
        else:
            st.session_state["chat_log"].append(("error", res.message(SETTINGS.api_url)))
        st.rerun()


def tab_payband() -> None:
    st.subheader("Pay-Band Advisor")
    opts = cached_get("/v1/payband/options")
    if not opts.ok:
        return fail(opts)
    cat, ex = opts.data["categories"], opts.data["example"]
    with st.form("band_form"):
        c = st.columns(3)
        body = {"department_id": c[0].selectbox("Department", cat["department_id"], index=cat["department_id"].index(ex["department_id"])),
                "role_id": c[1].selectbox("Role", cat["role_id"], index=cat["role_id"].index(ex["role_id"])),
                "job_level": int(c[2].number_input("Job level", 1, 10, int(ex["job_level"]))),
                "band_mid_2026": float(c[0].number_input("Role band mid-point (INR / year)", 1.0, 1e8, float(ex["band_mid_2026"]))),
                "market_index": float(c[1].number_input("Market index", 0.01, 10.0, float(ex["market_index"]), format="%.4f")),
                "prior_experience_years": float(c[2].number_input("Prior experience (years)", 0.0, 50.0, float(ex["prior_experience_years"]))),
                "skills_count": int(c[0].number_input("Skills count", 0, 50, int(ex["skills_count"]))),
                "work_location": c[1].selectbox("Work location", cat["work_location"], index=cat["work_location"].index(ex["work_location"])),
                "highest_education": c[2].selectbox("Highest education", cat["highest_education"], index=cat["highest_education"].index(ex["highest_education"]))}
        go = st.form_submit_button("Recommend a band")
    st.caption("Job-related inputs only. Gender, age, career gap, hire date and source are never used.")
    if go:
        res = client().post("/v1/payband/recommend", body)
        if not res.ok:
            return fail(res)
        d = res.data
        ai.metrics([("Low", f"\u20b9{d['recommended_low']:,.0f}"), ("Mid", f"\u20b9{d['recommended_mid']:,.0f}"), ("High", f"\u20b9{d['recommended_high']:,.0f}")])
        st.caption(d["notice"])


def tab_forecast() -> None:
    st.subheader("Six-Month Forecast")
    st.caption("How many people we are likely to have, and how many will leave, over the next six months - shown as a best guess with a range, and tested on past months.")
    ov = cached_get("/v1/overview")
    depts = {"ALL": "Company total"} | {d["id"]: d["name"] for d in ov.data.get("departments", [])} if ov.ok else {"ALL": "Company total"}
    c = st.columns(2)
    metric = c[0].selectbox("Metric", ["headcount_end", "exits", "attrition_rate_monthly_pct"], format_func=lambda m: {"headcount_end": "Headcount (end of month)", "exits": "Exits per month", "attrition_rate_monthly_pct": "Monthly attrition rate %"}[m])
    dept = c[1].selectbox("Department", list(depts), format_func=lambda d: f"{d} - {depts[d]}", disabled=metric != "headcount_end")
    res = client().get("/v1/forecast", metric=metric, department_id=dept if metric == "headcount_end" else "ALL")
    if not res.ok:
        return fail(res)
    d = res.data
    if not d.get("forecast"):
        return st.info("There is no forecast for this choice yet. Pick another metric or department.")
    if d.get("improved_model_status") and d["improved_model_status"] != "run":
        st.warning(f"Improved model status: {d['improved_model_status']}. The numbers below come from the seasonal baseline.")
    ai.html_frame(charts.forecast_chart_html(f"{depts.get(d['department_id'], d['department_id'])}: {metric.replace('_', ' ')}", d["history"], d["forecast"], d["events"],
                                               "%" if "pct" in metric else "", f"Method: {d['method']}. Trained through {d['trained_through']}."), height=440)
    st.markdown("**Monthly forecast**")
    ai.dataframe(to_df(d["forecast"]).rename(columns={"forecast_month": "month", "forecast_value": "forecast", "lower_80": "low (80%)", "upper_80": "high (80%)"}), hide_index=True)
    st.markdown("**How accurate was it on the past?**")
    acc = to_df(d["accuracy"], ["model", "scope", "n", "mae", "rmse", "mape_pct", "bias", "coverage_80"])
    ai.dataframe(acc, hide_index=True)
    st.caption("holdout_last_6_months: the last 6 known months were hidden and forecast. rolling_origin: the same test repeated from many past starting points. Coverage should be near 0.80.")
    ex = d.get("explanation")
    if ex:
        lt = ex["least_trusted_month"]
        ai.card(f'<div class="cx-card"><h4>What the forecast says</h4>Headcount moves from <b>{ex["current_headcount"]:.0f}</b> to <b>{ex["headcount_in_6_months"]:.0f}</b> ({ex["growth_pct_6m"]:+.1f}%) with about <b>{ex["expected_exits_6m"]:.0f}</b> exits in six months.'
                    f'<br>Least trusted month: <b>{lt["month"][:7]}</b>, because {lt["why"]}<br><small>{ex["events_handling"]}</small><ul>' + "".join(f"<li>{x}</li>" for x in ex["limits"]) + "</ul></div>", unsafe_allow_html=True)
    with st.expander("What-if: add a workforce event (needs the improved model)"):
        with st.form("scenario"):
            s = st.columns(4)
            eff = s[0].selectbox("Event", ["hiring_freeze", "hiring_boost", "policy_change"])
            a, b = s[1].text_input("Start (YYYY-MM)", "2026-11"), s[2].text_input("End (YYYY-MM)", "2027-01")
            sd = s[3].selectbox("Department", list(depts), format_func=lambda x: depts[x])
            go = st.form_submit_button("Run what-if")
        if go:
            r = client().post("/v1/forecast/scenario", {"scenarios": [{"effect": eff, "start": a, "end": b, "department_id": sd}]})
            if not r.ok:
                fail(r)
            else:
                fdf = to_df(r.data["forecast"])
                ai.dataframe(fdf[(fdf.metric == "headcount_end") & (fdf.department_id == "ALL")], hide_index=True)


def tab_attendance() -> None:
    st.subheader("Attendance Anomalies")
    st.caption("Enter one month of attendance. The form starts with a real month from the dataset; change any number to see how the flag reacts.")
    with st.form("attendance_form"):
        c = st.columns(3)
        row = {"employee_id": "EMP-0002", "month": "2025-04",
               "scheduled_days": int(c[0].number_input("Scheduled days", 0, 31, 20, key="att_sched")),
               "present_days": int(c[1].number_input("Present days", 0, 31, 20, key="att_present")),
               "wfh_days": int(c[2].number_input("Work-from-home days", 0, 31, 18, key="att_wfh")),
               "late_arrivals": int(c[0].number_input("Late arrivals", 0, 31, 1, key="att_late")),
               "avg_hours_worked": float(c[1].number_input("Average hours per day", 0.0, 24.0, 8.5, key="att_hours")),
               "overtime_hours": float(c[2].number_input("Overtime hours", 0.0, 200.0, 0.0, key="att_ot"))}
        go = st.form_submit_button("Check attendance")
    if go:
        if row["present_days"] > row["scheduled_days"]:
            return st.warning("Present days cannot be more than scheduled days. Please correct the numbers and try again.")
        if row["wfh_days"] > row["present_days"]:
            return st.warning("Work-from-home days cannot be more than present days. Please correct the numbers and try again.")
        res = client().post("/v1/anomaly/attendance", row)
        if not res.ok:
            return fail(res)
        if not res.data.get("results"):
            return st.info("The detector returned no result for this month. Please try different numbers.")
        r = res.data["results"][0]
        ai.markdown(f'Result: {pill("ALERT" if r.get("flagged") == "Y" else "OK")} anomaly score <b>{r.get("anomaly_score")}</b>', unsafe_allow_html=True)
        ai.dataframe(pd.DataFrame([r]).T.rename(columns={0: "value"}))
        st.caption(res.data["notice"])


def tab_anomalies() -> None:
    st.subheader("Anomalies")
    st.caption("Check a payslip or a month of attendance for numbers that look unusual. A flag means 'please take a look', never 'something is wrong'.")
    ai.badge()                                                       # visible before any result exists: the tab's output is always AI-generated
    st.session_state.setdefault("anomaly_kind", "Payroll")
    st.segmented_control("What would you like to check?", ["Payroll", "Attendance"], key="anomaly_kind")
    (tab_attendance if st.session_state.get("anomaly_kind") == "Attendance" else tab_payroll)()


def page_insights() -> None:
    st.title("Insights Console")
    role = st.session_state.get("user_role", "Admin").lower().replace(" ", "_")
    tabs_by_role = {"admin": INSIGHT_TABS, "hr_manager": INSIGHT_TABS,
                    "analyst": ["Risks", "Anomalies", "Assistant", "Forecast"],
                    "recruiter": ["Assistant", "Resume Screener"]}
    available = tabs_by_role.get(role, INSIGHT_TABS)
    st.info("Every result is AI-generated decision support for human review. CortexHR does not make hiring, termination, payroll or attendance decisions.")
    if st.session_state.get("insights_tab") not in available:
        st.session_state["insights_tab"] = available[0]
    st.segmented_control("Feature", available, key="insights_tab", label_visibility="collapsed")
    TAB_FUNCS[st.session_state.get("insights_tab") or available[0]]()




def page_data_foundation() -> None:
    """Expose the Phase 1â€“3 data audit and evidence in the integrated console."""
    st.title("Data Foundation")
    st.caption("Phase 1â€“3 data lineage, quality checks, leakage evidence and saved evaluation reports. This page reads the supplied artifacts; it does not silently rerun the original notebooks.")
    roots = [("Phase 1", PHASE1 / "03_Data"), ("Phase 2", PHASE2 / "03_Data"), ("Phase 3", PHASE3 / "03_Data")]
    inventory = []
    for phase, root in roots:
        if not root.exists():
            inventory.append({"phase": phase, "dataset": str(root), "rows": None, "size_kb": None, "status": "missing"})
            continue
        for path in sorted(root.rglob("*.csv")):
            try:
                frame = pd.read_csv(path)
                inventory.append({"phase": phase, "dataset": str(path.relative_to(root)), "rows": int(len(frame)), "columns": int(len(frame.columns)),
                                  "size_kb": round(path.stat().st_size / 1024, 1), "status": "readable"})
            except Exception as exc:
                inventory.append({"phase": phase, "dataset": str(path.relative_to(root)), "rows": None, "columns": None,
                                  "size_kb": round(path.stat().st_size / 1024, 1), "status": f"read error: {type(exc).__name__}"})
        for path in sorted(root.rglob("*.xlsx")):
            try:
                with pd.ExcelFile(path) as workbook:
                    sheet_count = len(workbook.sheet_names)
                inventory.append({"phase": phase, "dataset": str(path.relative_to(root)), "rows": None, "columns": None,
                                  "size_kb": round(path.stat().st_size / 1024, 1), "status": f"workbook Â· {sheet_count} sheets"})
            except Exception as exc:
                inventory.append({"phase": phase, "dataset": str(path.relative_to(root)), "rows": None, "columns": None,
                                  "size_kb": round(path.stat().st_size / 1024, 1), "status": f"workbook read error: {type(exc).__name__}"})
    tabs = st.tabs(["Dataset Inventory", "Audit & Leakage", "Model Evidence", "Documentation"])
    with tabs[0]:
        if inventory:
            frame = pd.DataFrame(inventory)
            ai.system_metrics([("CSV datasets discovered", str(len(frame)))])
            ai.dataframe(frame, hide_index=True)
        else:
            st.warning("No CSV datasets were found in the configured Phase 1â€“3 folders.")
        st.caption("Counts are read from the files currently present on disk. Restricted validation-vault data is intentionally not listed or exposed here.")
    with tabs[1]:
        evidence = [("Data audit â€” 17 findings", PHASE1 / "reports" / "tables" / "audit_table.csv"),
                    ("Leakage review", PHASE1 / "reports" / "tables" / "leakage_review.csv"),
                    ("Cleaning before / after", PHASE1 / "reports" / "tables" / "cleaning_report.csv"),
                    ("Schema inventory", PHASE1 / "reports" / "tables" / "schema_inventory.csv")]
        for label, path in evidence:
            with st.expander(label, expanded=("Leakage" in label)):
                if path.exists():
                    try: ai.dataframe(pd.read_csv(path), hide_index=True)
                    except Exception as exc: st.error(f"Could not read {path.name}: {type(exc).__name__}")
                else: st.info(f"Saved evidence not present: {path.relative_to(PHASE1)}")
        st.info("Leakage findings are historical audit evidence. The original source columns are retained for traceability; model scoring must use the leakage-safe feature pipeline.")
    with tabs[2]:
        for label, path in [("Phase 1 held-out evaluation metrics", PHASE1 / "reports" / "tables" / "evaluation_metrics.csv"),
                            ("Phase 1 baseline comparison", PHASE1 / "reports" / "tables" / "baseline_comparison.csv"),
                            ("Phase 2 payroll model comparison", PHASE2 / "reports" / "baseline_model_comparison.md")]:
            st.markdown(f"**{label}**")
            if path.suffix.lower() == ".csv" and path.exists(): ai.dataframe(pd.read_csv(path), hide_index=True)
            elif path.exists(): st.markdown(path.read_text(encoding="utf-8"))
            else: st.caption(f"Saved evidence not present: {path}")
        st.warning("Evaluation figures are saved historical evidence, not a fresh training run. Review the phase model cards and limitations before operational use.")
    with tabs[3]:
        for label, path in [("Phase 1 data audit report", PHASE1 / "reports" / "data_audit_report.md"),
                            ("Phase 1 manager summary", PHASE1 / "reports" / "manager_summary.md"),
                            ("Phase 2 model evaluation", PHASE2 / "reports" / "model_evaluation_report.md"),
                            ("Phase 3 methodology and limitations", PHASE3 / "docs" / "limitations.md")]:
            with st.expander(label):
                if path.exists(): st.markdown(path.read_text(encoding="utf-8"))
                else: st.info(f"Document not present: {path}")


def page_fairness() -> None:
    st.title("Fairness & Audit")
    if st.button("Run monitoring now", type="primary"):
        res = client().get("/v1/monitoring", refresh="true")
        cached_get.clear()
    else:
        res = cached_get("/v1/monitoring")
    if not res.ok:
        return fail(res)
    d = res.data
    rep = d["report"]
    ai.markdown(f'Overall status {pill(d["status"])} &nbsp; checked {d["created_utc"]} &nbsp; coverage <b>{d["coverage"]}</b>', unsafe_allow_html=True)
    if d["not_checked"]:
        st.warning("Not checked: " + ", ".join(d["not_checked"]) + ". The restricted file is not available in this environment, so that comparison was skipped, not passed.")
    drift = rep["components"].get("drift", {})
    if drift.get("table"):
        ai.html_frame(charts.drift_chart_html(drift["table"], d["history"]), height=min(560, 60 + 34 * len(drift["table"]) + 80))
        ai.dataframe(to_df(drift["table"], ["feature", "psi", "status", "reference_mean", "current_mean", "shift_pct_of_reference_sd", "current_top_value_share_pct"]), hide_index=True)
    for line in d["explanation"]:
        ai.markdown(f"- {line}")
    f1, f2 = st.columns(2)
    tw = rep["components"].get("twin_resume_gap", {})
    if tw.get("summary"):
        ai.card(container=f1, html=f'<div class="cx-card"><h4>Twin-resume gap {pill(tw["status"])}</h4>{tw["summary"]["comparisons"]} comparisons, largest gap <b>{tw["summary"]["max_abs_gap"]:.6f}</b> (tolerance {tw["tolerance"]}).<br><small>{tw["limit"]}</small></div>', unsafe_allow_html=True)
    pg = rep["components"].get("pay_gap", {})
    for k, v in pg.get("groups", {}).items():
        body = (f'model gap <b>{v["gap_model"]:.4f}</b> vs history <b>{v["gap_history"]:.4f}</b>; the same model with career gap as an input: {v["gap_model_with_career_gap"]:.4f}.' if "gap_model" in v else v.get("reason", ""))
        ai.card(container=f2, html=f'<div class="cx-card"><h4>Pay gap by {k.replace("_", " ")} {pill(v["status"])}</h4>{body}</div>', unsafe_allow_html=True)
    if pg.get("limit"):
        st.caption(pg["limit"])
    if rep.get("next_steps"):
        ai.markdown("**ALERT: next steps**\n\n" + "\n\n".join(rep["next_steps"]))
    st.markdown("**Status history**")
    ai.dataframe(to_df(d["history"]).iloc[::-1], hide_index=True)
    st.caption(d["notice"])


def page_models() -> None:
    st.title("Models & Explainability")
    m, p = cached_get("/v1/models"), cached_get("/v1/performance")
    if not m.ok:
        return fail(m)
    if not p.ok:
        return fail(p)
    d = p.data
    availability, performance, explainability, cards = st.tabs(["Model Registry", "Evaluation", "Attrition Explainability", "Model Cards"])
    with availability:
        st.markdown("**Runtime model availability and provenance**")
        ai.system_dataframe(to_df(list(m.data["models"].values()), ["name", "status", "version", "source", "detail"]), hide_index=True)
        st.caption("A fallback is not the same as a live model load. Unavailable models are not treated as healthy.")
    with performance:
        st.markdown("**Attrition v2 (12-month risk)**")
        ai.dataframe(to_df(d["attrition"]["cv"]), hide_index=True)
        ai.info(d["attrition"]["verdict"])
        st.markdown("**Resume ranker (555 human labels, 10 jobs)**")
        ai.dataframe(to_df(d["resume"]["ranking"]), hide_index=True)
        st.markdown("**Pay-band model (test set)**")
        ai.dataframe(to_df(d["payband"]["test_metrics"]), hide_index=True)
        if d.get("forecast"):
            st.markdown("**Headcount forecast (company total, backtests)**")
            ai.dataframe(to_df(d["forecast"]["comparison"], ["metric", "model", "method", "scope", "n", "mae", "mape_pct", "coverage_80"]), hide_index=True)
        st.caption(d["notice"])
    with explainability:
        shap_path = PHASE3 / "reports" / "task14" / "shap_full430_feature_values.csv"
        summary_path = PHASE3 / "reports" / "task14" / "shap_full430_summary.json"
        if not shap_path.exists():
            st.info("Saved per-employee SHAP explanations are not present in this deployment.")
        else:
            shap_df = pd.read_csv(shap_path)
            feature_cols = [c for c in shap_df.columns if c not in {"employee_id", "base_value", "prediction"}]
            employee_ids = shap_df.employee_id.astype(str).tolist()
            selected = st.selectbox("Employee ID", employee_ids, key="shap_employee")
            row = shap_df.loc[shap_df.employee_id.astype(str) == selected].iloc[0]
            contributions = pd.DataFrame({"feature": feature_cols, "contribution": [float(row[c]) for c in feature_cols]})
            contributions["direction"] = contributions.contribution.map(lambda x: "Raises model score" if x > 0 else "Lowers model score")
            contributions = contributions.reindex(contributions.contribution.abs().sort_values(ascending=False).index).head(12)
            ai.metrics([("Model output", f"{float(row.prediction) * 100:.2f}%"), ("Baseline output", f"{float(row.base_value) * 100:.2f}%")])
            st.markdown("**Largest saved feature contributions**")
            ai.dataframe(contributions, hide_index=True)
            st.caption("Positive contributions push the model output upward; negative contributions push it downward. These are saved Phase 3 SHAP explanations, not causal effects or proof of why a person will leave. Use them as review context only.")
            if summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                st.caption(f"SHAP artifact: {summary.get('rows')} employees Â· {summary.get('shap_version')} Â· max additivity error {summary.get('additivity_max_error'):.2e} Â· generated {summary.get('created_utc')}")
    with cards:
        card_dir = PHASE4_ROOT / "docs" / "model_cards"
        for name in ("attrition_v2.md", "payroll_anomaly.md", "attendance_anomaly.md", "chatbot_intent.md", "resume_ranker_v2.md", "salary_band_v1.md", "headcount_forecast.md"):
            path = card_dir / name
            with st.expander(name.replace(".md", "").replace("_", " ").title()):
                if path.exists(): st.markdown(path.read_text(encoding="utf-8"))
                else: st.info(f"Model card not found: {path.name}")


def tab_resume() -> None:
    st.subheader("Resume Screener")
    st.caption("Pick a job, then read the candidates in the order the screener suggests. You make every decision.")
    jobs = cached_get("/v1/resume/jobs")
    if not jobs.ok:
        return fail(jobs)
    jl = {j["job_id"]: j for j in jobs.data["jobs"]}
    c = st.columns([2, 1])
    jid = c[0].selectbox("Job", list(jl), format_func=lambda j: f"{j} - {jl[j]['job_title']} ({jl[j]['department_id']})")
    k = c[1].slider("Top candidates", 1, 20, 5)
    j = jl[jid]
    st.caption(f"Required: {j['required_skills']}. Minimum experience {j['min_experience_years']} years. Openings {j['openings']}.")
    res = client().post("/v1/resume/rank", {"job_id": jid, "top_k": k})
    if not res.ok:
        return fail(res)
    cand = res.data.get("candidates", [])
    if not cand:
        return st.info("No candidates were found for this job yet. Choose another job.")
    ai.dataframe(to_df(cand), hide_index=True)
    st.info("The screener never sees name, gender, age, college tier, e-mail or city. Use it to order resumes for a recruiter, never to reject anyone automatically.")
    st.caption(res.data.get("notice", AI_LABEL))
    with st.expander("Upload a new PDF / DOCX resume"):
        st.caption("Maximum 8 MB and 20 PDF pages. Text is extracted in memory, protected identifiers are masked before scoring, and the upload is not saved to disk. Uploaded candidates are cleared when the API restarts.")
        uploaded = st.file_uploader("Choose a resume", type=["pdf", "docx"], key="resume_upload_file")
        if uploaded is not None and st.button("Extract and rank uploaded resume", type="primary", key="resume_upload_submit"):
            mime = "application/pdf" if uploaded.name.lower().endswith(".pdf") else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            with st.spinner("Extracting text and evaluating against the selected jobâ€¦"):
                result = client().upload("/v1/resume/upload", uploaded.name, uploaded.getvalue(), mime, {"job_id": jid, "top_k": str(k)})
            if not result.ok:
                fail(result)
            else:
                st.success("Resume processed. Candidate identifier: " + result.data.get("uploaded_resume_id", "n/a"))
                ai.dataframe(pd.DataFrame(result.data.get("candidates", [])), hide_index=True)
                st.caption(result.data.get("score_comparability", ""))
                st.caption(result.data.get("retention", ""))
                st.warning(result.data.get("notice", "Human review is mandatory."))
    with st.expander("Next step (optional): suggest a pay band for an offer"):
        tab_payband()


def page_api() -> None:
    st.title("API & Deploy")
    import time
    t0 = time.perf_counter()
    h = client().get("/v1/health")
    ms = (time.perf_counter() - t0) * 1000
    if not h.ok:
        return fail(h)
    d = h.data
    c = st.columns(4)
    c[0].metric("Status", d["status"].upper())
    c[1].metric("Version", d["version"])
    c[2].metric("Uptime", f"{d['uptime_s'] / 60:.1f} min")
    c[3].metric("Round trip", f"{ms:.0f} ms")
    ai.system_dataframe(pd.DataFrame([{"model": k, "state": v} for k, v in d["models"].items()]), hide_index=True)
    st.markdown(f"API address: `{SETTINGS.api_url}`  &middot; interactive docs: `{SETTINGS.api_url}/docs`")
    st.markdown("**Endpoints (all under /v1)**")
    ai.system_dataframe(pd.DataFrame(ENDPOINTS, columns=["method", "path", "purpose"]), hide_index=True)


ENDPOINTS = [("GET", "/v1/health", "service and model status"), ("POST", "/v1/auth/login", "password sign-in; returns a short-lived bearer token"), ("POST", "/v1/auth/logout", "revoke the current bearer token"),
             ("GET", "/v1/models", "every model: status, version, source"), ("GET", "/v1/overview", "dashboard numbers"),
             ("POST", "/v1/attrition/predict", "12-month risk + reasons for one employee"), ("GET", "/v1/attrition/employees", "risk list with filters"),
             ("POST", "/v1/anomaly/payroll[/batch]", "payslip anomaly flags"), ("POST", "/v1/anomaly/attendance[/batch]", "attendance anomaly flags"), ("POST", "/v1/chat", "HR assistant"),
             ("POST", "/v1/resume/rank", "rank saved resumes for a job"), ("POST", "/v1/resume/upload", "in-memory PDF/DOCX extraction and ranking"), ("GET", "/v1/resume/jobs", "job list"), ("POST", "/v1/payband/recommend", "low / mid / high pay band"),
             ("GET", "/v1/payband/options", "allowed values for the pay-band form"), ("GET", "/v1/forecast", "6-month forecast with 80% interval"), ("POST", "/v1/forecast/scenario", "what-if workforce event"),
             ("GET", "/v1/monitoring", "drift + fairness status (OK / WATCH / ALERT)"), ("GET", "/v1/performance", "evaluation tables")]


def page_settings() -> None:
    st.title("Settings")
    st.markdown(f'<div class="cx-card"><h4>Connection & security</h4>API address <code>{SETTINGS.api_url}</code><br>Environment <code>{SETTINGS.env}</code><br>API key <code>{"configured" if SETTINGS.api_key else "not configured"}</code><br>Session role <code>{st.session_state.get("user_role", "Admin")}</code><br>'
                'Secrets are read from environment variables and are never written to project files.</div>', unsafe_allow_html=True)
    st.session_state["show_welcome"] = st.toggle("Show the welcome greeting on the dashboard", value=st.session_state.get("show_welcome", SETTINGS.show_welcome))
    st.markdown(f"Signed in as **{st.session_state.get('user_name', '')}** Â· **{st.session_state.get('user_role', 'Admin')}**")
    if st.button("Sign out", key="switch_user"):
        if st.session_state.get("access_token"):
            client().post("/v1/auth/logout", {})
        for key in ("access_token", "user_name", "user_role", "username", "page", "chat_log"):
            st.session_state.pop(key, None)
        st.rerun()


TAB_FUNCS = {"Risks": tab_attrition, "Anomalies": tab_anomalies, "Assistant": tab_assistant, "Resume Screener": tab_resume, "Forecast": tab_forecast}
ROUTE_TAB = {"forecast": "Forecast", "resume": "Resume Screener"}        # menu entries that open a tab of the Insights console
ROUTES = {"dashboard": page_dashboard, "data": page_data_foundation, "insights": page_insights, "forecast": page_insights, "fairness": page_fairness, "models": page_models, "resume": page_insights,
          "api": page_api, "settings": page_settings}


def _sidebar(active: str) -> str | None:
    """Logo, menu and promo card. The menu is made of real Streamlit buttons, so choosing a page never reloads the browser tab and the session
    (chat, selected feature tab, form values) is kept. Returns the page the user just chose, if any."""
    chosen = None
    with st.sidebar:
        st.markdown(dashboard.sidebar_html(active, flow=True, part="top"), unsafe_allow_html=True)
        role = st.session_state.get("username") and st.session_state.get("user_role", "Admin").lower().replace(" ", "_") or "admin"
        role_pages = {"admin": set(ROUTES), "hr_manager": set(ROUTES) - {"api"},
                      "recruiter": {"dashboard", "insights", "resume", "settings"},
                      "analyst": {"dashboard", "data", "insights", "forecast", "fairness", "models", "api", "settings"}}
        allowed = role_pages.get(role, role_pages["admin"])
        for key, label, _ in dashboard.NAV:
            if key not in allowed:
                continue
            if st.button(label, key=f"nav_{key}", use_container_width=True, type="primary" if key == active else "secondary"):
                chosen = key
        st.markdown(dashboard.sidebar_html(active, flow=True, part="bottom"), unsafe_allow_html=True)
    return chosen


def _sign_in() -> None:
    """Password-based sign-in. Local demo mode is explicitly labelled when auth is not configured."""
    st.markdown('<div class="cx-card"><h2 style="margin-top:0">Welcome to CortexHR</h2><p>Sign in with your organization account to access workforce intelligence.</p></div>', unsafe_allow_html=True)
    if not SETTINGS.auth_secret:
        if SETTINGS.env.lower() == "production":
            st.error("Authentication is not configured. Set CORTEXHR_AUTH_SECRET and CORTEXHR_AUTH_USERS before starting the production console.")
            return
        st.warning("Local development mode: password authentication is not configured. This mode is for local evaluation only; configure CORTEXHR_AUTH_SECRET and CORTEXHR_AUTH_USERS for real accounts.")
        with st.form("demo_sign_in"):
            name = st.text_input("Your name", max_chars=40, placeholder="For example: Priya Sharma")
            if st.form_submit_button("Continue in local demo mode", type="primary"):
                clean = " ".join(name.split())
                if len(clean) < 2:
                    st.warning("Please type at least 2 characters.")
                else:
                    st.session_state.update(user_name=clean, user_role="Admin", username="local-demo")
                    st.rerun()
        return
    with st.form("sign_in", clear_on_submit=True):
        username = st.text_input("Username", max_chars=120, autocomplete="username")
        password = st.text_input("Password", type="password", max_chars=256, autocomplete="current-password")
        if st.form_submit_button("Sign in", type="primary"):
            if not username.strip() or not password:
                st.warning("Enter your username and password.")
            else:
                result = ApiClient(SETTINGS.api_url, SETTINGS.api_key, SETTINGS.request_timeout_s).post("/v1/auth/login", {"username": username.strip(), "password": password})
                if not result.ok:
                    st.error(result.message(SETTINGS.api_url))
                else:
                    user = result.data["user"]
                    role_labels = {"admin": "Admin", "hr_manager": "HR Manager", "recruiter": "Recruiter", "analyst": "Analyst"}
                    st.session_state.update(access_token=result.data["access_token"], username=user["username"], user_name=user["display_name"], user_role=role_labels.get(user["role"], "User"))
                    st.rerun()


def main() -> None:
    st.set_page_config(page_title="CortexHR Insights Console", page_icon=str(dashboard.ASSETS / "cortexhr_logo_mark.png"), layout="wide", initial_sidebar_state="expanded")
    st.markdown(f"<style>{(Path(__file__).parent / 'static' / 'theme.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    if not st.session_state.get("user_name") or (SETTINGS.auth_secret and not st.session_state.get("access_token")):
        _sign_in()
        return
    if "page" not in st.session_state:                                  # first visit: honour a ?page= deep link
        wanted = st.query_params.get("page", "dashboard")
        st.session_state["page"] = wanted if wanted in ROUTES else "dashboard"
    chosen = _sidebar(st.session_state["page"])
    if chosen and chosen != st.session_state["page"]:
        st.session_state["page"] = chosen
        if chosen in ROUTE_TAB:
            st.session_state["insights_tab"] = ROUTE_TAB[chosen]
        elif chosen == "insights":
            st.session_state.setdefault("insights_tab", "Risks")
        st.rerun()
    role = st.session_state.get("user_role", "Admin").lower().replace(" ", "_")
    allowed_pages = {"admin": set(ROUTES), "hr_manager": set(ROUTES) - {"api"},
                     "recruiter": {"dashboard", "insights", "resume", "settings"},
                     "analyst": {"dashboard", "data", "insights", "forecast", "fairness", "models", "api", "settings"}}
    if st.session_state["page"] not in allowed_pages.get(role, allowed_pages["admin"]):
        st.session_state["page"] = "dashboard"
        st.warning("Your account role cannot open that page.")
    try:
        ROUTES[st.session_state["page"]]()
    except Exception as exc:               # last line of defence: a friendly message, never a stack trace
        st.error(f"Something went wrong while building this page ({type(exc).__name__}). Please refresh; if it keeps happening, tell the maintainer.")
        st.exception(exc) if SETTINGS.env == "development" else None


main()

