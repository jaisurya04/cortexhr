"""Generate the numbers-heavy documents from the saved evaluation files so they can never disagree with the evidence.

    python phase4/scripts/build_docs.py     ->  docs/model_cards/*.md, docs/architecture.png, docs/forecasting_report.md, docs/final_evaluation.md
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cortexhr_phase4.core.settings import PHASE2, PHASE3, PHASE4_ROOT, load_settings  # noqa: E402
from cortexhr_phase4.forecasting import data as fdata, eda  # noqa: E402
from cortexhr_phase4.services import handlers as h, samples  # noqa: E402
from cortexhr_phase4.services.registry import Registry  # noqa: E402

DOCS = PHASE4_ROOT / "docs"
OWNER, TODAY = "Jaisurya Marisamy (GM00134), AI/ML Engineer, Growmar", date.today().isoformat()
R3 = PHASE3 / "reports"
PHASE2_MODEL_REGISTRY = json.loads((PHASE2 / "models" / "model_registry.json").read_text(encoding="utf-8"))


def model_version(reg: Registry, name: str) -> str:
    info = reg.info.get(name)
    if info and info.version:
        return info.version
    saved = PHASE2_MODEL_REGISTRY.get(name, {}).get("version")
    if saved:
        return str(saved)
    raise RuntimeError(f"No verified version metadata is available for model {name!r}.")


def md_table(df: pd.DataFrame, nd: int = 3) -> str:
    d = df.copy()
    for c in d.select_dtypes("float").columns:
        d[c] = d[c].round(nd)
    head = "| " + " | ".join(map(str, d.columns)) + " |\n|" + "---|" * len(d.columns) + "\n"
    return head + "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in d.itertuples(index=False)) + "\n"


def card(name: str, version: str, purpose: str, not_for: str, data: str, method: str, metrics: str, expl: str, fairness: str, limits: str) -> str:
    return (f"# Model Card: {name}   (CX-20)\n\n**Purpose** {purpose}  \u00b7  **Not for** {not_for}\n\n**Data** {data}\n\n**Method** {method}\n\n**Metrics**\n\n{metrics}\n\n"
            f"**Explanations** {expl}\n\n**Fairness** {fairness}\n\n**Limits & risks** {limits}\n\n**Version & owner** `{version}` \u00b7 card written {TODAY} \u00b7 {OWNER}\n\n"
            "_Every output of this model is AI-generated and needs human review. Numbers in this card are copied from the evaluation files named in the text by `scripts/build_docs.py`._\n")


def build_cards(reg: Registry) -> dict[str, str]:
    s = load_settings()
    cards: dict[str, str] = {}
    # ---------------------------------------------------------------- attrition
    cv = pd.read_csv(R3 / "task13" / "cv_summary_with_lightgbm.csv")[["model", "roc_auc_mean", "pr_auc_mean", "brier_mean", "roc_auc_sd"]]
    pd_ = pd.read_csv(R3 / "task13" / "paired_differences.csv")
    imp = pd.read_csv(R3 / "task13" / "permutation_importance.csv").head(5)
    gen, age = pd.read_csv(R3 / "task13" / "group_gender.csv"), pd.read_csv(R3 / "task13" / "group_age.csv")
    ex = h.attrition_predict(reg, samples.employee_id())
    bands = h.attrition_list(reg, limit=1)["bands"]
    cards["attrition_v2"] = card("CortexHR attrition v2 (12-month leave risk)", "attrition-v2",
        "estimates the chance that an active employee leaves within 12 months so HR can prioritise conversations.",
        "dismissal, pay, promotion, hiring or any decision about a person without human review; not as a certainty.",
        f"`attrition_dataset.csv` (2,383 rows after removing 6 exact duplicates; snapshots 2021-01 to 2025-09); post-outcome columns, ids and `source` are removed as inputs; no protected attribute is an input. Scoring population: `attrition_scoring_2026.csv`, 430 active employees (as of 2026-09-30). Source evidence: `phase3/reports/task13/`.",
        "Phase 1 leakage-safe pipeline, then an equal-weight blend of balanced logistic regression and tuned HistGradientBoosting, calibrated with a sigmoid learned inside the folds; stratified 5-fold CV (seed 42) plus one untouched test split. A separate LightGBM challenger was tested on the same folds.",
        "Cross-validated, same folds (`phase3/reports/task13/cv_summary_with_lightgbm.csv`):\n\n" + md_table(cv) + "\nPaired fold differences against the baseline (`paired_differences.csv`):\n\n" + md_table(pd_) +
        f"\n**Verdict (Phase 3):** the gain over the simple logistic baseline is **not proven** (the 95% interval includes 0). Test split (used once): ROC-AUC 0.704, PR-AUC 0.382, Brier 0.113, ECE 0.028. Risk bands on the 2026 population: {bands}.",
        "Top permutation drivers (AUC drop):\n\n" + md_table(imp, 4) + f"\nWorked example `{ex['employee_id']}` ({ex['department_id']}): risk {ex['risk_score_12m'] * 100:.1f}% ({ex['risk_band']}); reasons shown to HR: " + "; ".join(ex["top_reasons"]) + ". Reasons come from the real `shap` library (PermutationExplainer 0.54.1) on 23 HR groups and show only factors that raised the score.",
        "Group tables from `task13/group_gender.csv` and `group_age.csv` (test predictions, n per group):\n\n" + md_table(gen[["gender", "n", "actual_leave_rate", "mean_predicted_risk", "recall_of_leavers_in_top10", "auc"]]) + "\n" + md_table(age[["age_band", "n", "actual_leave_rate", "mean_predicted_risk", "recall_of_leavers_in_top10", "auc"]]) +
        "\nMean predicted risk is similar across groups; the age band 45+ has only 28 people so no conclusion is possible. Similar averages do not prove equal treatment of individuals. Gender and age are never model inputs.",
        "AUC around 0.70 means many leavers are missed and many stayers flagged. **Input drift (2026):** the latest monitoring run reports an ALERT for `months_since_last_raise` (see `/v1/monitoring`), so 2026 scores that rely on this input should be read with extra caution until the cause is understood. Class-weighted models overstate risk until calibrated.")
    # ---------------------------------------------------------------- payroll / attendance / chatbot (Phase 2 evidence)
    att = pd.read_csv(PHASE2 / "reports" / "tables" / "attendance_detector_metrics.csv")
    cards["payroll_anomaly"] = card("CortexHR payroll anomaly detector", model_version(reg, "payroll"),
        "flags payslips that look unusual so a payroll clerk reviews them before payout.", "proof of error or fraud, deductions, discipline or any automatic payroll action.",
        "`payroll_monthly_features.csv`: 7,355 payslips, 18 monthly runs (2025-04 to 2026-09), synthetic. Training is unsupervised (no labels). Evidence: `phase2/reports/phase2/workflow38/payroll_model_documentation.md`.",
        "23 numeric features (ratios, identity gaps, amounts); median imputation, scaling; Isolation Forest (300 trees, contamination 0.03, seed 42) combined with rule checks (negative / zero / duplicate pay, gross-minus-deductions identity). Temporal split: train 2025-04..2026-03, validation 2026-04..06, test 2026-07..09.",
        "Test split (evaluated once, against synthetically injected errors, not the vault labels): precision 0.541, recall 0.958, F1 0.692, ROC-AUC 0.982, PR-AUC 0.702, flag share 6.5%. Production operating point flags 5.68% of real rows.",
        "Each flag carries reason codes and an anomaly score; the API returns them with every result (`/v1/anomaly/payroll`).",
        "Phase 2 fairness screen: 3 groups flagged more than 2x the overall rate (overall 5.68%). Flags are review candidates; reviewers should check the over-flagged groups.",
        "Precision depends on how common errors really are; synthetic data; money-at-risk is a proxy; known weak spot: inflated variable pay. Monthly PSI/flag-rate monitoring raised alerts in 2025-04, 2025-10, 2026-02 and 2026-04.")
    cards["attendance_anomaly"] = card("CortexHR attendance anomaly detector", model_version(reg, "attendance"),
        "flags unusual attendance months for HR review.", "performance rating, discipline or any automatic action.",
        "`attendance_monthly_features.csv`, 7,339 employee-months; unsupervised training, contamination 0.03.", "Isolation Forest on own-history features combined with rules (`phase2/reports/tables/attendance_detector_metrics.csv`).",
        "Against 150 injected anomalies (`attendance_detector_metrics.csv`):\n\n" + md_table(att) + "\nThe last row is marked POST-HOC and optimistic (tuned after seeing results); use the combined IF + rules row (precision 0.50, recall 0.84) as the honest figure.",
        "Rule-based reasons plus the model score are returned with each result.", "Not separately analysed by protected group in Phase 2 beyond the group flag-rate screen; no protected attribute is an input.",
        "About half of the flags are false alarms; injected anomalies are simpler than real ones; remote-work patterns can look unusual.")
    cards["chatbot_intent"] = card("CortexHR HR assistant (intent model)", model_version(reg, "chatbot"),
        "recognises what an employee is asking about HR (leave, payslip, policy ...) and returns a vetted answer or routes to a service.", "legal, medical or disciplinary advice; any answer that changes records.",
        "858 labelled example sentences across 33 intents (`chatbot_nlu_training.csv`, original set, no test-driven additions) plus an FAQ and response table. Held-out test 132 rows.",
        "TF-IDF + logistic regression; confidence threshold 0.30 (below it the bot asks for clarification); roles come from the identity directory, never from the message.",
        "Held-out accuracy **0.811** (25 errors in 132 rows) from `phase2/reports/phase2/chatbot_metric_summary.csv`.",
        "Every reply shows its intent and confidence in the console.", "Not tested by demographic group; wording is English only.",
        "Questions needing personal records call the Phase 2 mock HR service; without it the assistant says so instead of guessing. Never reveal other people's records (role-based access).")
    # ---------------------------------------------------------------- resume ranker
    rq = pd.read_csv(R3 / "task11" / "ranking_quality_summary.csv")
    tw = pd.read_csv(R3 / "task12" / "before_after_gap.csv")
    jb = h.resume_rank(reg, {"job_id": "J01", "top_k": 1})["candidates"][0]
    cards["resume_ranker_v2"] = card("CortexHR resume ranker v2 (masked)", reg.info["resume"].version,
        "orders resumes for one job so a recruiter reads the most relevant first.", "rejecting anyone automatically; jobs or skills outside its vocabulary; as a probability of success.",
        "185 resumes, 10 job descriptions, 555 human relevance labels (`resume_job_labels.csv`). Name, e-mail, city and college-tier text are masked before scoring. Evidence: `phase3/reports/task11/`, `task12/`.",
        "TF-IDF similarity + required and nice-to-have skill coverage + capped experience, weights fixed in advance and checked with 286 alternative weightings and leave-one-job-out; neutral hash tie-break.",
        md_table(rq) + "\nWeakest job J02 has only 3 relevant candidates, so its precision is unstable.",
        f"Every score is built from four numbers; example top candidate for J01: `{jb.get('resume_id')}` with score {float(jb['score']):.3f}.",
        "Twin-resume test (25 pairs, 75 comparisons): score gap before masking and after:\n\n" + md_table(tw) + "\nThe gap is zero after masking **by construction**; it shows that name, e-mail, city and tier text no longer move the score, not that the system is fair. Other proxies were not tested.",
        "Ten jobs and 555 labels are a small basis. Re-test fairness after every change.")
    # ---------------------------------------------------------------- pay band
    tm = pd.read_csv(R3 / "task15" / "test_metrics.csv")[["model", "MAE_rupees", "R2", "median_abs_pct_error"]]
    gg = pd.read_csv(R3 / "task15" / "gap_bin_gaps.csv")
    meta = json.loads((PHASE3 / "artifacts" / "metadata" / "salary_band_v1_metadata.json").read_text(encoding="utf-8"))
    pb = h.payband(reg, samples.payband_row())
    cards["salary_band_v1"] = card("CortexHR pay-band advisor v1", reg.info["payband"].version,
        "suggests a low / mid / high fixed-pay range for a role so a manager and HR can discuss an offer.", "setting an individual's pay, or any decision made without a person.",
        "`salary_training.csv` (past offers, 2,934 rows). Inputs are job-related only: department, role, level, band mid-point, market index, experience, skills, location, education. Gender, age, career gap, hire date and source are never inputs.",
        "Ridge regression on log salary (chosen over gradient boosting); the band is the prediction scaled by out-of-fold error quantiles so the range is honest; test hold-out used once.",
        md_table(tm) + f"\nBand coverage on the test set: **{meta['band_coverage_test']:.3f}** (target 0.80).",
        f"Worked example (example row from the training file): low \u20b9{pb['recommended_low']:,.0f}, mid \u20b9{pb['recommended_mid']:,.0f}, high \u20b9{pb['recommended_high']:,.0f}.",
        "Mean (predicted pay / band reference) gap between best and worst group (`task15/gap_bin_gaps.csv`):\n\n" + md_table(gg[["measure", "group_gap"]]) + "\nThe model that is *given* career gap would widen the career-gap gap (0.046); the chosen model, which never sees it, has a gap of 0.0085. The history itself contains a career-gap penalty, which the model must not copy.",
        "Past offers may contain past bias; only offers of the last years; the gender comparison needs a restricted file that is not in the repository.")
    # ---------------------------------------------------------------- forecast
    cmp = pd.read_csv(s.forecast_dir / "model_comparison.csv")
    fmeta = json.loads((s.forecast_dir / "forecast_run.json").read_text(encoding="utf-8"))
    sel = cmp[(cmp.department_id == "ALL") & (cmp.scope != "failed")][["metric", "model", "method", "scope", "n", "mae", "mape_pct", "bias", "coverage_80"]]
    study = pd.read_csv(s.forecast_dir / "interval_calibration_study.csv").groupby(["metric", "interval_method"])[["coverage_80"]].mean().reset_index()
    status = fmeta["improved_model_status"]
    cards["headcount_forecast"] = card("CortexHR headcount and exits forecast", fmeta["model_version"],
        "projects company and department headcount and company exits for the next six months with an 80% range.", "individual decisions, budgets set without review, or forecasts beyond 12 months.",
        f"`monthly_headcount_history.csv` (81 months, {fmeta['trained_through']} last) and `workforce_events.csv` (9 events, none after 2026-04). Department D00 (1 person) is not forecast.",
        "Baselines: *seasonal naive + yearly drift* for headcount and *same-month average of 3 years* for exits. Improved model: events + yearly seasonality regression with AR(1) errors (statsmodels SARIMAX) on the monthly net change; interval from the model's own error recursion. The improved model replaces the baseline only if its rolling-origin MAE is at least 1% lower.",
        f"**Improved-model status in the saved run: {status}.**\n\n" + md_table(sel) + "\nholdout = last 6 known months hidden and forecast; rolling_origin = the same test from many past starting points. Coverage should be near 0.80.\n\nBaseline interval calibration check (`interval_calibration_study.csv`, mean coverage across series):\n\n" + md_table(study),
        "Past events enter the improved model as 0/1 regressors (hiring freeze, hiring boost, policy change). Plain-language explanation in `docs/forecasting_report.md`.",
        "Not applicable: the forecast uses counts, not people's attributes.",
        "Only one hiring freeze (2020) is available to learn its effect; intervals ignore parameter uncertainty; the baseline 80% interval covered clearly less than 80% in rolling tests, so treat ranges as optimistic.")
    return cards


def architecture_png(path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(13, 7.6)); ax.set_xlim(0, 13); ax.set_ylim(0, 7.6); ax.axis("off")
    def box(x, y, w, hh, title, lines, fc="#0a3a37", ec="#c9a24d"):
        ax.add_patch(FancyBboxPatch((x, y), w, hh, boxstyle="round,pad=0.02,rounding_size=0.15", fc=fc, ec=ec, lw=1.4))
        ax.text(x + w / 2, y + hh - 0.28, title, ha="center", va="center", color="#ead9ae", fontsize=10.5, fontweight="bold")
        for i, t in enumerate(lines):
            ax.text(x + 0.15, y + hh - 0.62 - i * 0.27, t, ha="left", va="center", color="#e9efe9", fontsize=8.3)
    def arrow(x1, y1, x2, y2, label=""):
        ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", color="#d4a95f", lw=1.5))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, label, ha="center", color="#cfe0d8", fontsize=7.8)
    fig.patch.set_facecolor("#06302f")
    box(0.2, 5.2, 3.9, 2.1, "Phase 1-3 (read-only)", ["data: 03_Data/csv (history, events, attrition, salary,", "resumes, jobs)", "models: attrition_v2, resume_ranker_v2, salary_band_v1,", "Phase 2 payroll / attendance / chatbot", "reports: saved evaluation tables"])
    box(0.2, 2.6, 3.9, 2.1, "Phase 4 core (cortexhr_phase4)", ["core/legacy.py  read-only bridge (sha256-verified)", "forecasting/  baseline, SARIMAX, backtest, events", "monitoring/  PSI, twin-resume, pay-gap, status", "services/  Registry (load once) + handlers"], fc="#0d4a43")
    box(4.7, 4.5, 3.4, 2.8, "Unified API  /v1  (FastAPI + Pydantic)", ["health  models  overview  performance", "attrition  anomaly/payroll|attendance  chat", "resume/rank + upload  payband/recommend", "forecast  forecast/scenario  monitoring", "PBKDF2 login  HMAC sessions  role RBAC", "consistent JSON errors  request-id logs"], fc="#0d4a43")
    box(4.7, 1.3, 3.4, 2.5, "Outputs  (phase4/outputs)", ["forecast/  6-month CSV, backtests, explanation", "monitoring/  latest report + history.jsonl", "console/  preview + screenshots"])
    box(8.9, 4.5, 4.0, 2.8, "Insights Console  (Streamlit)", ["dashboard canvas + interactive workforce map", "Data Foundation  Insights  Forecast", "Fairness & Audit  Models + saved SHAP", "Resume PDF/DOCX  API & Deploy  Settings", "role-aware navigation + human review labels"], fc="#0d4a43")
    box(8.9, 1.3, 4.0, 2.5, "Scheduled monitor + smoke test", ["scripts/monitor.py  exit 0 / 1 / 2 = OK / WATCH / ALERT", "GitHub Actions or cron (deploy/)", "scripts/smoke_test.py  /v1/health + 1 call per model"])
    box(0.2, 0.2, 12.7, 0.8, "Deployment", ["API: Render web service (env vars)   Console: Streamlit Community Cloud (CORTEXHR_API_URL secret)   Tests: tests/run_all.py"], fc="#082b29")
    arrow(2.15, 5.2, 2.15, 4.72, "read"); arrow(4.1, 3.6, 4.7, 5.2, "services"); arrow(8.1, 5.9, 8.9, 5.9, "HTTP /v1"); arrow(6.4, 4.5, 6.4, 3.82, "reads")
    arrow(4.1, 3.1, 4.7, 2.5, "writes"); arrow(8.9, 2.55, 8.1, 2.55, "writes")
    ax.text(6.5, 7.45, "CortexHR v1.0.0 architecture", ha="center", color="#ead9ae", fontsize=14, fontweight="bold")
    fig.savefig(path, dpi=140, facecolor=fig.get_facecolor(), bbox_inches="tight"); plt.close(fig)


def forecasting_report(reg: Registry) -> str:
    s = load_settings()
    hist, ev = fdata.load_history(s.data_dir), fdata.load_events(s.data_dir)
    d = eda.describe_history(hist, ev)
    ex = json.loads((s.forecast_dir / "forecast_explanation.json").read_text(encoding="utf-8"))
    meta = json.loads((s.forecast_dir / "forecast_run.json").read_text(encoding="utf-8"))
    fc = pd.read_csv(s.forecast_dir / "forecast_next_6_months.csv")
    hc = fc[(fc.metric == "headcount_end") & (fc.department_id == "ALL")][["forecast_month", "forecast_value", "lower_80", "upper_80"]]
    unusual = pd.DataFrame(d["unusual_months"]); unusual["events"] = unusual.events.map(", ".join)
    cmp = pd.read_csv(s.forecast_dir / "model_comparison.csv"); cmp = cmp[(cmp.department_id == "ALL") & (cmp.scope != "failed")][["metric", "model", "scope", "n", "mae", "mape_pct", "bias", "coverage_80"]]
    tl = pd.DataFrame(ex["trust_by_month"])
    mo = pd.DataFrame(d["monthly_averages"])
    return f"""# Day 16 - Forecast the future: headcount with uncertainty

Generated {TODAY} by `scripts/build_docs.py` from `outputs/forecast/`. Run `python scripts/run_forecast.py` first to refresh the files.

## 1. What the history looks like (step 1)
![history](../outputs/forecast/headcount_history.png)

* **Trend:** headcount grew from {d['start_headcount']:.0f} ({d['first_month']}) to {d['end_headcount']:.0f} ({d['last_month']}), about {d['average_growth_per_month']:.1f} people per month on average. Year-end growth: {', '.join(f'{k}: {v:+.1f}%' for k, v in d['year_end_growth_pct'].items())}.
* **Seasons:** hires peak in month {d['season_peaks']['hires_peak_month']} and are lowest in month {d['season_peaks']['hires_low_month']}; exits peak in month {d['season_peaks']['exits_peak_month']} and are lowest in month {d['season_peaks']['exits_low_month']} (average per calendar month below).
* **Unusual months** (net change far from normal, robust z-score >= 3) and the events that overlap them:

{md_table(unusual)}
Average hires, exits and net change per calendar month:

{md_table(mo)}

## 2. Models (steps 2-4)
* **Seasonal baseline** ("same as last year plus trend"): value 12 months earlier + the latest year-on-year change (headcount); same-month average of the last 3 years (exits).
* **Better model:** statsmodels SARIMAX regression with AR(1) errors on the monthly net change, with workforce events (hiring freeze, hiring boost, policy change) and yearly seasonality as regressors. **Status in the saved run: {meta['improved_model_status']}.**
* **Workforce events:** {ex['events_handling']} Future events known in the data: {ex['future_events_known']}.

## 3. Backtest (step 5): hide the last 6 months, forecast them, measure the error
{md_table(cmp)}
`holdout_last_6_months` is exactly the requested test (2026-04 to 2026-09 hidden). `rolling_origin` repeats it from many past starting points for a steadier picture. `coverage_80` should be near 0.80; the baseline intervals are calibrated on the baseline's own past errors (see `interval_calibration_study.csv`).

## 4. The forecast (step 6)
Company headcount, 80% interval; all columns match `sample_forecast.csv` (`outputs/forecast/forecast_next_6_months.csv`).

{md_table(hc, 1)}
Headcount moves from {ex['current_headcount']:.0f} to {ex['headcount_in_6_months']:.0f} ({ex['growth_pct_6m']:+.1f}%), with about {ex['expected_exits_6m']:.0f} exits in six months. Model: {meta['model_version']}, trained through {meta['trained_through']}.

## 5. Think for yourself: which month would you trust least? Why?
{md_table(tl)}
**{ex['least_trusted_month']['month'][:7]}**: {ex['least_trusted_month']['why']} The range widens every month, so the sixth month is the least certain.

## 6. Limits
""" + "\n".join(f"* {x}" for x in ex["limits"]) + "\n"


def final_evaluation(reg: Registry) -> str:
    perf = h.performance(reg)
    mon = h.monitoring(reg)
    s = load_settings()
    lines = [f"# Day 20 - Final evaluation (re-run {TODAY})\n",
             "Phase 4 evaluations (forecast backtests, monitoring, tests) were re-run for this release. Phase 1-3 model evaluations are **read from their saved evaluation files**, not re-trained here (the restricted validation vault and the pinned scikit-learn 1.8.0 are needed to re-train them).\n",
             f"## Monitoring status now: {mon['status']} (coverage {mon['coverage']})\n", "\n".join(f"* {x}" for x in mon["explanation"]) + "\n",
             "## Attrition v2 (cross-validation)\n", md_table(pd.DataFrame(perf["attrition"]["cv"])), f"**{perf['attrition']['verdict']}**\n",
             "## Resume ranker\n", md_table(pd.DataFrame(perf["resume"]["ranking"])), "## Pay-band model\n", md_table(pd.DataFrame(perf["payband"]["test_metrics"])),
             "## Headcount forecast (company)\n", md_table(pd.DataFrame(perf["forecast"]["comparison"])[["metric", "model", "scope", "n", "mae", "mape_pct", "coverage_80"]]) if perf.get("forecast") else "not available\n"]
    return "\n".join(lines)


def main() -> None:
    reg = Registry(load_settings()).load_all()
    (DOCS / "model_cards").mkdir(parents=True, exist_ok=True)
    for name, text in build_cards(reg).items():
        (DOCS / "model_cards" / f"{name}.md").write_text(text, encoding="utf-8")
    architecture_png(DOCS / "architecture.png")
    (DOCS / "forecasting_report.md").write_text(forecasting_report(reg), encoding="utf-8")
    (DOCS / "final_evaluation.md").write_text(final_evaluation(reg), encoding="utf-8")
    print("wrote", sorted(p.name for p in (DOCS / "model_cards").glob("*.md")), "architecture.png forecasting_report.md final_evaluation.md")


if __name__ == "__main__":
    main()
