"""The monitor: drift (PSI) + fairness re-checks -> one report and one status (OK / WATCH / ALERT)."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..core.settings import PHASE3, Settings, load_settings
from . import drift, fairness, status

REPORT_VERSION = "monitor-1.0.0"
ALERT_PLAN = ["1. Confirm it is real: check the data pipeline, units and column mapping before blaming the world.",
              "2. Locate it: which features and which departments moved, and since which month?",
              "3. Measure impact: compare recent model scores and, once outcomes arrive, accuracy and calibration against the baseline.",
              "4. Decide and record: recalibrate, retrain or roll back, keep a human override in place, and write the decision in the audit log."]
EXIT_CODE = {status.OK: 0, status.WATCH: 1, status.ALERT: 2}


def run_monitor(settings: Settings | None = None, now: datetime | None = None) -> dict:
    """Run every check and return the report dict (does not write files)."""
    s = settings or load_settings()
    now = now or datetime.now(timezone.utc)
    comps, errors = {}, {}
    for name, fn in (("drift", lambda: drift.run_drift(s.data_dir, PHASE3 / "reports" / "task13" / "permutation_importance.csv")),
                     ("twin_resume_gap", lambda: fairness.twin_resume_check(s)), ("pay_gap", lambda: fairness.pay_gap_check(s))):
        try:
            comps[name] = fn()
        except Exception as exc:            # a failed check must be visible, never silently OK
            errors[name] = f"{type(exc).__name__}: {exc}"
            comps[name] = {"status": status.WATCH, "error": errors[name]}
    judged = [c["status"] for c in comps.values()]
    skipped = [f"pay_gap/{k}" for k, v in comps.get("pay_gap", {}).get("groups", {}).items() if v.get("status") == status.SKIPPED]
    overall = status.worst(judged)
    report = {"report_version": REPORT_VERSION, "created_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "status": overall,
              "coverage": "partial" if (skipped or errors) else "complete", "not_checked": skipped, "errors": errors,
              "thresholds": {"psi_watch": status.PSI_WATCH, "psi_alert": status.PSI_ALERT, "twin_tolerance": status.TWIN_TOLERANCE,
                             "pay_gap_watch": status.PAY_GAP_WATCH, "pay_gap_alert": status.PAY_GAP_ALERT, "min_group_size": status.MIN_GROUP_SIZE},
              "components": {k: _jsonable(v) for k, v in comps.items()}}
    report["explanation"] = explain(report)
    if overall == status.ALERT:
        report["next_steps"] = ALERT_PLAN
    return report


def _jsonable(c: dict) -> dict:
    out = {}
    for k, v in c.items():
        out[k] = v.round(4).to_dict("records") if isinstance(v, pd.DataFrame) else v
    return out


def explain(report: dict) -> list[str]:
    """Plain-language reasons behind the status."""
    lines, c = [], report["components"]
    d = c.get("drift", {})
    if "table" in d:
        t = pd.DataFrame(d["table"]); bad = t[t.status != status.OK].sort_values("psi", ascending=False)
        collapsed = bad[(bad.status == status.ALERT) & (bad.current_top_value_share_pct >= 50)]
        lines.append(f"Drift: {len(t)} features checked, max PSI {t.psi.max():.3f} ({t.loc[t.psi.idxmax(), 'feature']}); "
                     + (f"{len(bad)} above the WATCH line: " + ", ".join(f"{r.feature} {r.psi:.3f}" for r in bad.itertuples()) if len(bad) else "none above the WATCH line (0.10)."))
        for r in collapsed.itertuples():
            lines.append(f"Note: {r.feature} has collapsed rather than drifted: {r.current_top_value_share_pct:.0f}% of the current population share the value "
                         f"{r.current_top_value:g} (reference mean {r.reference_mean:.1f}, current mean {r.current_mean:.1f}). Check whether a company-wide event "
                         f"(for example a raise cycle) or a pipeline change explains it before trusting scores that use it.")
    tw = c.get("twin_resume_gap", {})
    if "summary" in tw:
        s = tw["summary"]; lines.append(f"Twin resumes: {s['comparisons']} comparisons, max gap {s['max_abs_gap']:.6f} (tolerance {tw['tolerance']}); status {tw['status']}.")
    pg = c.get("pay_gap", {}).get("groups", {})
    for k, v in pg.items():
        lines.append(f"Pay gap by {k}: " + (f"model gap {v['gap_model']:.4f} vs history {v['gap_history']:.4f}; status {v['status']}." if "gap_model" in v else v.get("reason", "")))
    for k, v in report["errors"].items():
        lines.append(f"Check '{k}' failed to run ({v}); treated as WATCH.")
    return lines


def render_text(report: dict) -> str:
    """One clear console report."""
    w = ["=" * 78, f"CortexHR monitoring report   {report['created_utc']}   status: {report['status']}   coverage: {report['coverage']}", "=" * 78]
    d = report["components"].get("drift", {})
    if "table" in d:
        w += ["", f"FEATURE DRIFT (PSI) - {d['reference']} vs {d['current']}", f"{'feature':34s}{'PSI':>8s}  {'status':7s}{'ref mean':>11s}{'cur mean':>11s}"]
        w += [f"{r['feature']:34s}{r['psi']:8.3f}  {r['status']:7s}{r['reference_mean']:11.2f}{r['current_mean']:11.2f}" for r in d["table"]]
    w += ["", "FAIRNESS RE-CHECKS"] + ["  " + line for line in report["explanation"] if not line.startswith("Drift")]
    if report["not_checked"]:
        w += ["", "NOT CHECKED: " + ", ".join(report["not_checked"])]
    w += ["", "WHY: " + report["explanation"][0] if report["explanation"] else "", "", "Limits: " + "; ".join(
        v["limit"] for v in (report["components"].get("twin_resume_gap", {}), report["components"].get("pay_gap", {})) if "limit" in v)]
    if "next_steps" in report:
        w += ["", "ALERT - NEXT STEPS"] + report["next_steps"]
    w.append("=" * 78)
    return "\n".join(w)


def persist(report: dict, settings: Settings | None = None) -> dict[str, Path]:
    """Write latest report (JSON + Markdown) and append one line to the audit history (JSONL)."""
    s = settings or load_settings(); out = s.monitoring_dir; out.mkdir(parents=True, exist_ok=True)
    latest, hist, md = out / "monitoring_latest.json", out / "monitoring_history.jsonl", out / "monitoring_report.md"
    tmp = latest.with_suffix(".json.tmp")                         # write-then-rename: a reader never sees a half-written file
    tmp.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, latest)
    brief = {"created_utc": report["created_utc"], "status": report["status"], "coverage": report["coverage"],
             "component_status": {k: v["status"] for k, v in report["components"].items()},
             "max_psi": max((r["psi"] for r in report["components"].get("drift", {}).get("table", [])), default=None)}
    with hist.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(brief) + "\n")
    md.write_text("```\n" + render_text(report) + "\n```\n", encoding="utf-8")
    return {"latest": latest, "history": hist, "markdown": md}


def read_history(settings: Settings | None = None) -> list[dict]:
    p = (settings or load_settings()).monitoring_dir / "monitoring_history.jsonl"
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []
