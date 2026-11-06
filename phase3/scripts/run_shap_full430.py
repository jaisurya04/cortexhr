"""Task 14 in one step: score the 430 employees, explain ALL of them with the real shap library (PermutationExplainer),
build the 3 plain-English reasons and write the report.

Needs:  pip install -r requirements.txt        (includes shap==0.54.1)
Run:    python scripts/run_shap_full430.py     (a few minutes on one CPU)
Then:   python scripts/plot_shap_full430.py    (charts, needs only matplotlib)

Writes artifacts/predictions/attrition_risk_2026.csv, reports/task14/attrition_risk_2026.csv (identical copies),
reports/task14/shap_full430_feature_values.csv, shap_full430_group_values.csv, shap_full430_summary.json,
explanation_run.json and shap_full430_run_log.txt."""
import json, sys, datetime, platform
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "05_Starter"))
LOG = ROOT / "reports" / "task14" / "shap_full430_run_log.txt"; LOG.parent.mkdir(parents=True, exist_ok=True)
class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, t): [s_.write(t) for s_ in self.streams]; return len(t)
    def flush(self): [s_.flush() for s_ in self.streams]
_logfile = open(LOG, "w", encoding="utf-8"); sys.stdout = _Tee(sys.__stdout__, _logfile); sys.stderr = _Tee(sys.__stderr__, _logfile)
print("run started (UTC):", datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "| python", platform.python_version(), "|", platform.platform())
try:
    import shap  # noqa: F401
except Exception as e:
    sys.exit(f"shap is not installed ({e!r}). Run: pip install -r requirements.txt")
from cortexhr.phase3.explanations.risk_report import run_task14
o = run_task14(write=True)
failed = [c for c in o["checks"] if not c["passed"]]
print(json.dumps({"shap_version": o["shap_version"], "rows": len(o["report"]), "background_rows": o["background_rows"], "additivity_max_error": o["additivity_max_error"],
                  "format_checks": f"{len(o['checks']) - len(failed)}/{len(o['checks'])}", "bands": o["report"].risk_band.value_counts().to_dict()}, indent=2))
assert not failed, failed
print("log saved to", LOG)
