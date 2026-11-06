"""Run the REAL shap library (PermutationExplainer) on ALL 430 employees and compare it with the saved Monte Carlo reasons.

Needs: pip install -r requirements.txt -r requirements-optional.txt   (shap==0.50.0)
Run:   python scripts/run_shap_full430.py          (a few minutes on one CPU)

Writes reports/task14/shap_full430_group_values.csv, shap_full430_comparison.csv and shap_full430_summary.json.
It does NOT change the 430-row report; it proves that the grouped Monte Carlo reasons agree with the real library on everyone."""
import json, sys, datetime, platform
from pathlib import Path
import numpy as np, pandas as pd
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
    sys.exit(f"shap is not installed ({e!r}). Run: pip install -r requirements.txt -r requirements-optional.txt")
import joblib
from cortexhr.phase3.attrition import v2 as V
from cortexhr.phase3.explanations import risk_report as RR
from cortexhr.phase3.explanations.groups import GROUPS, SENSITIVE_PROXY
from cortexhr.phase3.explanations.shap_explainer import explain_with_shap
from cortexhr.phase3.common.config import load_config
OUT = ROOT / "reports" / "task14"
cfg = load_config(); seed = cfg["seed"]
model = joblib.load(RR.MODELS / "attrition_v2.joblib"); scoring, X, _ = V.load_scoring()
bg = RR.build_background(); predict = lambda d: model.predict_proba(d)[:, 1]
res = explain_with_shap(predict, X, bg, n_rows=len(X), groups=GROUPS, seed=seed, max_background=50)
assert res is not None and len(res["predictions"]) == len(X) == 430, "expected real SHAP values for all 430 employees"
ids = scoring.reset_index(drop=True)["employee_id"].values
feat = res["feature_values"].copy(); feat.insert(0, "employee_id", ids); feat["base_value"] = res["base_values"].values; feat["prediction"] = res["predictions"].values
feat.to_csv(OUT / "shap_full430_feature_values.csv", index=False)
fsum = feat.drop(columns=["employee_id", "base_value", "prediction"]).sum(axis=1) + feat["base_value"]; feat_additivity = float((fsum - feat["prediction"]).abs().max())
assert len(feat) == 430 and feat_additivity < 1e-6, f"feature-level additivity check failed: {feat_additivity}"
shap_g = res["group_values"].copy(); shap_g.insert(0, "employee_id", ids); shap_g.to_csv(OUT / "shap_full430_group_values.csv", index=False)
mc = pd.read_csv(OUT / "group_contributions.csv").set_index("employee_id").loc[ids]
cols = [c for c in GROUPS if c not in SENSITIVE_PROXY]
def top3(df): return [list(r.sort_values(ascending=False).index[:3]) for _, r in df[cols].iterrows()]
t_mc, t_sh = top3(mc.reset_index(drop=True)), top3(res["group_values"])
ov = np.array([len(set(a) & set(b)) / 3 for a, b in zip(t_mc, t_sh)])
corr = np.array([np.corrcoef(mc.iloc[i][list(GROUPS)].to_numpy(float), res["group_values"].iloc[i][list(GROUPS)].to_numpy(float))[0, 1] for i in range(len(ids))])
cmp = pd.DataFrame({"employee_id": ids, "monte_carlo_top3": [" | ".join(t) for t in t_mc], "shap_library_top3": [" | ".join(t) for t in t_sh], "top3_overlap": ov, "same_top1": [a[0] == b[0] for a, b in zip(t_mc, t_sh)], "group_corr": corr})
cmp.to_csv(OUT / "shap_full430_comparison.csv", index=False)
summary = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "shap_version": res["shap_version"], "rows": len(ids), "background_rows": res["background_rows"],
           "additivity_max_error": res["additivity_max_error"], "feature_level_rows": len(feat), "feature_level_additivity_max_error": feat_additivity, "python": platform.python_version(), "platform": platform.platform(), "log_file": "reports/task14/shap_full430_run_log.txt", "files": ["shap_full430_feature_values.csv", "shap_full430_group_values.csv", "shap_full430_comparison.csv", "shap_full430_summary.json"], "mean_top3_overlap": float(ov.mean()), "share_all_three_identical": float((ov == 1).mean()),
           "share_same_top1": float(cmp.same_top1.mean()), "median_group_correlation": float(np.median(corr))}
(OUT / "shap_full430_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8"); print(json.dumps(summary, indent=2))

print("log saved to", LOG)

