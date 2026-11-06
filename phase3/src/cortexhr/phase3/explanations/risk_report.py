"""Task 14: score the 430 active employees with attrition-v2, band them, explain each, and export in the sample's exact format."""
from __future__ import annotations
import importlib.util
import joblib
import numpy as np
import pandas as pd
from cortexhr.phase3.attrition import v2 as V
from cortexhr.phase3.common.config import load_config
from cortexhr.phase3.common.io import read_csv, save_json, save_table, utc_now
from cortexhr.phase3.common.paths import MODELS, PREDICTIONS, ROOT, task_dir
from cortexhr.phase3.common.validation import require_probability, require_columns, require_unique
from .groups import GROUPS, SENSITIVE_PROXY, check_groups
from .reasons import sentence, wording_problems
from .shapley import additivity_error, shapley_values, stability, top_drivers, shap_cross_check

SAMPLE_COLUMNS = ["prediction_id", "employee_id", "department_id", "as_of_date", "risk_score_12m", "risk_band", "top_reason_1", "top_reason_2", "top_reason_3", "model_version", "model_note"]


def load_helpers():
    """Import the programme-supplied helper file (risk_bands, group_gap) from 05_Starter."""
    p = ROOT / "05_Starter" / "cortexhr_helpers.py"
    spec = importlib.util.spec_from_file_location("cortexhr_helpers", p); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def human_review_note() -> str:
    cfg = load_config()["explain"]
    return (cfg["human_review_text"] + " Bands: High = top 10% of scores, Medium = next 20%. Reasons are the factors that raised this score the most compared with an average employee; "
            "they show what moved the score, not certainty. Background factors that could stand in for protected traits (career history, education, location) are not shown as reasons but still count in the score.")


def select_reasons(phi: pd.DataFrame, n: int = 3, exclude=SENSITIVE_PROXY) -> pd.DataFrame:
    """The n groups that raised the score most (largest positive contribution), skipping sensitive-proxy groups.
    If fewer than n groups raised the score, the remainder is filled by largest absolute contribution (and then says 'lowered')."""
    names = np.array([c for c in phi.columns if c not in exclude]); a = phi[list(names)].to_numpy(); out = {}
    order = np.argsort(-a, axis=1, kind="stable")
    pos = (a > 0)
    for i in range(len(a)):
        picks = [j for j in order[i] if pos[i, j]][:n]
        if len(picks) < n:
            rest = [j for j in np.argsort(-np.abs(a[i]), kind="stable") if j not in picks]; picks += rest[: n - len(picks)]
        for k, j in enumerate(picks, 1):
            out.setdefault(f"driver_{k}", []).append(names[j]); out.setdefault(f"phi_{k}", []).append(a[i, j])
    return pd.DataFrame(out)


def build_background(n=200) -> pd.DataFrame:
    d = V.load_training(); allX = pd.concat([d["Xtr"], d["Xte"]]); return allX.sample(n, random_state=load_config()["seed"]).reset_index(drop=True)


def run_task14(write=True, n_permutations=None) -> dict:
    cfg = load_config(); P = n_permutations or cfg["explain"]["n_permutations"]; seed = cfg["seed"]
    model = joblib.load(MODELS / "attrition_v2.joblib"); scoring, X, join_rep = V.load_scoring(); check_groups(X.columns)
    p = model.predict_proba(X)[:, 1]; require_probability(p, "risk_score_12m")
    H = load_helpers(); band = H.risk_bands(p)
    bg = build_background(); predict = lambda df: model.predict_proba(df)[:, 1]
    half = P // 2
    phi_a, base_a, fx = shapley_values(predict, X, bg, GROUPS, half, seed); phi_b, base_b, _ = shapley_values(predict, X, bg, GROUPS, P - half, seed + 1)
    phi = (phi_a * half + phi_b * (P - half)) / P; base = (base_a * half + base_b * (P - half)) / P
    shap_check = shap_cross_check(predict, X, bg, n_rows=10, groups=GROUPS, seed=seed, max_background=50)
    out = {"phi_a": phi_a, "phi_b": phi_b, "model": model, "scoring": scoring, "X": X, "p": p, "band": np.asarray(band), "phi": phi, "base": base, "fx": fx, "background": bg, "join_report": join_rep,
           "additivity_max_error": additivity_error(phi, base, fx), "stability": stability(phi_a, phi_b), "n_permutations": P, "shap_cross_check": shap_check}
    td = select_reasons(phi, 3); rows = []; sc = scoring.reset_index(drop=True); Xr = X.reset_index(drop=True)
    for i in range(len(sc)):
        reasons = [sentence(td.loc[i, f"driver_{k}"], Xr.iloc[i], td.loc[i, f"phi_{k}"]) for k in (1, 2, 3)]
        rows.append({"prediction_id": f"AR-{i+1:04d}", "employee_id": sc.loc[i, "employee_id"], "department_id": sc.loc[i, "department_id"],
                     "as_of_date": pd.to_datetime(sc.loc[i, "observation_date"]).strftime("%Y-%m-%d"), "risk_score_12m": round(float(p[i]), 3), "risk_band": band[i],
                     "top_reason_1": reasons[0], "top_reason_2": reasons[1], "top_reason_3": reasons[2], "model_version": "attrition-v2", "model_note": human_review_note()})
    rep = pd.DataFrame(rows, columns=SAMPLE_COLUMNS); out["report"], out["drivers"] = rep, td
    out["checks"] = validate_report(rep, read_csv("sample_attrition_risk"))
    if write:
        save_table(rep, PREDICTIONS / "attrition_risk_2026.csv"); save_table(rep, task_dir(14) / "attrition_risk_2026.csv")
        sv = phi.copy(); sv.insert(0, "employee_id", sc["employee_id"].to_numpy()); sv["base_value"] = base.to_numpy(); sv["prediction"] = fx.to_numpy()
        save_table(sv, task_dir(14) / "group_contributions.csv")
        if shap_check is not None:
            save_table(shap_check["feature_values"].assign(employee_id=sc["employee_id"].head(len(shap_check["feature_values"])).to_numpy()), task_dir(14) / "shap_library_feature_cross_check.csv")
            save_table(shap_check["group_values"].assign(employee_id=sc["employee_id"].head(len(shap_check["group_values"])).to_numpy()), task_dir(14) / "shap_library_group_cross_check.csv")
        save_json({"created_utc": utc_now(), "n_permutations": P, "background_rows": len(bg), "additivity_max_error": out["additivity_max_error"], "stability": out["stability"],
                   "shap_library_cross_check": None if shap_check is None else {"shap_version": shap_check["shap_version"], "rows": len(shap_check["predictions"]), "background_rows": shap_check["background_rows"], "max_evals": shap_check["max_evals"], "additivity_max_error": shap_check["additivity_max_error"]},
                   "checks": out["checks"]}, task_dir(14) / "explanation_run.json")
    return out


def validate_report(rep: pd.DataFrame, sample: pd.DataFrame) -> list[dict]:
    """Every check returns {'check', 'passed', 'detail'}. Used by the notebook, the export step and the tests."""
    res = []
    def add(name, ok, detail=""): res.append({"check": name, "passed": bool(ok), "detail": str(detail)})
    add("430 rows", len(rep) == 430, len(rep))
    add("columns equal sample (names and order)", list(rep.columns) == list(sample.columns), list(rep.columns) if list(rep.columns) != list(sample.columns) else "")
    add("employee ids unique", rep.employee_id.is_unique)
    add("employee ids and order equal the scoring file", list(rep.employee_id) == list(read_csv("attrition_scoring_2026").employee_id))
    add("risk score in [0,1], no NaN", bool(rep.risk_score_12m.between(0, 1).all()) and not rep.risk_score_12m.isna().any())
    add("risk band values valid", set(rep.risk_band) <= {"Low", "Medium", "High"}, rep.risk_band.value_counts().to_dict())
    add("band counts 43/86/301 (top 10%, next 20%)", rep.risk_band.value_counts().to_dict() == {"Low": 301, "Medium": 86, "High": 43})
    add("band is monotone in score", bool(rep.groupby("risk_band").risk_score_12m.min().loc[["Low", "Medium", "High"]].is_monotonic_increasing))
    add("three non-empty reasons per row", bool(rep[["top_reason_1", "top_reason_2", "top_reason_3"]].apply(lambda c: c.astype(str).str.len() > 20).all().all()))
    add("the three reasons are different", bool((rep.top_reason_1 != rep.top_reason_2).all() and (rep.top_reason_2 != rep.top_reason_3).all() and (rep.top_reason_1 != rep.top_reason_3).all()))
    add("every row says human review is required", bool(rep.model_note.str.contains("Human review required").all()))
    add("model_version is attrition-v2", set(rep.model_version) == {"attrition-v2"})
    add("same dtype kinds as sample", all((rep[c].dtype.kind in "fi") == (sample[c].dtype.kind in "fi") for c in sample.columns))
    bad = [(i, w) for i, t in enumerate(rep[["top_reason_1", "top_reason_2", "top_reason_3"]].astype(str).agg(" ".join, axis=1)) for w in wording_problems(t)]
    add("no banned wording (certainty, blame, protected traits)", not bad, bad[:3])
    return res
