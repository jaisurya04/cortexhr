"""Re-run the two Phase 3 fairness checks on a schedule: the twin-resume gap and the pay-gap test.

Both checks use only data already in the repository except the *gender* pay gap, which needs the restricted CX-15 file in a local vault.
A passing check proves that this particular test found no gap - not that the system is fair.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..core import legacy
from ..core.settings import PHASE3, Settings
from . import status

GAP_BINS = [-1, 0, 6, 12, 24, 10 ** 4]
GAP_LABELS = ["none", "1-6 months", "7-12 months", "13-24 months", "over 24 months"]


def twin_resume_check(settings: Settings) -> dict:
    """Score every twin pair with the masked resume ranker (v2) and report the largest score gap."""
    legacy.bootstrap()
    from cortexhr.phase3.fairness.twins import gap_summary, twin_gaps
    d = settings.data_dir
    ranker = joblib.load(PHASE3 / "artifacts" / "models" / "resume_ranker_v2.joblib")
    pairs, labels = pd.read_csv(d / "resume_bias_pairs.csv"), pd.read_csv(d / "resume_job_labels.csv")
    g = twin_gaps(ranker, pairs, labels)
    s = gap_summary(g, status.TWIN_TOLERANCE)
    st = status.twin_status(float(s["max_abs_gap"]), int(s["comparisons_with_nonzero_gap"]))
    return {"status": st, "summary": {k: (float(v) if isinstance(v, (np.floating, float)) else int(v)) for k, v in s.items()},
            "tolerance": status.TWIN_TOLERANCE,
            "limit": "The masked screener removes name, e-mail, city and college-tier text, so a zero gap is expected by construction. "
                     "It does not test other proxies (for example hobbies or employer names)."}


def pay_gap_check(settings: Settings) -> dict:
    """Mean (predicted pay / band reference) by career-gap group, and by gender when the restricted file is available.

    Same method as Phase 3 Task 15: every hire gets an *out-of-sample* prediction (5-fold out-of-fold for the training part, the saved
    model for the hold-out part), so the model is judged on offers it did not learn from.
    """
    legacy.bootstrap()
    from cortexhr.phase3.salary import model as S
    bundle = joblib.load(PHASE3 / "artifacts" / "models" / "salary_band_v1.joblib")
    df = S.load_salary()
    tr, te = S.split(df)
    y_tr = tr[S.TARGET].to_numpy(float)
    oof_b = S.oof_predict(lambda: S.ridge(S.JOB_FEATURES), S.JOB_FEATURES, tr)                       # rupees
    oof_a = S.oof_predict(lambda: S.ridge(S.WITH_GAP_FEATURES), S.WITH_GAP_FEATURES, tr)
    model_a = S.ridge(S.WITH_GAP_FEATURES).fit(tr[S.WITH_GAP_FEATURES], np.log(y_tr))
    full = pd.concat([tr, te], ignore_index=True)
    norm = full["band_mid_2026"] * full["market_index"]
    full["ratio_actual"] = full[S.TARGET] / norm
    full["ratio_model"] = np.r_[oof_b, np.exp(bundle["model"].predict(te[bundle["features"]]))] / norm
    full["ratio_model_with_gap"] = np.r_[oof_a, np.exp(model_a.predict(te[S.WITH_GAP_FEATURES]))] / norm
    full["gap_bin"] = pd.cut(full.career_gap_months, GAP_BINS, labels=GAP_LABELS)
    out = {"model_version": bundle["version"], "n_hires": int(len(full)), "groups": {}}
    parts = []
    for col, label in (("gap_bin", "career_gap"), ("gender", "gender")):
        if col == "gender":
            vault = _vault_gender(settings)
            if vault is None:
                out["groups"][label] = {"status": status.SKIPPED, "reason": "restricted CX-15 file not found (set CORTEXHR_VAULT_DIR); gender pay gap NOT checked"}
                continue
            full = full.join(vault, on="hire_id")
        sizes = full.groupby(col, observed=True).size()
        keep = list(sizes[sizes >= status.MIN_GROUP_SIZE].index)
        if len(keep) < 2:                                     # one group cannot show a gap: never report that as OK
            out["groups"][label] = {"status": status.WATCH, "reason": f"fewer than 2 groups with at least {status.MIN_GROUP_SIZE} people; gap cannot be judged"}
            parts.append(status.WATCH)
            continue
        m = full[full[col].isin(keep)].groupby(col, observed=True)[["ratio_actual", "ratio_model", "ratio_model_with_gap"]].mean()
        gap = lambda c: float(m[c].max() - m[c].min())
        st = status.gap_status(gap("ratio_model"))
        parts.append(st)
        out["groups"][label] = {"status": st, "gap_history": gap("ratio_actual"), "gap_model": gap("ratio_model"), "gap_model_with_career_gap": gap("ratio_model_with_gap"),
                                "means": m.round(4).reset_index().astype(str).to_dict("records"), "groups_compared": [str(k) for k in keep],
                                "groups_too_small": [str(k) for k in sizes.index if k not in keep]}
    out["status"] = status.worst(parts) if parts else status.WATCH
    out["limit"] = ("Compares group means of predicted pay relative to the role band; it does not control for every legitimate factor and "
                    "small groups are not compared. A small gap does not prove the offers are fair.")
    return out


def _vault_gender(settings: Settings) -> pd.DataFrame | None:
    if not settings.vault_dir:
        return None
    p = Path(settings.vault_dir) / "unlock_CX-15_protected_attributes.csv"
    if not p.exists():
        return None
    v = pd.read_csv(p)
    return v[v.record_type == "salary_hire"].drop_duplicates("person_ref").set_index("person_ref")[["gender"]]
