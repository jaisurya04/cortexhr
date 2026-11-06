"""Task 12: counterfactual twin-resume audit. A twin differs from its original ONLY in name/e-mail (gender-coded) and college-tier text."""
from __future__ import annotations
import numpy as np
import pandas as pd
from ..common.config import load_config
from ..common.paths import VAULT


def vault_resume_attributes() -> pd.DataFrame | None:
    """Hidden protected attributes (restricted file, measurement only). Returns None if the file is not present."""
    p = VAULT / "unlock_CX-12_resume_protected_attrs.csv"
    return pd.read_csv(p).set_index("resume_id") if p.exists() else None


def validate_pairs(pairs: pd.DataFrame, resumes: pd.DataFrame) -> pd.DataFrame:
    """For every pair: do both resumes exist, and do ALL columns other than the expected ones really match?"""
    r = resumes.set_index("resume_id"); rows = []
    allowed = {"candidate_name", "email", "resume_text", "college_tier", "resume_id"}
    for _, p in pairs.iterrows():
        o, t = r.loc[p.resume_id_original], r.loc[p.resume_id_twin]
        diff = [c for c in r.columns if str(o[c]) != str(t[c])]
        rows.append({"pair_id": p.pair_id, "attribute_changed": p.attribute_changed, "columns_that_differ": ", ".join(diff),
                     "unexpected_differences": ", ".join(c for c in diff if c not in allowed), "skills_identical_flag": p.skills_identical,
                     "skills_actually_identical": o["skills"] == t["skills"], "experience_identical": o["total_experience_years"] == t["total_experience_years"],
                     "expected_score_difference": p.expected_score_difference})
    return pd.DataFrame(rows)


def twin_gaps(ranker, pairs: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Score both twins on every job they are labelled for. gap = twin - original (positive means the twin scored higher)."""
    rows = []
    pool = {j: list(g.resume_id) for j, g in labels.groupby("job_id")}; cache = {}
    for _, p in pairs.iterrows():
        o, t = p.resume_id_original, p.resume_id_twin
        for job in sorted(set(labels[labels.resume_id == o].job_id) & set(labels[labels.resume_id == t].job_id)):
            if job not in cache: cache[job] = ranker.score_job(job, pool[job]).set_index("resume_id")
            sc = cache[job]; so, st = float(sc.loc[o, "score"]), float(sc.loc[t, "score"])
            rows.append({"pair_id": p.pair_id, "job_id": job, "score_original": so, "score_twin": st, "gap": st - so, "abs_gap": abs(st - so),
                         "relative_gap_pct": (st - so) / so * 100 if so else np.nan, "rank_original": int(sc.loc[o, "rank"]), "rank_twin": int(sc.loc[t, "rank"]),
                         "twin_ranked_higher": bool(sc.loc[t, "rank"] < sc.loc[o, "rank"]), "attribute_changed": p.attribute_changed})
    return pd.DataFrame(rows)


def gap_summary(g: pd.DataFrame, tol: float | None = None) -> dict:
    tol = load_config()["fairness"]["tolerance"] if tol is None else tol
    return {"comparisons": len(g), "pairs": g.pair_id.nunique(), "mean_abs_gap": g.abs_gap.mean(), "median_abs_gap": g.abs_gap.median(), "max_abs_gap": g.abs_gap.max(),
            "mean_signed_gap(twin-original)": g.gap.mean(), "comparisons_with_nonzero_gap": int((g.abs_gap > 1e-9).sum()), "comparisons_above_tolerance": int((g.abs_gap > tol).sum()),
            "mean_abs_relative_gap_pct": g.relative_gap_pct.abs().mean(), "share_twin_ranked_higher": g.twin_ranked_higher.mean(),
            "mean_abs_rank_difference": (g.rank_original - g.rank_twin).abs().mean()}


def attach_protected(g: pd.DataFrame, pairs: pd.DataFrame, attrs: pd.DataFrame | None) -> pd.DataFrame:
    """Add the hidden attributes of the original and the twin (gender, college tier) for interpretation. Never used for scoring."""
    if attrs is None: return g
    m = pairs.set_index("pair_id")
    d = g.copy()
    d["gender_original"] = d.pair_id.map(m.resume_id_original).map(attrs.inferred_gender); d["gender_twin"] = d.pair_id.map(m.resume_id_twin).map(attrs.inferred_gender)
    d["tier_original"] = d.pair_id.map(m.resume_id_original).map(attrs.college_tier); d["tier_twin"] = d.pair_id.map(m.resume_id_twin).map(attrs.college_tier)
    d["gender_changed"] = d.gender_original != d.gender_twin; d["tier_changed"] = d.tier_original != d.tier_twin
    d["favoured_group"] = np.where(d.gap > 1e-9, d.gender_twin.astype(str) + "/" + d.tier_twin.astype(str) + " twin", np.where(d.gap < -1e-9, d.gender_original.astype(str) + "/" + d.tier_original.astype(str) + " original", "none"))
    return d
