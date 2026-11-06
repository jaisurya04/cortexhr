"""Weight sensitivity, leave-one-job-out weight selection and misranking analysis."""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
from .scoring import ResumeRanker
from .metrics import attach_labels, evaluate


def component_frame(ranker: ResumeRanker, labels: pd.DataFrame) -> pd.DataFrame:
    """Score every labelled pair once; components are reused to test any weighting without refitting."""
    pairs = labels[["resume_id", "job_id"]].drop_duplicates()
    rows = pd.DataFrame([ranker.score_pair(r, j) for r, j in zip(pairs["resume_id"], pairs["job_id"])])
    return rows.merge(labels[["resume_id", "job_id", "human_suitability_score", "shortlist"]], on=["resume_id", "job_id"], validate="one_to_one")


def reweight(comp: pd.DataFrame, w: dict) -> pd.DataFrame:
    d = comp.copy()
    d["score"] = w["tfidf"] * d["tfidf_scaled"] + w["required_skill"] * d["required_coverage"] + w["nice_skill"] * d["nice_coverage"] + w["experience"] * d["experience_score"]
    out = []
    for _, g in d.groupby("job_id"):
        out.append(ResumeRanker.rank_frame(g))
    o = pd.concat(out, ignore_index=True); o["rel"] = o["human_suitability_score"].astype(int); o["relevant"] = (o["shortlist"] == "Y").astype(int)
    return o


def weight_grid(step=0.1):
    vals = np.round(np.arange(0, 1 + 1e-9, step), 2); out = []
    for a, b, c in itertools.product(vals, repeat=3):
        d = round(1 - a - b - c, 2)
        if d >= 0: out.append({"tfidf": float(a), "required_skill": float(b), "nice_skill": float(c), "experience": float(d)})
    return out


def sensitivity(comp: pd.DataFrame, grid: list[dict], k=5, metric="ndcg@5") -> pd.DataFrame:
    rows = []
    for i, w in enumerate(grid):
        _, m = evaluate(reweight(comp, w), k); rows.append({"experiment_id": f"W{i:03d}", **w, **m})
    return pd.DataFrame(rows)


def leave_one_job_out(comp: pd.DataFrame, grid: list[dict], k=5, metric="ndcg@5") -> pd.DataFrame:
    """Choose weights on 9 jobs, report quality on the held-out 10th job (honest estimate, no tuning on the test job)."""
    jobs = sorted(comp["job_id"].unique()); rows = []
    cache = {i: {j: evaluate(reweight(comp[comp.job_id == j], w), k)[0].iloc[0] for j in jobs} for i, w in enumerate(grid)}
    for held in jobs:
        best = max(range(len(grid)), key=lambda i: np.mean([cache[i][j][metric] for j in jobs if j != held]))
        rows.append({"held_out_job": held, "chosen_weights": grid[best], f"train_mean_{metric}": float(np.mean([cache[best][j][metric] for j in jobs if j != held])),
                     f"held_out_{metric}": float(cache[best][held][metric]), "held_out_precision@5": float(cache[best][held]["precision@5"])})
    return pd.DataFrame(rows)


def find_misranks(ev: pd.DataFrame, k=5, n=5) -> pd.DataFrame:
    """Genuine misranks, measured by pairwise inversions against human judgement (humans' graded 1..5 score).
    under-ranked: a human-relevant candidate (score >= 4) that has candidates the humans rated >= 2 points lower ranked above it.
    over-ranked : a candidate the humans rated <= 2 that has candidates rated >= 2 points higher ranked below it.
    severity = number of such inversions. A relevant candidate at rank 9 is NOT a misrank when 15 candidates are relevant."""
    rows = []
    for job, g in ev.groupby("job_id"):
        g = g.sort_values("rank")
        for _, r in g.iterrows():
            above, below = g[g["rank"] < r["rank"]], g[g["rank"] > r["rank"]]
            if r["rel"] >= 4:
                sev = int((above["rel"] <= r["rel"] - 2).sum())
                if sev: rows.append({**r.to_dict(), "misrank_type": "relevant candidate ranked below clearly weaker candidates (under-ranked)", "severity": sev})
            if r["rel"] <= 2:
                sev = int((below["rel"] >= r["rel"] + 2).sum())
                if sev: rows.append({**r.to_dict(), "misrank_type": "weak candidate ranked above clearly stronger candidates (over-ranked)", "severity": sev})
    d = pd.DataFrame(rows)
    if not len(d): return d
    d = d.sort_values(["severity", "resume_id"], ascending=[False, True]).reset_index(drop=True)
    return d.head(n)


def diagnose(row: pd.Series) -> tuple[str, str, str]:
    """Rule-based root cause from the component scores of ONE misranked pair (no guessing beyond the numbers)."""
    miss = [s for s in str(row["missing_required"]).split("; ") if s]
    if "under-ranked" in row["misrank_type"]:
        if miss:
            return (f"Human scored it {int(row['rel'])}/5 although {len(miss)} required skill(s) are not found in the text ({', '.join(miss[:3])}).",
                    "Human reviewers valued overall profile more than strict skill coverage; required_coverage pushed the score down.",
                    "Add related-skill credit (e.g. a skill graph) or a softer coverage for near-miss skills; review the label with the recruiter.")
        return (f"All required skills matched but total score {row['score']:.2f} ranked it {int(row['rank'])}.",
                "Other candidates scored higher on TF-IDF or experience; candidates with identical scores (such as twin resumes) are ordered by a neutral hash.",
                "Use more informative tie-breaks (relevant years, nice-to-have skills) and review the experience cap.")
    return (f"Humans rated it {int(row['rel'])}/5, yet it outranks candidates they rated at least 2 points higher. Required skill coverage is {row['required_coverage']:.0%}, text similarity {row['tfidf_scaled']:.2f}, experience fit {row['experience_score']:.2f}.",
            f"All candidates around this rank miss most required skills, so the order between them is decided by weak signals (text similarity and experience); the score difference is small (model score {row['score']:.2f}).",
            "Low priority: the error sits among low-ranked candidates and cannot reach the top-5. Collect more labels, then test a skill graph or an evidence-of-use feature.")
