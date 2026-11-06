"""Ranking metrics against human judgement. `rel` = graded human score (1..5); relevant = shortlist 'Y' (score >= 4)."""
from __future__ import annotations
import numpy as np
import pandas as pd


def _topk(df: pd.DataFrame, k: int) -> pd.DataFrame:
    return df.sort_values("rank").head(k)


def precision_at_k(df, k=5): return float((_topk(df, k)["relevant"] == 1).sum() / k)
def recall_at_k(df, k=5):
    n = int(df["relevant"].sum()); return float((_topk(df, k)["relevant"] == 1).sum() / n) if n else np.nan
def hit_rate_at_k(df, k=5): return float((_topk(df, k)["relevant"] == 1).any())
def ndcg_at_k(df, k=5):
    gain = lambda r: (2.0 ** r - 1.0)
    d = _topk(df, k); dcg = sum(gain(r) / np.log2(i + 2) for i, r in enumerate(d["rel"]))
    ideal = sorted(df["rel"], reverse=True)[:k]; idcg = sum(gain(r) / np.log2(i + 2) for i, r in enumerate(ideal))
    return float(dcg / idcg) if idcg > 0 else np.nan
def mrr(df):
    hit = df.sort_values("rank")["relevant"].to_numpy(); idx = np.flatnonzero(hit == 1)
    return float(1.0 / (idx[0] + 1)) if idx.size else 0.0
def average_precision(df):
    hit = df.sort_values("rank")["relevant"].to_numpy(); n = hit.sum()
    if n == 0: return np.nan
    return float(np.sum(np.cumsum(hit)[hit == 1] / (np.flatnonzero(hit == 1) + 1)) / n)
def r_precision(df):
    n = int(df["relevant"].sum()); return float((_topk(df, n)["relevant"] == 1).sum() / n) if n else np.nan


def attach_labels(scored: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    m = scored.merge(labels[["resume_id", "job_id", "human_suitability_score", "shortlist"]], on=["resume_id", "job_id"], how="inner", validate="one_to_one")
    m["rel"] = m["human_suitability_score"].astype(int); m["relevant"] = (m["shortlist"] == "Y").astype(int)
    return m


def evaluate(labelled: pd.DataFrame, k: int = 5) -> tuple[pd.DataFrame, dict]:
    """Per-job table and macro averages. `labelled` needs rank, rel, relevant per job."""
    rows = []
    for job, g in labelled.groupby("job_id"):
        rows.append({"job_id": job, "candidates": len(g), "relevant": int(g["relevant"].sum()), f"precision@{k}": precision_at_k(g, k),
                     f"recall@{k}": recall_at_k(g, k), f"hit_rate@{k}": hit_rate_at_k(g, k), f"ndcg@{k}": ndcg_at_k(g, k),
                     "mrr": mrr(g), "average_precision": average_precision(g), "r_precision": r_precision(g)})
    per = pd.DataFrame(rows); num = per.drop(columns=["job_id", "candidates", "relevant"])
    return per, num.mean().to_dict()


def random_baseline(labelled: pd.DataFrame, k=5, n_shuffles=300, seed=42) -> dict:
    """Expected metrics of a random ordering (the floor any ranker must beat), averaged over shuffles."""
    rng = np.random.default_rng(seed); acc = []
    for _ in range(n_shuffles):
        d = labelled.copy(); d["rank"] = d.groupby("job_id")["resume_id"].transform(lambda s: rng.permutation(len(s)) + 1)
        acc.append(evaluate(d, k)[1])
    return pd.DataFrame(acc).mean().to_dict()
