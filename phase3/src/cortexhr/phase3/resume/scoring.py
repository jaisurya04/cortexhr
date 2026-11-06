"""Resume ranking engine: TF-IDF cosine + skill overlap + experience, never name / gender / age / college tier.

Two versions are kept on purpose (Task 12 compares them):
  mask_identifiers=False  -> "v1": scores the resume_text exactly as given (structured protected columns are NOT used,
                              but the free text still contains the name, e-mail and the "Tier N institution" line).
  mask_identifiers=True   -> "v2": name, e-mail, city and college-tier phrases are removed before scoring.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from . import text as T
from .skills import SkillExtractor, build_vocabulary, coverage, job_skill_sets, split_skill_list
from .experience import extract_experience, relevant_years, experience_score
from ..common.config import load_config
from ..common.validation import require_columns, require_unique

ENGINE_VERSION = "resume-ranker-v2-masked"
FORBIDDEN_SCORE_INPUTS = ("candidate_name", "email", "gender", "age", "college_tier", "city", "target_job_family")


def recommendation(score: float) -> str:
    if score >= 0.75: return "Strong match: put on the recruiter's shortlist for human review."
    if score >= 0.50: return "Possible match: recruiter should read the resume before deciding."
    return "Weak match on the stated requirements: low priority, a human must still decide."


@dataclass
class ResumeRanker:
    weights: dict = field(default_factory=lambda: dict(load_config()["resume"]["weights"]))
    mask: bool = True
    ngram_range: tuple = (1, 2)
    mask_parts: tuple = ("name", "email", "city", "tier")      # which identifier types are removed when mask=True (ablations in Task 12)

    # ------------------------------------------------------------------ fit
    def fit(self, resumes: pd.DataFrame, jobs: pd.DataFrame) -> "ResumeRanker":
        require_columns(resumes, ["resume_id", "resume_text"], "resumes"); require_unique(resumes, "resume_id", "resumes")
        require_columns(jobs, ["job_id", "job_title", "required_skills", "min_experience_years", "description_text"], "jobs")
        require_unique(jobs, "job_id", "jobs")
        w = self.weights
        if abs(sum(w.values()) - 1.0) > 1e-9 or any(v < 0 for v in w.values()):
            raise ValueError(f"weights must be >= 0 and sum to 1, got {w}")
        self.jobs = jobs.set_index("job_id", drop=False)
        self.extractor = SkillExtractor(build_vocabulary(jobs, resumes))
        self.phrases = [s for s in self.extractor.vocab if " " in s or "-" in s]
        self.resumes = resumes.set_index("resume_id", drop=False)
        self._prep = {rid: self._prepare(row) for rid, row in self.resumes.iterrows()}
        job_docs = {jid: self._job_doc(r) for jid, r in self.jobs.iterrows()}
        self.vec = TfidfVectorizer(ngram_range=self.ngram_range, sublinear_tf=True, min_df=1, token_pattern=r"[a-z0-9_+#./-]+", lowercase=False)
        self.vec.fit([p["doc"] for p in self._prep.values()] + list(job_docs.values()))
        self._R = self.vec.transform([self._prep[r]["doc"] for r in self.resumes.index])
        self._rid_idx = {r: i for i, r in enumerate(self.resumes.index)}
        self._J = {jid: self.vec.transform([d]) for jid, d in job_docs.items()}
        # per-job min/max of the cosine over the fit corpus (scales cosine to 0..1 comparably across jobs)
        self._cos_range = {}
        for jid, jv in self._J.items():
            c = (self._R @ jv.T).toarray().ravel(); self._cos_range[jid] = (float(c.min()), float(c.max()))
        return self

    def _prepare(self, row) -> dict:
        txt = row["resume_text"] if not T.is_blank(row["resume_text"]) else ""
        if self.mask:
            pp = self.mask_parts; a = (row.get("candidate_name") if "name" in pp else None, row.get("email") if "email" in pp else None, row.get("city") if "city" in pp else None)
            clean = T.for_skill_extraction(txt, *a, "tier" in pp, "email" in pp)
            doc_src = T.mask_identifiers(txt, *a, "tier" in pp, "email" in pp)
        else:
            clean, doc_src = txt, txt
        return {"doc": T.prepare_document(doc_src, self.phrases), "skills": self.extractor.extract(clean),
                "exp": extract_experience(clean if self.mask else txt)}

    def _job_doc(self, job) -> str:
        parts = [job["job_title"], job["required_skills"], job.get("nice_to_have_skills", ""), job.get("education_preference", ""), job["description_text"]]
        return T.prepare_document(" . ".join(str(p) for p in parts if not T.is_blank(p)), self.phrases)

    def add_job(self, job_id: str, title: str, required_skills: str, min_experience_years: float, description_text: str = "",
                nice_to_have_skills: str = "", education_preference: str = "") -> str:
        """Register a job that is not in job_descriptions.csv (used by the API for ad-hoc job texts)."""
        row = pd.Series({"job_id": job_id, "job_title": title, "required_skills": required_skills, "nice_to_have_skills": nice_to_have_skills,
                         "min_experience_years": float(min_experience_years), "education_preference": education_preference, "description_text": description_text or title})
        self.jobs.loc[job_id] = row; jv = self.vec.transform([self._job_doc(row)]); self._J[job_id] = jv
        c = (self._R @ jv.T).toarray().ravel(); self._cos_range[job_id] = (float(c.min()), float(c.max())); return job_id

    # ---------------------------------------------------------------- score
    def score_pair(self, resume_id: str, job_id: str) -> dict:
        if resume_id not in self._rid_idx: raise KeyError(f"unknown resume_id {resume_id}")
        if job_id not in self.jobs.index: raise KeyError(f"unknown job_id {job_id}")
        job = self.jobs.loc[job_id]; p = self._prep[resume_id]
        cos = float((self._R[self._rid_idx[resume_id]] @ self._J[job_id].T).toarray()[0, 0])
        lo, hi = self._cos_range[job_id]; cos_n = (cos - lo) / (hi - lo) if hi > lo else 0.0
        req, nice = job_skill_sets(job, self.extractor); cov = coverage(p["skills"], req, nice)
        e = p["exp"]; rel = relevant_years(e["roles"], job["job_title"]); xs = experience_score(e["total_years"], rel, job["min_experience_years"])
        w = self.weights
        total = w["tfidf"] * cos_n + w["required_skill"] * cov["required_coverage"] + w["nice_skill"] * cov["nice_coverage"] + w["experience"] * xs
        return {"resume_id": resume_id, "job_id": job_id, "score": round(float(total), 10), "tfidf_cosine": cos, "tfidf_scaled": cos_n,
                "required_coverage": cov["required_coverage"], "nice_coverage": cov["nice_coverage"], "experience_score": xs,
                "total_years": e["total_years"], "relevant_years": rel, "min_years": int(job["min_experience_years"]),
                "matched_required": "; ".join(cov["matched_required"]), "missing_required": "; ".join(cov["missing_required"]),
                "matched_nice": "; ".join(cov["matched_nice"]), "synonym_matches": "; ".join(cov["synonym_matches"]),
                "experience_flags": "; ".join(e["flags"])}

    def score_job(self, job_id: str, resume_ids=None) -> pd.DataFrame:
        ids = list(self.resumes.index) if resume_ids is None else list(resume_ids)
        df = pd.DataFrame([self.score_pair(r, job_id) for r in ids])
        return self.rank_frame(df)

    @staticmethod
    def _tie_key(job_id: str, resume_id: str) -> str:
        """Neutral deterministic tie-break: a hash of (job, resume). It is stable between runs but unrelated to id order,
        so identical scores (e.g. twin resumes) are not always ordered original-first."""
        return hashlib.md5(f"{job_id}|{resume_id}".encode()).hexdigest()

    @staticmethod
    def rank_frame(df: pd.DataFrame) -> pd.DataFrame:
        """Deterministic order: score (rounded to 9 d.p.) descending, ties broken by the neutral hash above."""
        d = df.assign(_s=df["score"].round(9), _t=[ResumeRanker._tie_key(j, r) for j, r in zip(df["job_id"], df["resume_id"])])
        d = d.sort_values(["_s", "_t"], ascending=[False, True]).drop(columns=["_s", "_t"]).reset_index(drop=True)
        d["rank"] = np.arange(1, len(d) + 1); return d

    def top_k(self, job_id: str, k: int = 5, resume_ids=None) -> pd.DataFrame:
        d = self.score_job(job_id, resume_ids).head(k).copy()
        d["recommendation"] = d["score"].map(recommendation)
        d["human_review"] = "Required: ranking is decision support, not a hiring decision."
        return d

    def score_all(self, pairs: pd.DataFrame) -> pd.DataFrame:
        """Score every (resume_id, job_id) pair in `pairs` and add a within-job rank over the pairs supplied."""
        rows = pd.DataFrame([self.score_pair(r, j) for r, j in zip(pairs["resume_id"], pairs["job_id"])])
        return pd.concat([self.rank_frame(g) for _, g in rows.groupby("job_id")], ignore_index=True)

    def used_inputs(self) -> dict:
        return {"score_inputs": ["resume_text (identifiers masked)" if self.mask else "resume_text (as given)", "job_title", "required_skills",
                                 "nice_to_have_skills", "education_preference", "description_text", "min_experience_years"],
                "never_used": list(FORBIDDEN_SCORE_INPUTS), "weights": self.weights, "version": ENGINE_VERSION if self.mask else "resume-ranker-v1-unmasked"}
