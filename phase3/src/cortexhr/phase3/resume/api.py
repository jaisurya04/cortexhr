"""Resume-ranking API contract, implemented as a pure function so it can be tested without a web server.

POST /v1/resumes/rank
  request : {"job_id": "J01"}  OR  {"job": {"title", "required_skills", "min_experience_years", "description_text"?, "nice_to_have_skills"?}},
            "resume_ids": [..]?  (default: every resume the ranker knows),  "top_k": 1..20 (default 5)
  response: {"engine_version", "job_id", "top_k", "candidates": [{rank, resume_id, score, score_breakdown, matched_required_skills,
             missing_required_skills, experience_evidence, recommendation, human_review}], "used_inputs", "never_used", "notice"}
  errors  : 422 with {"error": "validation", "details": [..]}; 404 for unknown job_id / resume_id.
Candidate names, gender, age, college tier, e-mail and city are neither accepted nor returned.
"""
from __future__ import annotations
from .scoring import ResumeRanker, recommendation

FORBIDDEN_FIELDS = ("candidate_name", "name", "gender", "age", "college_tier", "email", "city", "photo")


class ApiError(Exception):
    def __init__(self, status: int, error: str, details=None):
        super().__init__(error); self.status, self.error, self.details = status, error, details or []

    def body(self): return {"error": self.error, "details": self.details}


def validate_request(payload: dict) -> dict:
    d = []
    if not isinstance(payload, dict): raise ApiError(422, "validation", ["body must be a JSON object"])
    bad = [k for k in payload if k in FORBIDDEN_FIELDS]
    if bad: d.append(f"fields not accepted (never used in ranking): {bad}")
    unknown = [k for k in payload if k not in ("job_id", "job", "resume_ids", "top_k") + FORBIDDEN_FIELDS]
    if unknown: d.append(f"unknown fields: {unknown}")
    if ("job_id" in payload) == ("job" in payload): d.append("provide exactly one of job_id or job")
    k = payload.get("top_k", 5)
    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= 20: d.append("top_k must be an integer from 1 to 20")
    if "resume_ids" in payload and (not isinstance(payload["resume_ids"], list) or not payload["resume_ids"] or not all(isinstance(x, str) for x in payload["resume_ids"])):
        d.append("resume_ids must be a non-empty list of strings")
    job = payload.get("job")
    if job is not None:
        if not isinstance(job, dict): d.append("job must be an object")
        else:
            for f in ("title", "required_skills", "min_experience_years"):
                if f not in job: d.append(f"job.{f} is required")
            if "min_experience_years" in job and (not isinstance(job["min_experience_years"], (int, float)) or not 0 <= job["min_experience_years"] <= 40): d.append("job.min_experience_years must be a number from 0 to 40")
            if "required_skills" in job and (not isinstance(job["required_skills"], str) or not job["required_skills"].strip()): d.append("job.required_skills must be a non-empty ';'-separated string")
    if d: raise ApiError(422, "validation", d)
    return {"top_k": k}


def handle_rank_request(payload: dict, ranker: ResumeRanker) -> dict:
    opts = validate_request(payload); k = opts["top_k"]
    if "job_id" in payload:
        job_id = payload["job_id"]
        if job_id not in ranker.jobs.index: raise ApiError(404, "unknown job_id", [str(job_id)])
    else:
        j = payload["job"]; job_id = ranker.add_job(f"CUSTOM-{abs(hash(str(sorted(j.items())))) % 10**6:06d}", j["title"], j["required_skills"], j["min_experience_years"],
                                                   j.get("description_text", ""), j.get("nice_to_have_skills", ""))
    ids = payload.get("resume_ids")
    if ids:
        unknown = [r for r in ids if r not in ranker._rid_idx]
        if unknown: raise ApiError(404, "unknown resume_id", unknown[:5])
    top = ranker.top_k(job_id, k, ids)
    cands = [{"rank": int(r["rank"]), "resume_id": r["resume_id"], "score": round(float(r["score"]), 4),
              "score_breakdown": {"text_similarity": round(float(r["tfidf_scaled"]), 4), "required_skill_coverage": round(float(r["required_coverage"]), 4),
                                  "nice_to_have_coverage": round(float(r["nice_coverage"]), 4), "experience_fit": round(float(r["experience_score"]), 4)},
              "matched_required_skills": [s for s in r["matched_required"].split("; ") if s], "missing_required_skills": [s for s in r["missing_required"].split("; ") if s],
              "experience_evidence": {"total_years": r["total_years"], "years_in_relevant_roles": r["relevant_years"], "job_minimum_years": r["min_years"]},
              "recommendation": r["recommendation"], "human_review": r["human_review"]} for _, r in top.iterrows()]
    u = ranker.used_inputs()
    return {"engine_version": u["version"], "job_id": job_id, "top_k": k, "candidates": cands, "used_inputs": u["score_inputs"], "never_used": u["never_used"],
            "weights": u["weights"], "notice": "Decision support only. A human recruiter must review every candidate."}
