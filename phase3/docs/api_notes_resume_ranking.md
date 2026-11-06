# API notes: resume ranking endpoint

`POST /v1/resumes/rank` (contract implemented and tested in `src/cortexhr/phase3/resume/api.py`).

**Request**
```json
{"job_id": "J05", "top_k": 5, "resume_ids": ["RES-0001", "RES-0002"]}
```
or, for a new job that is not in the file,
```json
{"job": {"title": "Data Analyst", "required_skills": "Python; SQL", "min_experience_years": 2, "nice_to_have_skills": "Tableau", "description_text": "..."}, "top_k": 5}
```
Rules: exactly one of `job_id` or `job`; `top_k` an integer 1 to 20 (default 5); `resume_ids` optional, non-empty list of strings; `job.min_experience_years` 0 to 40. Fields `candidate_name`, `gender`, `age`, `college_tier`, `email`, `city` are **rejected** with 422 because they are never used.

**Response 200**
```json
{"engine_version": "resume-ranker-v2-masked", "job_id": "J05", "top_k": 5,
 "candidates": [{"rank": 1, "resume_id": "RES-0042", "score": 0.91,
   "score_breakdown": {"text_similarity": 0.8, "required_skill_coverage": 1.0, "nice_to_have_coverage": 0.5, "experience_fit": 1.0},
   "matched_required_skills": ["..."], "missing_required_skills": [],
   "experience_evidence": {"total_years": 4.2, "years_in_relevant_roles": 3.0, "job_minimum_years": 3},
   "recommendation": "Strong match: put on the recruiter's shortlist for human review.", "human_review": "Required: ranking is decision support, not a hiring decision."}],
 "used_inputs": ["..."], "never_used": ["candidate_name", "email", "gender", "age", "college_tier", "city", "target_job_family"],
 "weights": {"tfidf": 0.3, "required_skill": 0.4, "nice_skill": 0.1, "experience": 0.2},
 "notice": "Decision support only. A human recruiter must review every candidate."}
```
**Errors:** `422 {"error": "validation", "details": [...]}` for bad input; `404 {"error": "unknown job_id" | "unknown resume_id", "details": [...]}`.

**Loading.** The fitted engine is loaded once at start-up (`artifacts/models/resume_ranker_v2.joblib`) and kept in memory, as in the Phase 2 service.
