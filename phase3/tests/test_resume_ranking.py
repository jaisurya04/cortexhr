from _common import *
from cortexhr.phase3.resume import text as T
from cortexhr.phase3.resume.skills import SkillExtractor, build_vocabulary, split_skill_list
from cortexhr.phase3.resume.experience import extract_experience, experience_score
from cortexhr.phase3.resume.metrics import precision_at_k, recall_at_k, hit_rate_at_k, ndcg_at_k, mrr, attach_labels, evaluate, random_baseline
from cortexhr.phase3.resume.scoring import ResumeRanker, FORBIDDEN_SCORE_INPUTS
from cortexhr.phase3.resume.api import handle_rank_request, ApiError
from cortexhr.phase3.common.config import load_config


def test_masking_removes_name_email_and_tier():
    r = data()["resumes"].iloc[3]; m = T.mask_identifiers(r.resume_text, r.candidate_name, r.email, r.city).lower()
    assert r.email.lower() not in m and r.candidate_name.split()[0].lower() not in m and "tier" not in m


def test_skill_extraction_synonyms_and_span_consumption():
    d = data(); ext = SkillExtractor(build_vocabulary(d["job_descriptions"], d["resumes"]))
    f = ext.extract("Strengths include MS Excel, Tally and Customer Communication")
    assert f["excel"] == "synonym" and "customer communication" in f and "communication" not in f
    assert ext.extract(None) == {} and ext.extract(float("nan")) == {}


def test_skill_extraction_quality_against_structured_lists():
    d = data(); ext = SkillExtractor(build_vocabulary(d["job_descriptions"], d["resumes"])); rec = []
    for _, r in d["resumes"].iterrows():
        truth = {ext.canon(s) for s in split_skill_list(r.skills)}; got = set(ext.extract(T.for_skill_extraction(r.resume_text, r.candidate_name, r.email, r.city))); rec.append(len(truth & got) / len(truth))
    assert sum(rec) / len(rec) > 0.95


def test_experience_extraction_matches_the_structured_years_for_every_resume():
    for _, r in data()["resumes"].iterrows():
        e = extract_experience(T.mask_identifiers(r.resume_text, r.candidate_name, r.email, r.city)); assert e["total_years"] is not None and abs(e["total_years"] - r.total_experience_years) <= 0.15, r.resume_id


def test_experience_edge_cases():
    assert extract_experience("")["total_years"] is None and "empty_text" in extract_experience(None)["flags"]
    assert extract_experience("Analyst, ACME Ltd (2.0 yrs)\nAnalyst, ACME Ltd (2.0 yrs)")["total_years"] == 2.0     # duplicate role not double counted
    assert experience_score(30, 30, 3) == 1.0 and experience_score(0, 0, 3) == 0.0                                   # capped at the requirement


def test_metric_functions_on_a_toy_ranking():
    df = pd.DataFrame({"rank": [1, 2, 3, 4, 5, 6], "relevant": [1, 0, 1, 0, 0, 1], "rel": [5, 2, 4, 1, 1, 4]})
    assert precision_at_k(df, 5) == 0.4 and abs(recall_at_k(df, 5) - 2 / 3) < 1e-12 and hit_rate_at_k(df, 5) == 1.0 and mrr(df) == 1.0
    assert 0 < ndcg_at_k(df, 5) <= 1 and ndcg_at_k(df.assign(rel=[5, 4, 4, 1, 1, 2]), 5) <= 1


def test_ranking_is_deterministic_and_returns_top_k():
    v1, v2 = rankers(); a = v2.top_k("J01", 5); b = v2.top_k("J01", 5)
    assert list(a.resume_id) == list(b.resume_id) and len(a) == 5 and a["rank"].tolist() == [1, 2, 3, 4, 5] and (a.score.diff().dropna() <= 1e-12).all()
    assert (a["human_review"].str.contains("Required")).all()


def test_ranking_inputs_exclude_protected_fields():
    u = rankers()[1].used_inputs()
    assert {"candidate_name", "gender", "age", "college_tier"} <= set(u["never_used"]) and "candidate_name" not in " ".join(u["score_inputs"])
    assert "target_job_family" in FORBIDDEN_SCORE_INPUTS


def test_changing_only_the_name_does_not_change_the_masked_score():
    v1, v2 = rankers(); p = data()["resume_bias_pairs"].iloc[0]
    assert v2.score_pair(p.resume_id_original, "J01")["score"] == v2.score_pair(p.resume_id_twin, "J01")["score"]


def test_weights_must_sum_to_one():
    d = data()
    try: ResumeRanker(weights=dict(tfidf=.5, required_skill=.5, nice_skill=.5, experience=.5)).fit(d["resumes"], d["job_descriptions"])
    except ValueError: return
    raise AssertionError("bad weights accepted")


def test_ranking_quality_meets_the_success_criteria_fixed_in_advance():
    d = data(); v1, v2 = rankers(); ev = attach_labels(v2.score_all(d["resume_job_labels"][["resume_id", "job_id"]]), d["resume_job_labels"])
    per, agg = evaluate(ev, 5); rb = random_baseline(ev, 5, 100, 42); c = load_config()["resume"]["success_criteria"]
    assert agg["precision@5"] >= c["precision_at_5_min_multiple_of_random"] * rb["precision@5"] and agg["hit_rate@5"] >= c["hit_rate_at_5_min"]


def test_api_contract_good_and_bad_requests():
    v2 = rankers()[1]; r = handle_rank_request({"job_id": "J01", "top_k": 3}, v2)
    assert len(r["candidates"]) == 3 and "candidate_name" not in str(r["candidates"]) and r["notice"].startswith("Decision support")
    for bad, status in (({"job_id": "J01", "gender": "F"}, 422), ({"job_id": "J01", "top_k": 99}, 422), ({}, 422), ({"job_id": "NOPE"}, 404), ({"job_id": "J01", "resume_ids": ["RES-9999"]}, 404)):
        try: handle_rank_request(bad, v2)
        except ApiError as e: assert e.status == status, (bad, e.status)
        else: raise AssertionError(f"accepted {bad}")
    r2 = handle_rank_request({"job": {"title": "Data Analyst", "required_skills": "Python; SQL", "min_experience_years": 2}, "top_k": 2}, v2); assert len(r2["candidates"]) == 2
