from _common import *
from cortexhr.phase3.fairness.twins import validate_pairs, twin_gaps, gap_summary
from cortexhr.phase3.resume.scoring import ResumeRanker


def test_twin_pairs_are_valid():
    d = data(); v = validate_pairs(d["resume_bias_pairs"], d["resumes"])
    assert len(v) == 25 and (v.unexpected_differences == "").all() and v.skills_actually_identical.all() and v.experience_identical.all()


def test_unmasked_screener_leaks_through_the_free_text():
    d = data(); g = twin_gaps(rankers()[0], d["resume_bias_pairs"], d["resume_job_labels"])
    assert len(g) == 75 and g.pair_id.nunique() == 25 and (g.abs_gap > 1e-9).sum() >= 70


def test_masked_screener_has_no_twin_gap():
    d = data(); g = twin_gaps(rankers()[1], d["resume_bias_pairs"], d["resume_job_labels"]); s = gap_summary(g)
    assert s["max_abs_gap"] == 0 and s["comparisons_with_nonzero_gap"] == 0


def test_name_and_email_are_the_source_of_the_gap():
    d = data(); rk = ResumeRanker(mask=True, mask_parts=("name", "email")).fit(d["resumes"], d["job_descriptions"])
    assert gap_summary(twin_gaps(rk, d["resume_bias_pairs"], d["resume_job_labels"]))["max_abs_gap"] == 0


def test_ties_are_not_always_broken_in_favour_of_the_original():
    d = data(); g = twin_gaps(rankers()[1], d["resume_bias_pairs"], d["resume_job_labels"])
    assert 0.2 < g.twin_ranked_higher.mean() < 0.8


def test_ranking_quality_does_not_collapse_after_the_fix():
    from cortexhr.phase3.resume.metrics import attach_labels, evaluate
    d = data(); v1, v2 = rankers(); q = [evaluate(attach_labels(r.score_all(d["resume_job_labels"][["resume_id", "job_id"]]), d["resume_job_labels"]), 5)[1] for r in (v1, v2)]
    assert q[0]["ndcg@5"] - q[1]["ndcg@5"] < 0.05


def test_group_gap_helper_signature_and_values():
    from cortexhr.phase3.explanations.risk_report import load_helpers
    H = load_helpers(); means, gap = H.group_gap(np.array([1.0, 2.0, 3.0, 5.0]), np.array(["a", "a", "b", "b"])); assert abs(gap - 2.5) < 1e-12 and set(means) == {"a", "b"}
