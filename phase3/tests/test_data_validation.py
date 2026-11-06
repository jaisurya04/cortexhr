from _common import *
from cortexhr.phase3.common import validation as V
from cortexhr.phase3.common.config import load_config
from cortexhr.phase3.explanations.groups import GROUPS, check_groups


def _raises(fn, *a):
    try: fn(*a)
    except ValueError: return True
    return False


def test_phase3_data_files_have_expected_shapes():
    d = data()
    assert d["resumes"].shape[0] == 185 and d["job_descriptions"].shape[0] == 10
    assert d["resume_job_labels"].shape[0] == 555 and d["resume_bias_pairs"].shape[0] == 25
    assert read_csv("attrition_scoring_2026").shape[0] == 430 and read_csv("sample_attrition_risk").shape[0] == 430
    assert read_csv("salary_training").shape[0] == 2934


def test_keys_are_unique_and_links_are_intact():
    d = data()
    assert d["resumes"].resume_id.is_unique and d["job_descriptions"].job_id.is_unique
    assert not d["resume_job_labels"].duplicated(["resume_id", "job_id"]).any()
    V.require_subset(d["resume_job_labels"].resume_id, d["resumes"].resume_id); V.require_subset(d["resume_job_labels"].job_id, d["job_descriptions"].job_id)
    V.require_subset(d["resume_bias_pairs"].resume_id_twin, d["resumes"].resume_id)


def test_labels_are_internally_consistent():
    l = data()["resume_job_labels"]
    assert l.human_suitability_score.between(1, 5).all() and ((l.human_suitability_score >= 4) == (l.shortlist == "Y")).all()


def test_scoring_file_matches_the_sample_format_order():
    sc, sm = read_csv("attrition_scoring_2026"), read_csv("sample_attrition_risk")
    assert sc.employee_id.is_unique and list(sc.employee_id) == list(sm.employee_id)


def test_validators_raise_on_bad_input():
    df = pd.DataFrame({"a": [1, 1], "b": [2, 3]})
    assert _raises(V.require_columns, df, ["zzz"]) and _raises(V.require_unique, df, "a") and _raises(V.require_subset, ["x"], ["y"])
    assert _raises(V.require_probability, [0.2, 1.4]) and _raises(V.require_probability, [float("nan")])


def test_config_is_consistent():
    c = load_config(); w = c["resume"]["weights"]
    assert abs(sum(w.values()) - 1) < 1e-9 and c["seed"] == 42 and c["explain"]["top_n"] == 3 and "Human review required" in c["explain"]["human_review_text"]


def test_explanation_groups_cover_every_model_input_once():
    from cortexhr.phase3.attrition import v2
    sc, X, rep = v2.load_scoring(); check_groups(X.columns)
    assert sum(len(v) for v in GROUPS.values()) == X.shape[1] == 43


def test_scoring_inputs_have_no_forbidden_or_protected_columns():
    from cortexhr.phase3.attrition import v2
    from cortexhr import config as C
    sc, X, rep = v2.load_scoring()
    assert not [c for c in C.FORBIDDEN_AS_INPUT if c in X.columns]
    assert not [c for c in X.columns if any(k in c.lower() for k in ("gender", "age_years", "religion", "marital"))]
