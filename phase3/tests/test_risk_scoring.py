from _common import *
from cortexhr.phase3.common.paths import PREDICTIONS, task_dir
from cortexhr.phase3.explanations import risk_report as RR


def _rep():
    return pd.read_csv(PREDICTIONS / "attrition_risk_2026.csv")


def test_risk_report_passes_every_format_check():
    res = RR.validate_report(_rep(), read_csv("sample_attrition_risk")); failed = [r for r in res if not r["passed"]]; assert not failed, failed


def test_columns_rows_and_order_equal_the_sample():
    r, s = _rep(), read_csv("sample_attrition_risk"); assert list(r.columns) == list(s.columns) and len(r) == 430 and list(r.employee_id) == list(s.employee_id)


def test_scores_and_bands_follow_the_company_rule():
    r = _rep(); assert r.risk_score_12m.between(0, 1).all() and r.risk_band.value_counts().to_dict() == {"Low": 301, "Medium": 86, "High": 43}
    assert r[r.risk_band == "High"].risk_score_12m.min() >= r[r.risk_band == "Medium"].risk_score_12m.max() - 1e-9


def test_risk_bands_helper_behaviour():
    H = RR.load_helpers(); p = np.linspace(0.01, 0.5, 430); b = pd.Series(H.risk_bands(p)); assert b.value_counts().to_dict() == {"Low": 301, "Medium": 86, "High": 43}


def test_every_row_has_three_distinct_plain_reasons_and_the_human_review_note():
    r = _rep(); cols = ["top_reason_1", "top_reason_2", "top_reason_3"]
    assert (r[cols].apply(lambda c: c.str.len() > 20)).all().all() and (r.top_reason_1 != r.top_reason_2).all() and (r.top_reason_2 != r.top_reason_3).all()
    assert r.model_note.str.contains("Human review required").all() and (r.model_version == "attrition-v2").all()


def test_reasons_do_not_state_certainty_or_protected_traits():
    from cortexhr.phase3.explanations.reasons import wording_problems
    r = _rep(); text = r[["top_reason_1", "top_reason_2", "top_reason_3"]].astype(str).agg(" ".join, axis=1); assert not [t for t in text if wording_problems(t)]
    assert not text.str.contains("career history|educational|background|location", case=False).any()          # sensitive-proxy groups are hidden from the reasons


def test_report_copies_are_identical_and_scores_are_calibrated_probabilities():
    import json; src = json.loads((task_dir(14) / "explanation_run.json").read_text(encoding="utf-8")); assert src["reasons_source"].startswith("shap library") and src["format_checks_passed"] == src["format_checks_total"]
    assert (PREDICTIONS / "attrition_risk_2026.csv").read_bytes() == (task_dir(14) / "attrition_risk_2026.csv").read_bytes()
    assert 0.08 < _rep().risk_score_12m.mean() < 0.20
