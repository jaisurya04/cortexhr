from _common import *
import json, joblib
from cortexhr.phase3.salary import model as S
from cortexhr.phase3.common.paths import MODELS, METADATA, task_dir
from cortexhr.phase3.common.config import load_config


def test_role_mean_baseline_logic():
    X = pd.DataFrame({"role_id": ["a", "a", "b"], "market_index": [1.0, 1.0, 2.0]}); b = S.RoleMeanBaseline().fit(X, np.array([10.0, 20.0, 40.0]))
    assert list(b.predict(X)) == [15.0, 15.0, 40.0] and b.predict(pd.DataFrame({"role_id": ["zzz"], "market_index": [1.0]}))[0] == np.mean([10, 20, 40])
    b2 = S.RoleMeanBaseline(True).fit(X, np.array([10.0, 20.0, 40.0])); assert abs(b2.predict(X)[2] - 40.0) < 1e-9


def test_metric_function():
    m = S.metrics([100, 200], [110, 180]); assert abs(m["MAE_rupees"] - 15) < 1e-9 and m["R2"] < 1


def test_chosen_features_are_job_related_only():
    assert "career_gap_months" not in S.JOB_FEATURES and "career_gap_months" in S.WITH_GAP_FEATURES
    assert not set(S.NEVER_INPUTS) & set(S.JOB_FEATURES) and {"gender", "age_years", "age_band"} <= set(S.NEVER_INPUTS)


def test_data_has_no_protected_columns_and_the_target_is_positive():
    d = S.load_salary(); assert not [c for c in ("gender", "age_years", "age_band") if c in d.columns] and (d.offered_annual_fixed > 0).all() and d.pay_vs_band.between(0.5, 3).all()


def test_saved_model_makes_ordered_bands():
    art = load_artifact(MODELS / "salary_band_v1.joblib"); d = S.load_salary().iloc[:50]; r = S.recommend_band(art["model"], d, art["band_multipliers"], art["features"])
    assert ((r.recommended_low < r.recommended_mid) & (r.recommended_mid < r.recommended_high)).all() and (r > 0).all().all()


def test_metadata_and_artifact_checksum():
    from cortexhr.phase3.attrition.v2 import file_sha256
    meta = json.loads((METADATA / "salary_band_v1_metadata.json").read_text()); assert meta["model_version"] == "salary-band-v1" and "career_gap_months" in meta["excluded_job_unrelated"]
    assert meta["sha256"] == file_sha256(MODELS / "salary_band_v1.joblib")


def test_model_beats_the_simple_baseline_on_the_test_set():
    t = pd.read_csv(task_dir(15) / "test_metrics.csv").set_index("model"); base = t.loc["baseline: average offered salary by role"]
    best = t.drop(index=[i for i in t.index if i.startswith("baseline")]).MAE_rupees.min(); assert best < base.MAE_rupees


def test_band_coverage_is_close_to_the_claimed_80_percent():
    meta = json.loads((METADATA / "salary_band_v1_metadata.json").read_text()); c = load_config()["salary"]["success_criteria"]
    assert abs(meta["band_coverage_test"] - c["band_target_coverage"]) <= 0.08


def test_band_output_is_a_range_not_a_single_number_and_has_no_protected_column():
    b = pd.read_csv(task_dir(15) / "band_recommendations_test.csv"); assert {"recommended_low", "recommended_mid", "recommended_high"} <= set(b.columns)
    assert not [c for c in b.columns if c in ("gender", "age_years", "age_band")]


def test_fairness_tables_exist_when_the_restricted_file_was_used():
    if vault_present("unlock_CX-15_protected_attributes.csv"):
        for f in ("gender_gaps.csv", "career_gap_means.csv", "gender_means.csv"): assert (task_dir(15) / f).exists()
