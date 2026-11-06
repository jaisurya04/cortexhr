from _common import *
import json, joblib
from sklearn.dummy import DummyClassifier
from cortexhr.phase3.attrition import v2 as V
from cortexhr.phase3.common.config import load_config
from cortexhr.phase3.common.paths import MODELS, METADATA, ROOT


def test_blend_classifier_averages_probabilities():
    X = pd.DataFrame({"a": [0, 1, 0, 1]}); y = pd.Series([0, 1, 0, 1])
    b = V.BlendClassifier([DummyClassifier(strategy="prior"), DummyClassifier(strategy="prior")]).fit(X, y)
    p = b.predict_proba(X); assert np.allclose(p.sum(1), 1) and np.allclose(p[:, 1], 0.5)


def test_metric_helpers_on_toy_values():
    y = np.array([0, 0, 1, 1]); p = np.array([0.1, 0.4, 0.35, 0.8])
    t = V.threshold_metrics(y, p, 0.5); assert t["tp"] == 1 and t["fp"] == 0 and t["fn"] == 1 and t["tn"] == 2
    assert V.top_fraction_metrics(y, p, 0.25)["precision_top10"] == 1.0
    assert V.expected_calibration_error(np.array([0, 1] * 50), np.linspace(0.01, 0.99, 100)) < 0.3


def test_verdict_rule_is_conservative():
    assert V.verdict({"mean_diff": 0.01, "sd_diff": .01, "boot_ci_low": -0.002, "boot_ci_high": 0.02})["gain_is_meaningful"] is False
    assert V.verdict({"mean_diff": 0.02, "sd_diff": .01, "boot_ci_low": 0.004, "boot_ci_high": 0.03})["gain_is_meaningful"] is True


def test_same_folds_as_phase_one_and_no_leakage_columns():
    d = V.load_training(); from cortexhr import config as C
    assert len(d["folds"]) == 5 and len(d["Xtr"]) == 1906 and len(d["Xte"]) == 477 and C.SEED == 42
    assert not [c for c in C.FORBIDDEN_AS_INPUT if c in d["X"].columns]
    assert set(d["Xtr"].index).isdisjoint(set(d["Xte"].index))


def test_saved_model_artifacts_and_metadata():
    meta = json.loads((METADATA / "attrition_v2_metadata.json").read_text())
    assert meta["model_version"] == "attrition-v2" and meta["seed"] == 42 and meta["calibration_method"] in ("sigmoid", "isotonic")
    for f in ("attrition_v2.joblib", "attrition_v2_train_only.joblib"):
        assert V.file_sha256(MODELS / f) == meta["artifact_sha256"][f]
    assert not (ROOT / "artifacts" / "models" / "attrition_model_v1.joblib").exists()          # Phase 1 baseline artifacts are not overwritten (different folder and names)


def test_model_scores_the_430_employees_in_range_and_is_reproducible():
    m = load_artifact(MODELS / "attrition_v2.joblib"); sc, X, rep = V.load_scoring(); p = m.predict_proba(X)[:, 1]
    assert len(p) == 430 and ((p >= 0) & (p <= 1)).all() and np.allclose(p, m.predict_proba(X)[:, 1]) and 0.08 < p.mean() < 0.25


def test_calibration_meets_the_success_criterion():
    meta = json.loads((METADATA / "attrition_v2_metadata.json").read_text()); lim = load_config()["attrition"]["success_criteria"]["max_expected_calibration_error"]
    v2m = [r for r in meta["test_metrics"] if r["model"].startswith("attrition-v2")][0]; assert v2m["ece"] <= lim and v2m["brier"] < 0.15


def test_gradient_boosting_comparison_is_reported_honestly():
    t = pd.read_csv(ROOT / "reports" / "task13" / "paired_differences.csv"); assert {"roc_auc", "pr_auc"} <= set(t.metric) and (t.n_folds == 5).all()
    log = pd.read_csv(ROOT / "reports" / "task13" / "tuning_log.csv"); assert len(log) >= 10 and log.decision.str.startswith("selected").sum() == 1


def test_lightgbm_candidate_is_real_and_uses_phase1_pipeline_when_installed():
    from cortexhr.phase3.attrition import v2 as V
    if V.optional_libs().get("lightgbm"):
        candidate = V.make_lightgbm()
        assert hasattr(candidate, "named_steps")
        assert "features" in candidate.named_steps and "model" in candidate.named_steps
        assert candidate.named_steps["model"].__class__.__name__ == "LGBMClassifier"
        assert len(V.lightgbm_tuning_grid()) >= 2
    else:
        assert V.optional_libs().get("lightgbm") is None


def test_lightgbm_audit_artifacts_and_metrics_are_saved_consistently():
    import json, joblib
    from cortexhr.phase3.attrition import v2 as V
    from cortexhr.phase3.common.paths import MODELS, METADATA, task_dir
    if not V.optional_libs().get("lightgbm"):
        return
    td = task_dir(13)
    for name in ("lightgbm_tuning_log.csv", "lightgbm_fold_metrics.csv", "lightgbm_cv_summary.csv", "lightgbm_test_metrics.csv", "cv_summary_with_lightgbm.csv", "test_metrics_with_lightgbm.csv"):
        assert (td / name).exists(), name
    artifact = MODELS / "attrition_lightgbm_candidate.joblib"
    meta_path = METADATA / "attrition_lightgbm_candidate_metadata.json"
    assert artifact.exists() and meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["model_version"] == "attrition-lightgbm-candidate-v1"
    assert meta["artifact_sha256"] == V.file_sha256(artifact)
    model = load_artifact(artifact)
    assert hasattr(model, "predict_proba")
    folds = pd.read_csv(td / "lightgbm_fold_metrics.csv")
    assert folds.groupby("model").fold.nunique().eq(5).all()
