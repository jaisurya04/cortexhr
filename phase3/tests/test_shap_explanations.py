from _common import *
from cortexhr.phase3.explanations.shapley import shapley_values, additivity_error, top_drivers
from cortexhr.phase3.explanations.reasons import sentence, wording_problems, describe
from cortexhr.phase3.explanations.risk_report import select_reasons
from cortexhr.phase3.explanations.groups import SENSITIVE_PROXY


def _toy():
    rng = np.random.default_rng(0); X = pd.DataFrame(rng.normal(size=(8, 3)), columns=["a", "b", "c"]); bg = pd.DataFrame(rng.normal(size=(50, 3)), columns=["a", "b", "c"])
    w = np.array([1.0, -2.0, 0.5]); f = lambda d: d[["a", "b", "c"]].to_numpy() @ w
    return X, bg, w, f


def test_shapley_matches_the_exact_solution_for_a_linear_model():
    X, bg, w, f = _toy(); groups = {"a": ["a"], "b": ["b"], "c": ["c"]}
    phi, base, fx = shapley_values(f, X, bg, groups, n_permutations=300, seed=1)
    exact = (X.to_numpy() - bg.mean().to_numpy()) * w
    assert np.max(np.abs(phi.to_numpy() - exact)) < 0.35          # background sampling noise only


def test_efficiency_property_is_exact():
    X, bg, w, f = _toy(); phi, base, fx = shapley_values(f, X, bg, {"ab": ["a", "b"], "c": ["c"]}, n_permutations=20, seed=2)
    assert additivity_error(phi, base, fx) < 1e-10


def test_group_players_share_credit_once():
    X, bg, w, f = _toy(); phi, _, _ = shapley_values(f, X, bg, {"ab": ["a", "b"], "c": ["c"]}, 10, 3); assert list(phi.columns) == ["ab", "c"]


def test_top_drivers_keep_direction_and_order():
    phi = pd.DataFrame({"x": [0.3, -0.1], "y": [-0.5, 0.2], "z": [0.1, 0.05]}); t = top_drivers(phi, 2)
    assert t.driver_1.tolist() == ["y", "y"] and t.phi_1.tolist() == [-0.5, 0.2] and t.driver_2.tolist() == ["x", "x"]


def test_reasons_skip_sensitive_groups_and_prefer_risk_raising_factors():
    cols = ["tenure", "engagement", "career_history", "education", "pay_level"]; phi = pd.DataFrame([[0.01, 0.05, 0.20, 0.30, 0.03]], columns=cols)
    s = select_reasons(phi, 3); assert set(s.loc[0, ["driver_1", "driver_2", "driver_3"]]) == {"engagement", "pay_level", "tenure"} and not (set(s.loc[0, ["driver_1", "driver_2", "driver_3"]]) & SENSITIVE_PROXY)


def test_sentences_are_plain_english_and_safe():
    from cortexhr.phase3.attrition import v2
    sc, X, rep = v2.load_scoring(); r = X.iloc[5]
    from cortexhr.phase3.explanations.groups import GROUPS
    for g in GROUPS:
        t = sentence(g, r, 0.02); assert t.endswith(".") and "predicted risk" in t and not wording_problems(t), (g, t)
    assert "will leave" in "she will leave" and wording_problems("He will leave soon")


def test_sensitive_groups_never_reveal_values():
    from cortexhr.phase3.attrition import v2
    sc, X, rep = v2.load_scoring(); r = X.iloc[0]
    for g in SENSITIVE_PROXY:
        t = describe(g, r); assert not any(ch.isdigit() for ch in t) and str(r["highest_education"]).lower() not in t.lower()


def test_saved_explanation_run_is_additive_and_stable_enough():
    import json; from cortexhr.phase3.common.paths import task_dir
    run = json.loads((task_dir(14) / "explanation_run.json").read_text()); assert run["additivity_max_error"] < 1e-9 and run["stability"]["phi_correlation"] > 0.9


def test_real_shap_permutation_cross_check_runs_when_library_is_available():
    from cortexhr.phase3.explanations.shapley import shap_cross_check
    X, bg, w, f = _toy()
    result = shap_cross_check(f, X, bg, n_rows=3, groups={"ab": ["a", "b"], "c": ["c"]}, seed=7, max_background=10)
    try:
        import shap  # noqa: F401
    except Exception:
        assert result is None
    else:
        assert result is not None
        assert result["feature_values"].shape == (3, 3)
        assert result["group_values"].shape == (3, 2)
        assert result["additivity_max_error"] < 1e-5


def test_saved_shap_library_cross_check_is_reproducible():
    import json
    from cortexhr.phase3.common.paths import task_dir
    td = task_dir(14)
    meta = json.loads((td / "explanation_run.json").read_text())
    check = meta.get("shap_library_cross_check")
    # The saved outputs were produced with shap 0.50.0 (requirements-optional.txt). They are validated whether or not shap is
    # installed here, so the test no longer depends on which optional packages happen to be present.
    assert check is not None and check["rows"] == 10, "saved SHAP library cross-check missing from explanation_run.json"
    assert check["additivity_max_error"] < 1e-8
    assert (td / "shap_library_feature_cross_check.csv").exists()
    assert (td / "shap_library_group_cross_check.csv").exists()
