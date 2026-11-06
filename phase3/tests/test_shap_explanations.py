from _common import *
from cortexhr.phase3.explanations.shap_explainer import explain_with_shap, additivity_error, top_drivers
from cortexhr.phase3.explanations.reasons import sentence, wording_problems, describe
from cortexhr.phase3.explanations.risk_report import select_reasons
from cortexhr.phase3.explanations.groups import SENSITIVE_PROXY


def _toy():
    rng = np.random.default_rng(0); X = pd.DataFrame(rng.normal(size=(8, 3)), columns=["a", "b", "c"]); bg = pd.DataFrame(rng.normal(size=(50, 3)), columns=["a", "b", "c"])
    w = np.array([1.0, -2.0, 0.5]); f = lambda d: d[["a", "b", "c"]].to_numpy() @ w
    return X, bg, w, f


def _shap_or_skip():
    try:
        import shap  # noqa: F401
    except Exception:
        raise Skip("shap is not installed here: pip install -r requirements.txt")


def test_shap_matches_the_exact_solution_for_a_linear_model():
    _shap_or_skip(); X, bg, w, f = _toy(); groups = {"a": ["a"], "b": ["b"], "c": ["c"]}
    r = explain_with_shap(f, X, bg, groups=groups, seed=1, max_background=50)
    exact = (X.to_numpy() - bg.sample(50, random_state=1).mean().to_numpy()) * w
    assert np.max(np.abs(r["group_values"].to_numpy() - exact)) < 1e-6    # permutation SHAP is exact for a linear model


def test_efficiency_property_is_exact():
    _shap_or_skip(); X, bg, w, f = _toy(); r = explain_with_shap(f, X, bg, groups={"ab": ["a", "b"], "c": ["c"]}, seed=2, max_background=20)
    assert additivity_error(r["group_values"], r["base_values"], r["predictions"]) < 1e-10 and r["additivity_max_error"] < 1e-10


def test_group_players_share_credit_once():
    _shap_or_skip(); X, bg, w, f = _toy(); r = explain_with_shap(f, X, bg, groups={"ab": ["a", "b"], "c": ["c"]}, seed=3, max_background=10)
    assert list(r["group_values"].columns) == ["ab", "c"] and r["feature_values"].shape == (8, 3)


def test_missing_shap_raises_instead_of_falling_back():
    import sys
    X, bg, w, f = _toy(); saved = sys.modules.get("shap", "absent")
    sys.modules["shap"] = None          # makes `import shap` fail, as if the package were not installed
    try:
        try: explain_with_shap(f, X, bg)
        except ImportError: ok = True
        else: ok = False
    finally:
        if saved == "absent": sys.modules.pop("shap", None)
        else: sys.modules["shap"] = saved
    assert ok, "expected ImportError when shap is missing"


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


def test_saved_explanation_run_comes_from_real_shap_on_all_430():
    import json; from cortexhr.phase3.common.paths import task_dir
    td = task_dir(14); run = json.loads((td / "explanation_run.json").read_text())
    assert run["rows"] == 430 and run["shap_version"] and "shap library" in run["reasons_source"]
    assert run["additivity_max_error"] < 1e-8 and run["feature_level_additivity_max_error"] < 1e-8 and run["format_checks_passed"] == run["format_checks_total"]
    for name in ("shap_full430_feature_values.csv", "shap_full430_group_values.csv", "shap_full430_summary.json"): assert (td / name).exists(), name
    assert len(pd.read_csv(td / "shap_full430_group_values.csv")) == 430


def test_no_home_made_shapley_code_is_left():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]; assert not (root / "src/cortexhr/phase3/explanations/shapley.py").exists()
    src = "\n".join(p.read_text(encoding="utf-8") for p in (root / "src").rglob("*.py")); assert "shapley_values" not in src and "monte carlo" not in src.lower()
