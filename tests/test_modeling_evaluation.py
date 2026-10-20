from _common import *
from cortexhr import modeling as M, evaluation as E, features as F, attendance as T

def test_split_stratified_reproducible_and_target_separate():
    if not data_ok(): return
    S = frame(); a = M.make_split(S["X"], S["y"]); b = M.make_split(S["X"], S["y"])
    assert (a[0].index == b[0].index).all() and C.TARGET not in S["Xtr"].columns
    assert abs(S["ytr"].mean() - S["yte"].mean()) < 0.01 and len(S["Xte"]) == 477

def test_same_folds_for_every_model():
    if not data_ok(): return
    S = frame(); f1, f2 = M.make_folds(S["Xtr"], S["ytr"]), M.make_folds(S["Xtr"], S["ytr"])
    assert all((a[1] == b[1]).all() for a, b in zip(f1, f2)) and len(f1) == 5
    assert set(np.concatenate([v for _, v in f1])) == set(range(len(S["Xtr"])))   # every row validated exactly once

def test_baselines_sane_no_leakage():
    if not data_ok(): return
    S = frame(); oof, fm, summ = M.evaluate_zoo(S["Xtr"], S["ytr"], S["folds"], True, ["Dummy (always stays)", "Logistic Regression (balanced)"])
    s = summ.set_index("model"); assert s.loc["Dummy (always stays)", "roc_auc_mean"] == 0.5 and s.loc["Dummy (always stays)", "recall_mean"] == 0
    assert 0.6 < s.loc["Logistic Regression (balanced)", "roc_auc_mean"] < 0.9
    assert s.loc["Dummy (always stays)", "folds_with_undefined_precision"] == 5

def test_binary_report_zero_division_and_top10():
    y = np.array([0] * 90 + [1] * 10); s = np.zeros(100); r = E.binary_report(y, s)
    assert r["precision"] == 0 and not r["precision_defined"] and r["recall"] == 0 and r["accuracy"] == 0.9
    s2 = np.linspace(0, 1, 100); y2 = (np.arange(100) >= 90).astype(int); r2 = E.binary_report(y2, s2)
    assert r2["roc_auc"] == 1.0 and r2["top_k"] == 10 and r2["top_precision"] == 1.0 and r2["top_recall"] == 1.0 and abs(r2["top_lift"] - 10) < 1e-9

def test_risk_bands_10_20_70_and_ties_deterministic():
    s = np.random.default_rng(0).random(1000); b = E.risk_bands(s)
    assert (b == "High").sum() == 100 and (b == "Medium").sum() == 200 and (b == "Low").sum() == 700
    assert s[(b == "High").to_numpy()].min() > s[(b == "Medium").to_numpy()].max() > s[(b == "Low").to_numpy()].max() - 1e-12
    t = np.ones(50); b1, b2 = E.risk_bands(t), E.risk_bands(t); assert (b1 == b2).all() and list(b1[:5]) == ["High"] * 5 and (b1 == "High").sum() == 5
    assert (E.risk_bands(np.arange(7.0)) == "High").sum() == 1                      # ceil(0.7)=1

def test_band_summary_and_gains():
    y = np.r_[np.ones(10), np.zeros(90)].astype(int); s = np.r_[np.linspace(1, .9, 10), np.linspace(.5, 0, 90)]
    bs = E.band_summary(y, E.risk_bands(s)); assert bs.set_index("band").loc["High", "share_of_all_leavers_pct"] == 100
    g = E.gains_table(y, s); assert g.capture_rate_pct.iloc[0] == 100 and g.lift.iloc[0] == 10

def test_recalibrator_monotone():
    p = np.linspace(.05, .95, 50); y = (np.random.default_rng(1).random(50) < p * .3).astype(int); c = E.Recalibrator().fit(p, y).predict(p)
    assert (np.diff(c) >= -1e-12).all()

def test_smote_only_adds_minority_and_is_reproducible():
    X = np.random.default_rng(0).normal(size=(100, 3)); y = np.r_[np.ones(10), np.zeros(90)].astype(int)
    a = M.SimpleSMOTE(random_state=1).fit_resample(X, y); b = M.SimpleSMOTE(random_state=1).fit_resample(X, y)
    assert (a[1] == 1).sum() == (a[1] == 0).sum() == 90 and np.allclose(a[0], b[0]) and np.allclose(a[0][:100], X)

def test_catch_rate_math():
    r = E.catch_rate(["a", "b", "c", "d"], ["a", "b", "x"]); assert r["caught"] == 2 and r["precision"] == .5 and abs(r["recall"] - 2 / 3) < 1e-9

def test_attendance_never_uses_labels():
    for c in ("is_anomaly", "anomaly_type"):
        try: T.assert_no_labels(["present_days", c])
        except ValueError: continue
        raise AssertionError("label allowed")
    assert not ({c.lower() for c in T.RAW_FEATURES + T.HISTORY_FEATURES} & T.LABEL_LIKE)

def test_attendance_rules_and_detector():
    if not data_ok(): return
    from cortexhr import data_loading as D
    f = T.detect(D.load_sheet(C.SHEET_ATT)); assert 215 <= f.iforest_flag.sum() <= 225 and f.rule_zero_attendance.sum() == 50
    assert f.review_rank.dropna().is_unique and int(f.review_rank.max()) == int(f.flagged.sum())
