"""Day 16: forecasting maths, events, backtesting, outputs."""
import numpy as np
import pandas as pd
from _common import csv_dir, need, skip

from cortexhr_phase4.forecasting import backtest as bt
from cortexhr_phase4.forecasting import data, events, intervals, metrics, models, pipeline
from cortexhr_phase4.forecasting.baseline import SeasonalMean, SeasonalNaiveDrift


def _series(n=60, seed=1, slope=1.0, season=3.0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="MS")
    v = 100 + slope * np.arange(n) + season * np.sin(2 * np.pi * np.arange(n) / 12) + rng.normal(0, .5, n)
    return pd.Series(v, index=idx)


def test_metrics_hand_computed():
    a, f = [10, 20, 30], [12, 18, 33]
    assert metrics.mae(a, f) == (2 + 2 + 3) / 3
    assert abs(metrics.rmse(a, f) - np.sqrt((4 + 4 + 9) / 3)) < 1e-12
    assert abs(metrics.mape([10, 0, 20], [11, 5, 18]) - np.mean([10, 10])) < 1e-9      # zero actual is skipped
    assert metrics.bias(a, f) == (-2 + 2 - 3) / 3
    assert metrics.coverage([1, 5, 9], [0, 0, 0], [2, 4, 10]) == 2 / 3


def test_seasonal_naive_drift_formula():
    y = _series(40)
    fc = SeasonalNaiveDrift().forecast(y, 6)
    v = y.to_numpy()
    drift = v[-1] - v[-13]
    for h in range(1, 7):
        assert abs(fc[h - 1] - (v[len(v) - 12 + h - 1] + drift)) < 1e-9
    try:
        SeasonalNaiveDrift().forecast(y.iloc[:12], 3)
        raise AssertionError("expected ValueError for < 13 months")
    except ValueError:
        pass


def test_seasonal_mean_formula():
    y = _series(48)
    fc = SeasonalMean(3).forecast(y, 3)
    v = y.to_numpy()
    n = len(v)
    assert abs(fc[0] - np.mean([v[n - 12], v[n - 24], v[n - 36]])) < 1e-9


def test_ar1_cumulative_sd_matches_simulation():
    rng = np.random.default_rng(0)
    phi, s2, H, N = 0.4, 2.0, 6, 40000
    u = np.zeros(N)
    paths = []
    for _ in range(H):
        u = phi * u + rng.normal(0, np.sqrt(s2), N)
        paths.append(u.copy())
    sim = np.cumsum(np.array(paths), axis=0).std(axis=1)
    assert np.allclose(sim, intervals.ar1_cumulative_sd(phi, s2, H), rtol=0.03)
    for bad in (1.0, -1.2):
        try:
            intervals.ar1_cumulative_sd(bad, 1, 3)
            raise AssertionError("phi outside (-1,1) must be rejected")
        except ValueError:
            pass


def test_empirical_interval_needs_enough_errors_and_is_symmetric():
    errs = {h: np.linspace(-3, 3, 20) for h in range(1, 4)}
    lo, hi = intervals.empirical_interval(np.array([50., 50., 50.]), errs)
    assert np.allclose(50 - lo, hi - 50) and (hi > 50).all()
    try:
        intervals.empirical_interval(np.array([1.0]), {1: np.array([1., 2.])})
        raise AssertionError("fewer than 8 errors must be refused")
    except ValueError:
        pass


def test_backtest_never_sees_the_future():
    y = _series(70)
    seen = []

    def fn(train, hz, n):
        assert len(train) == n and train.index[-1] == y.index[n - 1], "model received more than its training window"
        seen.append(n)
        p = SeasonalNaiveDrift().forecast(train, hz)
        return p, p - 1, p + 1

    out = bt.rolling_backtest(y, 6, fn, 49, 5)
    assert seen == list(range(49, 65 + 1, 5))
    assert (out.groupby("origin").h.max() == 6).all()
    first = out[out.origin == y.index[48]]
    assert (first.month.to_numpy() == y.index[49:55]).all(), "targets must be the months right after the origin"


def test_rolling_errors_counts():
    y = _series(70)
    e = bt.rolling_errors(y, SeasonalNaiveDrift().forecast, 6, 36)
    assert len(e[1]) == 70 - 36 and len(e[6]) == 70 - 36 - 5


def test_events_classification_and_matrix():
    assert events.classify("COVID-19 hiring freeze") == "hiring_freeze"
    assert events.classify("Engineering hiring drive") == "hiring_boost"
    assert events.classify("Hybrid work policy") == "policy_change"
    ev = data.load_events(csv_dir())
    idx = pd.date_range("2020-01-01", periods=12, freq="MS")
    m_all = events.event_matrix(idx, ev, "ALL")
    assert m_all.hiring_freeze.sum() >= 2 and set(np.unique(m_all.to_numpy())) <= {0.0, 1.0}
    only_dept = events.event_matrix(pd.date_range("2022-01-01", periods=48, freq="MS"), ev, "D10")
    assert only_dept.sum().sum() <= events.event_matrix(pd.date_range("2022-01-01", periods=48, freq="MS"), ev, "ALL").sum().sum()
    sc = events.Scenario("hiring_freeze", "2026-11", "2026-12")
    fut = events.event_matrix(pd.date_range("2026-10-01", periods=4, freq="MS"), ev, "ALL", [sc])
    assert fut.hiring_freeze.tolist() == [0, 1, 1, 0]
    for bad in (("nonsense", "2026-01", "2026-02"), ("hiring_freeze", "2026-05", "2026-01")):
        try:
            events.Scenario(*bad)
            raise AssertionError("invalid scenario accepted")
        except ValueError:
            pass


def test_data_validation_rejects_broken_history(tmp_path=None):
    import shutil
    import tempfile
    from pathlib import Path
    d = Path(tempfile.mkdtemp())
    for f in (data.HISTORY_FILE, data.EVENTS_FILE):
        shutil.copy(csv_dir() / f, d / f)
    h = pd.read_csv(d / data.HISTORY_FILE)
    h.loc[5, "hires"] += 1                                  # break the headcount identity
    h.to_csv(d / data.HISTORY_FILE, index=False)
    try:
        data.load_history(d)
        raise AssertionError("identity violation not detected")
    except data.DataError as e:
        assert "identity" in str(e)
    h2 = pd.concat([h, h.iloc[[0]]])
    h2.loc[5, "hires"] -= 1
    h2.to_csv(d / data.HISTORY_FILE, index=False)
    try:
        data.load_history(d)
        raise AssertionError("duplicate row not detected")
    except data.DataError as e:
        assert "duplicate" in str(e)


def test_history_matches_sample_series_ids():
    h = data.load_history(csv_dir())
    sample = pd.read_csv(csv_dir() / data.SAMPLE_FILE)
    assert data.series_ids(h) == ["ALL"] + sorted(set(sample[sample.metric == "headcount_end"].department_id) - {"ALL"})


def test_pipeline_output_matches_sample_forecast_layout():
    run = pipeline.run_forecast(csv_dir(), use_improved=False, rolling_step=6)
    sample = pd.read_csv(csv_dir() / data.SAMPLE_FILE)
    f = run.forecast
    assert list(f.columns) == list(sample.columns) == data.SAMPLE_COLUMNS
    assert len(f) == 6 * 11 + 6 + 6 and set(f.metric) == {"headcount_end", "exits", "attrition_rate_monthly_pct"}
    assert (f.groupby(["metric", "department_id"]).size() == 6).all()
    assert f.forecast_month.iloc[0] == "2026-10-01" and (f.trained_through == "2026-09").all()
    assert (f.lower_80 <= f.forecast_value).all() and (f.forecast_value <= f.upper_80).all()
    assert (f.lower_80 >= 0).all() and f.forecast_id.is_unique
    assert set(f.method) and run.meta["coverage"] == 0.8


def test_pipeline_is_reproducible():
    a = pipeline.run_forecast(csv_dir(), use_improved=False, rolling_step=6)
    b = pipeline.run_forecast(csv_dir(), use_improved=False, rolling_step=6)
    assert pipeline.content_hash(a.forecast) == pipeline.content_hash(b.forecast)
    assert a.comparison.fillna(-1).equals(b.comparison.fillna(-1))


def test_backtest_tables_are_honest_and_complete():
    run = pipeline.run_forecast(csv_dir(), use_improved=False, rolling_step=6)
    hold = run.holdout[(run.holdout.department_id == "ALL") & (run.holdout.metric == "headcount_end")]
    assert len(hold) == 6 and hold.month.min() == pd.Timestamp("2026-04-01") and hold.month.max() == pd.Timestamp("2026-09-01")
    hist = data.get_series(data.load_history(csv_dir()), "ALL", "headcount_end")
    assert np.allclose(hold.actual.to_numpy(), hist.loc["2026-04-01":].to_numpy()), "holdout actuals must be the real last 6 months"
    cmp = run.comparison
    row = cmp[(cmp.department_id == "ALL") & (cmp.metric == "headcount_end") & (cmp.scope == "holdout_last_6_months")].iloc[0]
    assert abs(row.mae - np.mean(np.abs(hold.actual - hold.forecast))) < 1e-9
    assert "coverage_80" in cmp.columns and cmp.coverage_80.between(0, 1).all()


def test_improved_model_status_is_reported_not_hidden():
    run = pipeline.run_forecast(csv_dir(), use_improved=True, rolling_step=12, departments=["ALL"])
    if models.available():
        assert run.meta["improved_model_status"] == "run" and "improved" in set(run.comparison.model)
    else:
        assert run.meta["improved_model_status"].startswith("NOT RUN") and set(run.comparison.model) == {"baseline"}


def test_horizon_validation():
    for h in (0, 13):
        try:
            pipeline.run_forecast(csv_dir(), horizon=h, use_improved=False)
            raise AssertionError("bad horizon accepted")
        except pipeline.ForecastError:
            pass


def test_forecast_only_refuses_when_it_would_ignore_the_scenario():
    """Runs whether or not statsmodels is installed: when it is, the improved model is switched off for this call (as in a baseline-only deployment)."""
    real = models.available
    models.available = lambda: False
    try:
        try:
            pipeline.forecast_only(csv_dir(), [events.Scenario("hiring_freeze", "2026-11", "2026-12")])
            raise AssertionError("a baseline cannot react to events; the call must be refused")
        except models.ModelUnavailable:
            pass
    finally:
        models.available = real


def test_statsmodels_level_and_count_models():
    need("statsmodels")
    y = _series(70)
    design = events.fourier_terms(y.index.append(pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=6, freq="MS")))
    p, lo, hi = models.forecast_level(y, design, 6)
    assert p.shape == lo.shape == hi.shape == (6,) and (lo <= p).all() and (p <= hi).all()
    assert (hi - lo)[-1] > (hi - lo)[0], "uncertainty must grow with the horizon"
    c = pd.Series(np.random.default_rng(2).poisson(5, 70).astype(float), index=y.index)
    p2, lo2, hi2 = models.forecast_count(c, design, 6)
    assert (lo2 >= 0).all() and (p2 >= 0).all()


def test_statsmodels_event_regressor_changes_forecast():
    need("statsmodels")
    ev = data.load_events(csv_dir())
    hist = data.load_history(csv_dir())
    y = data.get_series(hist, "ALL", "headcount_end")
    fut = pd.date_range("2026-10-01", periods=6, freq="MS")
    base = events.design_matrix(y.index.append(fut), ev, "ALL")
    frz = events.design_matrix(y.index.append(fut), ev, "ALL", [events.Scenario("hiring_freeze", "2026-10", "2027-03")])
    p0, _, _ = models.forecast_level(y, base, 6)
    p1, _, _ = models.forecast_level(y, frz, 6)
    assert p1[-1] < p0[-1], "a hiring freeze must lower the projected headcount"


def test_eda_describes_trend_seasons_and_unusual_months():
    import tempfile
    from pathlib import Path

    from cortexhr_phase4.forecasting import eda
    hist, ev = data.load_history(csv_dir()), data.load_events(csv_dir())
    d = eda.describe_history(hist, ev)
    assert d["months"] == 81 and d["end_headcount"] == 441 and d["average_growth_per_month"] > 0
    assert len(d["monthly_averages"]) == 12 and set(d["season_peaks"]) == {"hires_peak_month", "hires_low_month", "exits_peak_month", "exits_low_month"}
    assert all("events" in m and m["month"] for m in d["unusual_months"])
    png = eda.plot_history(hist, ev, Path(tempfile.mkdtemp()) / "h.png")
    assert png.exists() and png.stat().st_size > 10_000


# ------------------------------------------------------------ interval width calibration of the improved model (runs without statsmodels)
def test_width_scale_is_fitted_on_past_errors_and_applied_symmetrically():
    from cortexhr_phase4.forecasting import intervals
    rng = np.random.default_rng(5)
    f = np.full(400, 100.0)
    a = f + rng.normal(0, 4.0, 400)                      # true errors: sd 4  -> true 80% half-width about 5.1
    hi = f + 2.0                                          # a model interval that is far too narrow
    k = intervals.fit_width_scale(a, f, hi)
    assert 2.0 < k < 3.1, k
    lo2, hi2 = intervals.apply_width_scale(f, hi, k)
    assert abs(np.mean((a >= lo2) & (a <= hi2)) - 0.80) < 0.03
    lo3, _ = intervals.apply_width_scale(np.array([1.0]), np.array([3.0]), 5.0)
    assert lo3[0] == 0.0                                  # the floor at zero still applies


def test_out_of_sample_check_uses_only_earlier_origins_to_fit():
    from cortexhr_phase4.forecasting import intervals
    rng = np.random.default_rng(6)
    origins = pd.date_range("2020-01-01", periods=60, freq="MS")
    rows = [{"origin": o, "h": h, "actual": 100 + rng.normal(0, 4), "forecast": 100.0, "lower_80": 98.0, "upper_80": 102.0} for o in origins for h in (1, 2, 3)]
    res = intervals.out_of_sample_check(pd.DataFrame(rows))
    assert res["coverage_raw"] < 0.55 and abs(res["coverage_calibrated"] - 0.80) < 0.12, res
    assert res["n_test"] == 60                            # the last third of the origins x 3 horizons
    assert intervals.out_of_sample_check(pd.DataFrame(rows[:10])) is None      # too little data: no calibration is claimed


def test_walk_forward_calibration_uses_only_known_earlier_errors_and_adapts_to_regimes():
    from cortexhr_phase4.forecasting import intervals
    rng = np.random.default_rng(11)
    origins = pd.date_range("2018-01-01", periods=48, freq="MS")
    rows = []
    for i, o in enumerate(origins):
        sd = 8.0 if i < 20 else 2.5                      # noisy early period, calm recent period (the situation seen in the real data)
        for h in range(1, 7):
            rows.append({"metric": "m", "department_id": "ALL", "origin": o, "h": h, "month": o + pd.DateOffset(months=h), "actual": 100 + rng.normal(0, sd), "forecast": 100.0, "lower_80": 100 - 3.0, "upper_80": 100 + 3.0})
    roll = pd.DataFrame(rows)
    wf = intervals.pooled_walk_forward(roll)
    assert wf is not None and wf["origin"].min() > origins[2]                    # a warm-up period is never scored
    hit = ((wf["actual"] >= wf["lo_cal"]) & (wf["actual"] <= wf["hi_cal"])).mean()
    raw = ((wf["actual"] >= wf["lower_80"]) & (wf["actual"] <= wf["upper_80"])).mean()
    assert raw < 0.75 and hit > raw + 0.08 and 0.72 <= hit <= 0.92, (raw, hit)
    late = wf[wf["origin"] > origins[40]]
    assert late["k"].max() < wf["k"].max()                     # the noisy early period left the window: the factor shrinks again
    # no look-ahead: scores for one origin are unchanged when FUTURE rows are removed
    cut = roll[roll["origin"] <= origins[30]]
    a = wf[wf["origin"] == origins[30]]["k"].iloc[0]
    b = intervals.pooled_walk_forward(cut)
    assert abs(b[b["origin"] == origins[30]]["k"].iloc[0] - a) < 1e-12
    assert intervals.pooled_walk_forward(roll[roll["origin"] < origins[3]]) is None


def test_pipeline_reports_walk_forward_calibration_honestly():
    """Integration without statsmodels: a stand-in 'improved' model whose interval is far too narrow must be widened and reported walk-forward."""
    from cortexhr_phase4.forecasting import models as fm

    def narrow(y, design, horizon):
        p = np.asarray(SeasonalNaiveDrift().forecast(y, horizon), float)
        return p, p - 0.5, p + 0.5

    real_avail, real_pair = fm.available, pipeline._model_pair
    fm.available = lambda: True
    pipeline._model_pair = lambda metric: (real_pair(metric)[0], narrow)
    try:
        run = pipeline.run_forecast(csv_dir(), departments=["ALL"], rolling_step=1)
    finally:
        fm.available, pipeline._model_pair = real_avail, real_pair
    cmp = run.comparison
    key = (cmp.metric == "headcount_end") & (cmp.department_id == "ALL") & (cmp.model == "improved")
    raw = cmp[key & (cmp.scope == "rolling_origin_raw_same_rows")].iloc[0]
    cal = cmp[key & (cmp.scope == "rolling_origin_calibrated_walk_forward")].iloc[0]
    assert raw.n == cal.n and raw.coverage_80 < 0.5 and cal.coverage_80 > raw.coverage_80 + 0.2
    entry = run.meta["interval_calibration"]["headcount_end"]
    assert entry["final_scale"] > 1 and "pooled" in entry["method"]
    assert "EARLIER, resolved errors" in run.meta["interval_calibration_note"]
