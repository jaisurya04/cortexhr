"""End-to-end headcount / exits forecast: baseline vs improved model, backtests, selection, explanation, persistence."""
from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest as bt
from . import data, events, intervals, models
from .baseline import SeasonalMean, SeasonalNaiveDrift
from .intervals import COVERAGE

VERSION = "cortexhr-forecast-1.0.0"
IMPROVEMENT_MARGIN = 0.99          # the improved model must beat the baseline's rolling MAE by at least 1% to be selected


class ForecastError(RuntimeError):
    """A forecast could not be produced (bad input, not enough history, ...)."""


@dataclass
class ForecastRun:
    forecast: pd.DataFrame                 # sample_forecast columns
    detailed: pd.DataFrame                 # both models side by side
    holdout: pd.DataFrame                  # last-6-months backtest rows
    rolling: pd.DataFrame                  # rolling-origin backtest rows
    comparison: pd.DataFrame               # accuracy per model
    events_used: pd.DataFrame
    explanation: dict
    meta: dict
    history: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)


def _targets(history: pd.DataFrame) -> list[tuple[str, str]]:
    return [("headcount_end", d) for d in data.series_ids(history)] + [("exits", "ALL")]


def _model_pair(metric: str):
    if metric == "headcount_end":
        base = SeasonalNaiveDrift()
        return base, models.forecast_level
    return SeasonalMean(3), models.forecast_count


def _calibrate_improved_pooled(improved_roll, final, cmp_rows, calibration):
    """Widen the improved model's intervals with ONE factor per metric learned from all series' own recent, already-known errors.

    Reported honestly: coverage is walk-forward (each origin scored with a factor fitted only on earlier, resolved errors), next to the raw coverage on
    exactly the same rows. The real forecast uses the factor fitted on the most recent origins. Chosen from five candidate methods compared in
    scripts/interval_calibration_lab.py (pooled, last 12 origins: 79.4% over all series vs 63.0% raw; company headcount 75.4% vs 69.3%)."""
    if not improved_roll:
        return
    roll = pd.concat(improved_roll, ignore_index=True)
    wf = intervals.pooled_walk_forward(roll)
    ks = intervals.pooled_final_scale(roll)
    if wf is None or not ks:
        return
    for (metric, dept), g in wf.groupby(["metric", "department_id"]):
        ids = {"metric": metric, "department_id": dept, "model": "improved", "n": int(len(g))}
        raw = float(((g["actual"] >= g["lower_80"]) & (g["actual"] <= g["upper_80"])).mean())
        cal = float(((g["actual"] >= g["lo_cal"]) & (g["actual"] <= g["hi_cal"])).mean())
        cmp_rows.append({**ids, "scope": "rolling_origin_raw_same_rows", "coverage_80": raw, "mean_interval_width": float((g["upper_80"] - g["lower_80"]).mean())})
        cmp_rows.append({**ids, "scope": "rolling_origin_calibrated_walk_forward", "coverage_80": cal, "mean_interval_width": float((g["hi_cal"] - g["lo_cal"]).mean())})
    for metric, g in wf.groupby("metric"):
        calibration[metric] = {"method": "pooled over all series, last 12 origins", "final_scale": ks.get(metric), "scored_rows": int(len(g)),
                               "coverage_raw_same_rows": float(((g["actual"] >= g["lower_80"]) & (g["actual"] <= g["upper_80"])).mean()),
                               "coverage_calibrated_walk_forward": float(((g["actual"] >= g["lo_cal"]) & (g["actual"] <= g["hi_cal"])).mean())}
    for (metric, dept, kind), (label, point, lo, hi) in list(final.items()):
        if kind == "improved" and metric in ks:
            lo2, hi2 = intervals.apply_width_scale(point, hi, ks[metric])
            final[(metric, dept, kind)] = (label, point, lo2, hi2)


def _score_rows(table: pd.DataFrame, **ids) -> dict:
    return {**ids, **bt.score(table)}


def run_forecast(csv_dir: Path, horizon: int = 6, holdout: int = 6, scenarios: list[events.Scenario] | None = None,
                 use_improved: bool = True, rolling_step: int = 1, departments: list[str] | None = None) -> ForecastRun:
    """Forecast ``horizon`` months of headcount (company + departments) and exits (company), with 80% intervals."""
    if horizon < 1 or horizon > 12:
        raise ForecastError("horizon must be between 1 and 12 months")
    csv_dir = Path(csv_dir)
    history, ev = data.load_history(csv_dir), data.load_events(csv_dir)
    last = history.month.max()
    future_idx = pd.date_range(last + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    improved_on = use_improved and models.available()
    targets = _targets(history)
    if departments:
        targets = [t for t in targets if t[1] in set(departments) | {"ALL"}]

    fc_rows, hold_rows, roll_rows, cmp_rows, final = [], [], [], [], {}
    calibration: dict[str, dict] = {}
    improved_roll: list = []
    for metric, dept in targets:
        y = data.get_series(history, dept, metric)
        if len(y) < bt.first_origin_with_interval(horizon) + holdout:
            raise ForecastError(f"{dept}/{metric}: {len(y)} months is too short to calibrate and backtest")
        base, improved_fn = _model_pair(metric)
        full_idx = y.index.append(future_idx)
        design = events.design_matrix(full_idx, ev, dept, scenarios)
        base_fc = lambda tr, hz, n, base=base: bt.baseline_with_interval(base.forecast, tr, hz)
        imp_fc = lambda tr, hz, n, d=design, f=improved_fn: f(tr, d.iloc[:n + hz], hz)
        fns = {"baseline": (base.name, base_fc)}
        if improved_on:
            fns["improved"] = (models.NAME, imp_fc)
        for kind, (label, fn) in fns.items():
            try:
                h_tab = bt.evaluate_origin(y, len(y) - holdout, holdout, fn)
                r_tab = bt.rolling_backtest(y, horizon, fn, bt.first_origin_with_interval(horizon), rolling_step)
                point, lo, hi = fn(y, horizon, len(y))
            except (models.ModelFitError, ValueError) as exc:
                cmp_rows.append({"metric": metric, "department_id": dept, "model": kind, "method": label, "scope": "failed", "note": str(exc)})
                continue
            for tab in (h_tab, r_tab):
                tab.insert(0, "model", kind); tab.insert(0, "department_id", dept); tab.insert(0, "metric", metric)
            hold_rows.append(h_tab); roll_rows.append(r_tab)
            cmp_rows.append(_score_rows(h_tab, metric=metric, department_id=dept, model=kind, method=label, scope="holdout_last_6_months"))
            cmp_rows.append(_score_rows(r_tab, metric=metric, department_id=dept, model=kind, method=label, scope="rolling_origin"))
            if kind == "improved":
                improved_roll.append(r_tab)
            final[(metric, dept, kind)] = (label, point, lo, hi)

    _calibrate_improved_pooled(improved_roll, final, cmp_rows, calibration)
    comparison = pd.DataFrame(cmp_rows)
    detailed_rows, chosen = [], {}
    for (metric, dept) in dict.fromkeys((m, d) for m, d, _ in final):
        roll = comparison[(comparison.metric == metric) & (comparison.department_id == dept) & (comparison.scope == "rolling_origin")].set_index("model")
        pick = "baseline"
        if {"improved", "baseline"} <= set(roll.index) and roll.loc["improved", "mae"] < IMPROVEMENT_MARGIN * roll.loc["baseline", "mae"]:
            pick = "improved"
        chosen[(metric, dept)] = pick
        for kind in ("baseline", "improved"):
            if (metric, dept, kind) in final:
                label, p, lo, hi = final[(metric, dept, kind)]
                for h, m in enumerate(future_idx):
                    detailed_rows.append({"metric": metric, "department_id": dept, "model": kind, "selected": kind == pick, "method": label,
                                          "forecast_month": m, "h": h + 1, "forecast_value": p[h], "lower_80": lo[h], "upper_80": hi[h]})
    detailed = pd.DataFrame(detailed_rows)
    sel = detailed[detailed.selected].copy()
    forecast = _sample_format(sel, history, future_idx, last)
    holdout_df = pd.concat(hold_rows, ignore_index=True) if hold_rows else pd.DataFrame()
    rolling_df = pd.concat(roll_rows, ignore_index=True) if roll_rows else pd.DataFrame()
    scen = [s.__dict__ for s in (scenarios or [])]
    meta = {"model_version": VERSION, "horizon": horizon, "holdout_months": holdout, "coverage": COVERAGE, "trained_through": last.strftime("%Y-%m"),
            "improved_model": "statsmodels SARIMAX (events + seasonality)" if improved_on else None,
            "improved_model_status": "run" if improved_on else ("disabled by caller" if not use_improved else "NOT RUN: statsmodels is not installed in this environment"),
            "statsmodels_version": models.statsmodels_version(), "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "interval_calibration": calibration,
            "interval_calibration_note": "The improved model's own intervals were too narrow in rolling backtests. One width factor per metric is learned from ALL series' most recent already-known errors (pooled adaptive split-conformal, window = last 12 origins, chosen from five methods fixed in advance in scripts/interval_calibration_lab.py). Coverage is measured walk-forward: every origin is scored with a factor fitted only on EARLIER, resolved errors (scope rolling_origin_calibrated_walk_forward), next to the raw coverage on the same rows. Neighbouring origins overlap and the history is short, so this is indicative, not a guarantee.",
            "scenarios": scen, "selection_rule": f"improved model selected only if rolling-origin MAE < {IMPROVEMENT_MARGIN} x baseline MAE",
            "selected": {f"{m}/{d}": k for (m, d), k in chosen.items()},
            "data_sha256": {data.HISTORY_FILE: data.sha256(csv_dir / data.HISTORY_FILE), data.EVENTS_FILE: data.sha256(csv_dir / data.EVENTS_FILE)}}
    run = ForecastRun(forecast, detailed, holdout_df, rolling_df, comparison, events.event_table(ev), {}, meta, history)
    run.explanation = explain(run, scenarios)
    return run


def _sample_format(sel: pd.DataFrame, history: pd.DataFrame, future_idx: pd.DatetimeIndex, last: pd.Timestamp) -> pd.DataFrame:
    """Rows in the exact column layout of ``sample_forecast.csv`` (headcount_end, exits, attrition_rate_monthly_pct)."""
    rows = []
    hc_all = sel[sel.metric == "headcount_end"]
    for dept in ["ALL"] + sorted(d for d in hc_all.department_id.unique() if d != "ALL"):
        for _, r in hc_all[hc_all.department_id == dept].sort_values("h").iterrows():
            rows.append((r.metric, r.department_id, r.forecast_month, r.forecast_value, r.lower_80, r.upper_80, r.method, 1))
    ex = sel[(sel.metric == "exits") & (sel.department_id == "ALL")].sort_values("h")
    for _, r in ex.iterrows():
        rows.append(("exits", "ALL", r.forecast_month, r.forecast_value, r.lower_80, r.upper_80, r.method, 1))
    hc = sel[(sel.metric == "headcount_end") & (sel.department_id == "ALL")].sort_values("h")
    start = [float(history[(history.department_id == "ALL") & (history.month == last)].headcount_end.iloc[0])] + hc.forecast_value.tolist()[:-1]
    for (_, e), s in zip(ex.iterrows(), start):
        rows.append(("attrition_rate_monthly_pct", "ALL", e.forecast_month, e.forecast_value / s * 100, e.lower_80 / s * 100, e.upper_80 / s * 100,
                     "exits forecast / projected start-of-month headcount", 2))
    out = pd.DataFrame(rows, columns=["metric", "department_id", "forecast_month", "forecast_value", "lower_80", "upper_80", "method", "dp"])
    for c in ("forecast_value", "lower_80", "upper_80"):
        out[c] = [round(v, dp) for v, dp in zip(out[c], out.dp)]
    out["forecast_month"] = pd.to_datetime(out.forecast_month).dt.strftime("%Y-%m-%d")
    out["trained_through"] = last.strftime("%Y-%m")
    out["model_version"] = VERSION
    out.insert(0, "forecast_id", [f"FC-{i:04d}" for i in range(1, len(out) + 1)])
    return out[data.SAMPLE_COLUMNS].reset_index(drop=True)


def explain(run: ForecastRun, scenarios: list[events.Scenario] | None) -> dict:
    """Plain-language facts derived only from the run's own tables (nothing is hard-coded)."""
    hc = run.forecast[(run.forecast.metric == "headcount_end") & (run.forecast.department_id == "ALL")].reset_index(drop=True)
    cur = float(run.history[(run.history.department_id == "ALL")].sort_values("month").headcount_end.iloc[-1])
    end = float(hc.forecast_value.iloc[-1])
    ex = run.forecast[run.forecast.metric == "exits"]
    width = (hc.upper_80 - hc.lower_80)
    hold = run.comparison[(run.comparison.scope == "holdout_last_6_months") & (run.comparison.department_id == "ALL") & (run.comparison.metric == "headcount_end")]
    roll = run.comparison[(run.comparison.scope == "rolling_origin") & (run.comparison.department_id == "ALL") & (run.comparison.metric == "headcount_end")]
    trust = pd.DataFrame({"forecast_month": hc.forecast_month, "interval_width": width.round(1), "relative_width_pct": (width / hc.forecast_value * 100).round(2)})
    least = trust.sort_values(["interval_width", "forecast_month"]).iloc[-1]
    evs = events.active_events(run.events_used, run.history.month.min(), run.history.month.max())
    future_events = run.meta["scenarios"]
    return {
        "current_headcount": cur, "headcount_in_6_months": end, "growth_pct_6m": round((end / cur - 1) * 100, 1),
        "expected_exits_6m": round(float(ex.forecast_value.sum()), 1),
        "trust_by_month": trust.to_dict("records"),
        "least_trusted_month": {"month": str(least.forecast_month), "interval_width": float(least.interval_width),
                                "why": "its 80% interval is the widest: uncertainty about monthly hires and exits accumulates month after month."},
        "backtest_company_headcount": {r.model: {"holdout_mae": float(r.mae), "holdout_coverage_80": float(getattr(r, "coverage_80", np.nan))} for r in hold.itertuples()},
        "rolling_company_headcount": {r.model: {"rolling_mae": float(r.mae), "rolling_coverage_80": float(getattr(r, "coverage_80", np.nan))} for r in roll.itertuples()},
        "events_in_history": evs[["event_id", "event_name", "effect", "department_id", "start_date", "end_date"]].astype(str).to_dict("records"),
        "future_events_known": "none in workforce_events.csv after 2026-04" if not future_events else future_events,
        "events_handling": "Past events enter the improved model as 0/1 regressors (hiring freeze, hiring boost, policy change); a future "
                           "what-if event can be supplied as a scenario. The baseline ignores events on purpose: it is the benchmark.",
        "limits": ["A single hiring freeze (2020) is the only example the model has of that effect, so its size is uncertain.",
                   "Intervals ignore parameter uncertainty and assume no new unmodelled event (e.g. a sudden layoff).",
                   "Department series are small; their intervals are wide relative to the level."],
    }


def write_outputs(run: ForecastRun, out_dir: Path) -> dict[str, Path]:
    """Persist everything to ``out_dir`` (CSV / JSON / Markdown). Deterministic content; only ``created_utc`` varies."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    files = {"forecast": out / "forecast_next_6_months.csv", "detailed": out / "forecast_detailed_both_models.csv",
             "holdout": out / "backtest_holdout_last_6_months.csv", "rolling": out / "backtest_rolling_origin.csv",
             "comparison": out / "model_comparison.csv", "events": out / "events_applied.csv",
             "explanation": out / "forecast_explanation.json", "meta": out / "forecast_run.json"}
    run.forecast.to_csv(files["forecast"], index=False)
    _round(run.detailed).to_csv(files["detailed"], index=False)
    _round(run.holdout).to_csv(files["holdout"], index=False)
    _round(run.rolling).to_csv(files["rolling"], index=False)
    _round(run.comparison).to_csv(files["comparison"], index=False)
    run.events_used.to_csv(files["events"], index=False)
    files["explanation"].write_text(json.dumps(run.explanation, indent=2, default=str), encoding="utf-8")
    meta = dict(run.meta, content_hash=content_hash(run.forecast), created_utc=pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"))
    files["meta"].write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return files


def _round(df: pd.DataFrame, nd: int = 4) -> pd.DataFrame:
    """Round float columns only (datetime columns are left alone)."""
    out = df.copy()
    num = out.select_dtypes(include="float").columns
    out[num] = out[num].round(nd)
    return out


def content_hash(forecast: pd.DataFrame) -> str:
    import hashlib
    return hashlib.sha256(forecast.to_csv(index=False).encode()).hexdigest()


def forecast_only(csv_dir: Path, scenarios: list[events.Scenario], departments: list[str] | None = None, horizon: int = 6,
                  selected: dict | None = None) -> pd.DataFrame:
    """Fast what-if forecast (no backtests): refit the improved model on the full history with scenario events as future regressors.

    Raises :class:`models.ModelUnavailable` when statsmodels is missing - a baseline cannot react to events, so returning one
    would silently ignore the scenario.
    """
    if not models.available():
        raise models.ModelUnavailable("scenario forecasts need the improved model (statsmodels), which is not installed here")
    if not scenarios:
        raise ForecastError("at least one scenario event is required")
    csv_dir = Path(csv_dir)
    history, ev = data.load_history(csv_dir), data.load_events(csv_dir)
    last = history.month.max()
    future_idx = pd.date_range(last + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    rows = []
    for metric, dept in _targets(history):
        if departments and dept not in set(departments) | {"ALL"}:
            continue
        y = data.get_series(history, dept, metric)
        design = events.design_matrix(y.index.append(future_idx), ev, dept, scenarios)
        fn = models.forecast_level if metric == "headcount_end" else models.forecast_count
        point, lo, hi = fn(y, design, horizon)
        for h, m in enumerate(future_idx):
            rows.append({"metric": metric, "department_id": dept, "model": "improved", "selected": True, "method": models.NAME,
                         "forecast_month": m, "h": h + 1, "forecast_value": point[h], "lower_80": lo[h], "upper_80": hi[h]})
    return _sample_format(pd.DataFrame(rows), history, future_idx, last)
