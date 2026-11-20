"""Run the six-month headcount and exits forecast, backtests and write every output to phase4/outputs/forecast/.

    python phase4/scripts/run_forecast.py                  # full run (improved model needs statsmodels)
    python phase4/scripts/run_forecast.py --baseline-only
    python phase4/scripts/run_forecast.py --scenario hiring_freeze:2026-11:2027-01
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cortexhr_phase4.core.settings import load_settings  # noqa: E402
from cortexhr_phase4.forecasting import data, eda, events, pipeline  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--baseline-only", action="store_true")
    ap.add_argument("--rolling-step", type=int, default=1, help="evaluate every n-th rolling origin (larger = faster)")
    ap.add_argument("--scenario", action="append", default=[], metavar="EFFECT:START:END[:DEPT]", help="what-if event, e.g. hiring_freeze:2026-11:2027-01")
    a = ap.parse_args()
    s = load_settings()
    scen = []
    for spec in a.scenario:
        p = spec.split(":")
        scen.append(events.Scenario(p[0], p[1], p[2], p[3] if len(p) > 3 else "ALL", name="cli scenario"))
    run = pipeline.run_forecast(s.data_dir, use_improved=not a.baseline_only, rolling_step=a.rolling_step, scenarios=scen or None)
    files = pipeline.write_outputs(run, s.forecast_dir)
    hist, ev = data.load_history(s.data_dir), data.load_events(s.data_dir)
    import json
    (s.forecast_dir / "history_description.json").write_text(json.dumps(eda.describe_history(hist, ev), indent=2), encoding="utf-8")
    eda.plot_history(hist, ev, s.forecast_dir / "headcount_history.png")
    print(f"improved model: {run.meta['improved_model_status']}")
    print(run.forecast.groupby("metric").size().to_string())
    print(f"growth in 6 months: {run.explanation['growth_pct_6m']}% | least trusted month: {run.explanation['least_trusted_month']['month']}")
    for k, v in files.items():
        print(f"  {k:12s} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
