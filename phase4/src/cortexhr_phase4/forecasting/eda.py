"""Step 1 of Day 16: look at the history before modelling - trend, seasons and unusual months."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import data, events


def describe_history(history: pd.DataFrame, ev: pd.DataFrame, robust_z: float = 3.0) -> dict:
    """Numbers behind the written description (all computed from the data)."""
    a = data.get_series(history, "ALL", "headcount_end")
    hires, exits = data.get_series(history, "ALL", "hires"), data.get_series(history, "ALL", "exits")
    net = a.diff().dropna()
    t = np.arange(len(a)); slope, _ = np.polyfit(t, a.to_numpy(), 1)
    yoy = (a.groupby(a.index.year).last().pct_change() * 100).dropna().round(1)
    season = pd.DataFrame({"hires": hires.groupby(hires.index.month).mean(), "exits": exits.groupby(exits.index.month).mean(),
                           "net_change": net.groupby(net.index.month).mean()}).round(2)
    med = net.median(); mad = float(np.median(np.abs(net - med))) or 1.0
    z = 0.6745 * (net - med) / mad
    odd = net[np.abs(z) >= robust_z]
    unusual = []
    for m, v in odd.items():
        near = events.active_events(ev, m, m)
        unusual.append({"month": m.strftime("%Y-%m"), "net_change": float(v), "robust_z": round(float(z[m]), 1),
                        "events": near.event_name.tolist() or ["no recorded event"]})
    return {"first_month": a.index[0].strftime("%Y-%m"), "last_month": a.index[-1].strftime("%Y-%m"), "months": int(len(a)),
            "start_headcount": float(a.iloc[0]), "end_headcount": float(a.iloc[-1]), "average_growth_per_month": round(float(slope), 2),
            "year_end_growth_pct": {str(k): float(v) for k, v in yoy.items()},
            "season_peaks": {"hires_peak_month": int(season.hires.idxmax()), "hires_low_month": int(season.hires.idxmin()),
                             "exits_peak_month": int(season.exits.idxmax()), "exits_low_month": int(season.exits.idxmin())},
            "monthly_averages": season.reset_index(names="month").to_dict("records"), "unusual_months": unusual}


def plot_history(history: pd.DataFrame, ev: pd.DataFrame, path: Path) -> Path:
    """Headcount, hires/exits and event bands in one figure (matplotlib, Agg backend)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a = data.get_series(history, "ALL", "headcount_end"); hires = data.get_series(history, "ALL", "hires"); exits = data.get_series(history, "ALL", "exits")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(a.index, a, color="#0b6b5a", lw=2); ax1.set_ylabel("Headcount (end of month)"); ax1.set_title("Monthly headcount history: company total")
    ax2.bar(hires.index, hires, width=20, color="#c9a24d", label="Hires"); ax2.bar(exits.index, -exits, width=20, color="#8a4b3a", label="Exits (shown below zero)")
    ax2.axhline(0, color="#444", lw=.6); ax2.set_ylabel("Hires / exits"); ax2.legend(loc="upper left", ncol=2, fontsize=8)
    for r in events.event_table(ev).itertuples():
        for ax in (ax1, ax2):
            ax.axvspan(r.start_date, r.end_date + pd.Timedelta(days=1), color={"hiring_freeze": "#7a8aa0", "hiring_boost": "#7fb069"}.get(r.effect, "#d9c58a"), alpha=.28)
        ax1.annotate(r.event_id, (r.start_date, a.max() * 1.02), fontsize=7, color="#333")
    for ax in (ax1, ax2):
        ax.grid(alpha=.25)
    fig.tight_layout(); path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=130); plt.close(fig)
    return path
