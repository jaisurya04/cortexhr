"""Turn ``workforce_events`` into model inputs instead of ignoring them.

Each event is classed by its name into one of three effects, then expanded to a 0/1 monthly indicator per series:

* ``hiring_freeze``  - hiring almost stops (e.g. "COVID-19 hiring freeze")
* ``hiring_boost``   - hiring drive / team build-out / scale-up
* ``policy_change``  - appraisal, increment, bonus or work-policy change (can move exits)

An event applies to a department series when its ``department_id`` is ``ALL`` or equals the series; the company total
(``ALL``) feels every event. Future what-if events are supplied as :class:`Scenario` objects.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EFFECTS = ("hiring_freeze", "hiring_boost", "policy_change")
_RULES = (("freeze", "hiring_freeze"), ("drive", "hiring_boost"), ("scale-up", "hiring_boost"), ("build-out", "hiring_boost"),
          ("expansion", "hiring_boost"))


@dataclass(frozen=True)
class Scenario:
    """A what-if workforce event placed in the future (e.g. a planned hiring freeze)."""

    effect: str
    start: str
    end: str
    department_id: str = "ALL"
    name: str = "scenario"

    def __post_init__(self) -> None:
        if self.effect not in EFFECTS:
            raise ValueError(f"effect must be one of {EFFECTS}, got {self.effect!r}")
        if pd.Timestamp(self.end) < pd.Timestamp(self.start):
            raise ValueError("scenario end is before start")


def classify(event_name: str) -> str:
    """Effect class of an event name (anything not recognised is a policy/process change)."""
    low = event_name.lower()
    for key, effect in _RULES:
        if key in low:
            return effect
    return "policy_change"


def event_table(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    out["effect"] = out.event_name.map(classify)
    return out


def event_matrix(index: pd.DatetimeIndex, events: pd.DataFrame, department_id: str, scenarios: list[Scenario] | None = None) -> pd.DataFrame:
    """0/1 indicator matrix (rows = months, columns = :data:`EFFECTS`) for one series."""
    ev = event_table(events)
    rows = [(r.effect, r.start_date, r.end_date, r.department_id) for r in ev.itertuples()]
    rows += [(s.effect, pd.Timestamp(s.start), pd.Timestamp(s.end), s.department_id) for s in (scenarios or [])]
    m = pd.DataFrame(0.0, index=index, columns=list(EFFECTS))
    for effect, start, end, dept in rows:
        if department_id == "ALL" or dept in ("ALL", department_id):
            m.loc[(index >= start.to_period("M").to_timestamp()) & (index <= end), effect] = 1.0
    return m


def fourier_terms(index: pd.DatetimeIndex, harmonics: int = 2) -> pd.DataFrame:
    """Yearly seasonality as sin/cos pairs (few parameters, suits ~80 monthly points)."""
    t = index.month.to_numpy() - 1
    cols = {}
    for k in range(1, harmonics + 1):
        cols[f"sin{k}"] = np.sin(2 * np.pi * k * t / 12)
        cols[f"cos{k}"] = np.cos(2 * np.pi * k * t / 12)
    return pd.DataFrame(cols, index=index)


def design_matrix(index: pd.DatetimeIndex, events: pd.DataFrame, department_id: str, scenarios: list[Scenario] | None = None,
                  harmonics: int = 2) -> pd.DataFrame:
    """Events + seasonality: the exogenous regressors of the improved model."""
    return pd.concat([event_matrix(index, events, department_id, scenarios), fourier_terms(index, harmonics)], axis=1)


def active_events(events: pd.DataFrame, start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DataFrame:
    """Events overlapping ``[start, end]`` (used to annotate charts and explain unusual months)."""
    ev = event_table(events)
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    return ev[(ev.end_date >= s) & (ev.start_date <= e)].reset_index(drop=True)
