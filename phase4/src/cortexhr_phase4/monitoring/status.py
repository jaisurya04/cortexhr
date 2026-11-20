"""OK / WATCH / ALERT vocabulary and thresholds (defaults documented in docs/monitoring_policy.md)."""
from __future__ import annotations

OK, WATCH, ALERT, SKIPPED = "OK", "WATCH", "ALERT", "SKIPPED"
_RANK = {OK: 0, WATCH: 1, ALERT: 2}

PSI_WATCH, PSI_ALERT = 0.10, 0.25            # official Day 18 bands: below 0.10 / 0.10 to 0.25 / above 0.25
TWIN_TOLERANCE = 0.001                       # Phase 3 fairness tolerance (score points)
PAY_GAP_WATCH, PAY_GAP_ALERT = 0.02, 0.05    # gap between best and worst group mean of (predicted pay / band reference)
MIN_GROUP_SIZE = 30                          # same minimum as Phase 3 (smaller groups are not compared)


def psi_status(value: float, watch: float = PSI_WATCH, alert: float = PSI_ALERT) -> str:
    """PSI < watch -> OK, watch <= PSI <= alert -> WATCH, PSI > alert -> ALERT."""
    if value != value:                       # NaN: cannot be judged, never silently OK
        return WATCH
    if value > alert:
        return ALERT
    return WATCH if value >= watch else OK


def gap_status(value: float, watch: float = PAY_GAP_WATCH, alert: float = PAY_GAP_ALERT) -> str:
    if value != value:
        return WATCH
    return ALERT if value >= alert else (WATCH if value >= watch else OK)


def twin_status(max_abs_gap: float, nonzero: int, tol: float = TWIN_TOLERANCE) -> str:
    """ALERT if any twin pair differs by more than the tolerance, WATCH if any tiny difference exists, else OK."""
    if max_abs_gap > tol:
        return ALERT
    return WATCH if nonzero > 0 else OK


def worst(statuses) -> str:
    """Worst judged status; SKIPPED components are ignored here and reported separately as a coverage gap."""
    judged = [s for s in statuses if s in _RANK]
    return max(judged, key=_RANK.__getitem__) if judged else WATCH
