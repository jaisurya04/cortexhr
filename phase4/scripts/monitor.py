"""Drift (PSI) + fairness re-check. Prints ONE clear report, saves it, and returns a status through the exit code.

    python phase4/scripts/monitor.py            # exit code 0 = OK, 1 = WATCH, 2 = ALERT
    python phase4/scripts/monitor.py --no-save  # print only

Schedule it with cron or GitHub Actions (see phase4/deploy/github-actions-monitor.yml and docs/monitoring_policy.md).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cortexhr_phase4.core.settings import load_settings  # noqa: E402
from cortexhr_phase4.monitoring import monitor  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-save", action="store_true", help="do not write the report files")
    a = ap.parse_args()
    s = load_settings()
    report = monitor.run_monitor(s)
    print(monitor.render_text(report))
    if not a.no_save:
        files = monitor.persist(report, s)
        print(f"\nsaved: {files['latest']}  (history: {files['history']})")
    return monitor.EXIT_CODE[report["status"]]


if __name__ == "__main__":
    sys.exit(main())
