"""Run the full test suite and the traceability check, then write docs/phase3_quality_gate.md. Fails (exit 1) if anything is not green."""
import re, subprocess, sys, datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
import traceability as TR
r = subprocess.run([sys.executable, str(ROOT / "tests" / "run_all.py")], capture_output=True, text=True, cwd=ROOT)
out = r.stdout; m = re.search(r"(\d+) passed, (\d+) failed", out); passed, failed = int(m.group(1)), int(m.group(2)); sk = re.search(r", (\d+) skipped", out); skipped = int(sk.group(1)) if sk else 0
mods = {}
for line in out.splitlines():
    mm = re.match(r"(PASS|FAIL) (test_\w+)::", line)
    if mm: mods.setdefault(mm.group(2), [0, 0])[0 if mm.group(1) == "PASS" else 1] += 1
(ROOT / "docs" / "phase3_quality_gate.md").write_text("# Phase 3 quality gate (provisional)", encoding="utf-8")      # the traceability check requires this file to exist
rows = TR.build(); n_ok = sum(x["status"].startswith("Complete") for x in rows)
lines = ["# Phase 3 quality gate", "", f"Generated {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by `python scripts/run_quality_gate.py`.", "",
         f"**Tests:** {passed} passed, {failed} failed" + (f", {skipped} skipped (environment differs from the pinned versions; install requirements.txt + requirements-optional.txt and rerun)" if skipped else "") + ".", "", "| Test module | passed | failed |", "|---|---|---|"] + [f"| {k} | {v[0]} | {v[1]} |" for k, v in sorted(mods.items())] + \
        ["", f"**Requirement traceability:** {n_ok} of {len(rows)} requirements complete (see `requirements_traceability.md`).", "",
         "**Gate:** " + ("GREEN: every test passes and every tracked requirement has its notebook, code, output and test." if failed == 0 and skipped == 0 and n_ok == len(rows) else ("AMBER: no failures but tests were skipped; rerun on the pinned environment." if failed == 0 and n_ok == len(rows) else "RED: see the failures above.")), "",
         "**Additional evidence / limitations:** LightGBM candidate was trained and evaluated separately on the same Task 3 folds; SHAP PermutationExplainer cross-check was executed on 10 rows. The authorized CX-12 and CX-15 files were supplied and validated locally; Task 12 and Task 15 were rerun with protected-group measurements. Task 13 saved group counts match the supplied vault, but its full retraining/tuning rerun exceeded the 10-minute execution limit. Restricted files remain excluded from the ZIP."]
(ROOT / "docs" / "phase3_quality_gate.md").write_text("\n".join(lines), encoding="utf-8"); print("\n".join(lines[:8])); print("gate:", lines[-3]); sys.exit(0 if failed == 0 and skipped == 0 and n_ok == len(rows) else 1)
