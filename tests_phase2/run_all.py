"""Run the Phase 2 tests without pytest:  python tests_phase2/run_all.py
(pytest is preferred where installed:  pytest -q tests_phase2)
A module that needs pytest or fastapi and cannot import it is reported as SKIPPED, never as passed."""
import importlib
import pathlib
import sys
import traceback

here = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(here)); sys.path.insert(0, str(here.parent / "src"))
ok = bad = skipped = 0
for f in sorted(here.glob("test_*.py")):
    try:
        m = importlib.import_module(f.stem)
    except ImportError as exc:
        skipped += 1; print(f"SKIPPED {f.stem}: {exc}"); continue
    for n in [n for n in dir(m) if n.startswith("test_")]:
        try:
            getattr(m, n)(); ok += 1; print("PASS", f.stem, n)
        except Exception:
            bad += 1; print("FAIL", f.stem, n); traceback.print_exc()
print(f"\n{ok} passed, {bad} failed, {skipped} module(s) skipped")
sys.exit(1 if bad else 0)
