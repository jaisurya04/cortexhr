"""Fallback runner when pytest is unavailable: python tests/run_all.py  (pytest tests/ is preferred)."""
import sys, pathlib, importlib, traceback
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
ok = bad = 0
for f in sorted(pathlib.Path(__file__).parent.glob("test_*.py")):
    m = importlib.import_module(f.stem)
    for n in [n for n in dir(m) if n.startswith("test_")]:
        try: getattr(m, n)(); ok += 1; print("PASS", f.stem, n)
        except Exception: bad += 1; print("FAIL", f.stem, n); traceback.print_exc()
print(f"\n{ok} passed, {bad} failed"); sys.exit(1 if bad else 0)
