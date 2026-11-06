"""Run every test without pytest: python tests/run_all.py   (also works with: pytest -q tests)
Prints the installed vs pinned package versions first. A test that cannot run because the environment differs from the pinned one
is reported as SKIP with the reason (never silently counted as passed)."""
import importlib, importlib.metadata as md, inspect, pathlib, sys, time, traceback
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "src")); sys.path.insert(0, str(HERE.parent / "05_Starter"))
from _common import Skip, pinned
print("Environment (installed / pinned):")
mismatch = []
for pkg in ("scikit-learn", "pandas", "numpy", "scipy", "joblib", "lightgbm", "shap"):
    try: have = md.version(pkg)
    except md.PackageNotFoundError: have = None
    want = pinned(pkg); ok = have is not None and (want is None or have == want)
    print(f"  {pkg:13s} {have or 'NOT INSTALLED':14s} / {want or '-':8s} {'ok' if ok else 'DIFFERENT'}")
    if not ok: mismatch.append(pkg)
if mismatch: print("  -> differs from the pinned versions:", ", ".join(mismatch), "| run: pip install -r requirements.txt -r requirements-optional.txt")
passed = failed = skipped = 0; failures = []
for f in sorted(HERE.glob("test_*.py")):
    mod = importlib.import_module(f.stem)
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        if not name.startswith("test_") or fn.__module__ != mod.__name__: continue
        t = time.time()
        try: fn(); passed += 1; print(f"PASS {f.stem}::{name} ({time.time() - t:.1f}s)", flush=True)
        except Skip as e: skipped += 1; print(f"SKIP {f.stem}::{name}: {e}", flush=True)
        except Exception: failed += 1; failures.append((f.stem, name, traceback.format_exc())); print(f"FAIL {f.stem}::{name}", flush=True)
for m, n, tb in failures: print("=" * 70, f"\n{m}::{n}\n{tb}")
print(f"{passed} passed, {failed} failed" + (f", {skipped} skipped (environment differs from the pinned versions)" if skipped else "")); sys.exit(1 if failed else 0)
