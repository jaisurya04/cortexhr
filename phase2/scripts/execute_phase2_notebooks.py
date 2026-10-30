"""Execute the Phase 2 notebooks (05_Starter/notebooks/phase2/*.ipynb) cell by cell and embed outputs.
Run from the repository root:  python scripts/execute_phase2_notebooks.py [name_substring ...]
Uses the same runner as scripts/execute_notebooks.py (for machines without Jupyter). Fails on any error."""
import pathlib, sys, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import execute_notebooks as X
ROOT = pathlib.Path(__file__).resolve().parents[1]
os.chdir(ROOT)
NB2 = ROOT / "05_Starter" / "notebooks" / "phase2"
if __name__ == "__main__":
    want = sys.argv[1:]; bad = 0
    for p in sorted(NB2.glob("P2_*.ipynb")):
        if want and not any(w in p.name for w in want): continue
        n, errs = X.run(p); print(f"{p.name}: {n} code cells, {len(errs)} errors"); bad += len(errs)
        for e in errs: print(e)
    sys.exit(1 if bad else 0)
