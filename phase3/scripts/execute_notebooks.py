"""Execute notebooks (no Jupyter needed). Usage: python scripts/execute_notebooks.py [prefix ...]"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nbkit import execute_notebook

nbdir = Path(__file__).resolve().parents[1] / "notebooks"
prefixes = sys.argv[1:]
for p in sorted(nbdir.glob("*.ipynb")):
    if prefixes and not any(p.name.startswith(x) for x in prefixes): continue
    t = time.time(); r = execute_notebook(p); print(f"{p.name}: {r['code_cells']} code cells, {r['images']} images, {time.time() - t:.0f}s", flush=True)
