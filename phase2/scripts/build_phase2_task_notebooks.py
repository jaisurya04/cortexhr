"""Build the Phase 2 task notebooks (Tasks 6-10) and the 37-item chatbot classification master, Phase-1 style:
12 'Stage N of 12' headings, 'Think for yourself' + Answer, checklist cell, evidence/real-companies/connects footer.
    python scripts/build_phase2_task_notebooks.py && python scripts/execute_phase2_notebooks.py"""
import importlib, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
for mod in ["t06", "t07", "t08", "t09", "t10", "master37"]:
    try:
        m = importlib.import_module(f"p2nb.{mod}")
    except ModuleNotFoundError as e:
        if e.name != f"p2nb.{mod}": raise
        print("skip (not written yet):", mod); continue
    print("built", m.build())
