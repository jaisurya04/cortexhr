"""Day 3: baselines. Run: python scripts/train_baselines.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from cortexhr.runs import run_baselines
if __name__ == "__main__":
    o = run_baselines(); print(o["comparison"].round(3).to_string()); print(*o["interpretation"], sep="\n")
    from cortexhr.workflow_outputs import baseline_extras; baseline_extras(o)   # confusion matrices, tuning, interpretation
