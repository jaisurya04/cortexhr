"""Day 4: honest evaluation + risk bands. Run: python scripts/evaluate_models.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from cortexhr.runs import run_evaluation
if __name__ == "__main__":
    o = run_evaluation(); print(o["metrics"][["evaluation", "roc_auc", "pr_auc", "precision", "recall", "top_precision", "top_recall", "top_lift"]].round(3).to_string())
    for k, t in o["band_summary"].items(): print(k); print(t.round(2).to_string())
