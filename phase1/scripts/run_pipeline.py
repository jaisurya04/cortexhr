"""Day 2: reusable feature pipeline - join report, 3-row smoke test, training-only proof. Run: python scripts/run_pipeline.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from cortexhr.workflow_outputs import pipeline_outputs
if __name__ == "__main__":
    c = pipeline_outputs(); print(*(f"{k}: {v}" for k, v in c.items()), sep="\n")
    assert c["all_values_finite"] and not c["target_in_output"] and not c["leakage_column_in_output"] and c["unseen_category_and_missing_handled"] and c["n_test_rows_used_to_fit"] == 0
    print("3-row smoke test PASSED")
