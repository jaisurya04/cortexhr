"""Evidence files must agree with each other and with the code (no stale numbers)."""
import pandas as pd

from _helpers import CSV, ROOT

R = ROOT / "reports" / "phase2"


def test_validation_summary_agrees_with_the_chatbot_metric_summary():
    s = pd.read_csv(R / "phase2_validation_summary.csv").set_index("check").result
    m = pd.read_csv(R / "chatbot_metric_summary.csv").set_index("metric").value
    assert abs(float(s["Chatbot held-out accuracy"]) - float(m["heldout_accuracy"])) < 1e-4
    assert int(s["Chatbot held-out errors"]) == int(m["heldout_errors"])
    assert int(s["Chatbot training examples"]) == 858 and int(s["Chatbot held-out test rows"]) == 132


def test_payroll_explanations_file_is_official_and_ranked():
    out = pd.read_csv(R / "payroll_anomaly_explanations.csv")
    assert list(out.columns) == list(pd.read_csv(CSV / "sample_anomaly_results.csv").columns)
    y = out[out.flagged == "Y"]
    assert len(y) > 0 and (out.flagged.iloc[:len(y)] == "Y").all() and y.money_at_risk.is_monotonic_decreasing
    assert not out.reason_text.str.contains("validation rules triggered").any()


def test_the_20_row_clerk_review_sample_exists():
    t = pd.read_csv(R / "payroll_review_sample_20.csv")
    assert len(t) == 20 and t.reason_text.str.len().min() > 30


def test_misclassification_count_matches_the_accuracy():
    m = pd.read_csv(R / "chatbot_metric_summary.csv").set_index("metric").value
    errs = pd.read_csv(R / "chatbot_misclassifications.csv")
    assert len(errs) == int(m["heldout_errors"]) == round(132 * (1 - float(m["heldout_accuracy"])))
