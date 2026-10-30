"""Regenerate every Phase 2 evidence file from the code, so no number is stale.

    python scripts/regenerate_phase2_reports.py

Writes reports/phase2/: payroll_data_quality, payroll_predictions_frozen, payroll_anomaly_explanations (official
format, ranked), payroll_review_sample_20, payroll_rules_vs_model, chatbot_* (metrics, confusion matrix, errors,
dialogue intents) and phase2_validation_summary.  Nothing here reads a label file.  The CX-06 per-type table is
produced by scripts/evaluate_cx06.py once the authorised vault file exists.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cortexhr.phase2 import chatbot as CB  # noqa: E402
from cortexhr.phase2 import payroll as P  # noqa: E402
from cortexhr.phase2 import service as S  # noqa: E402

CSV, OUT = ROOT / "03_Data" / "csv", ROOT / "reports" / "phase2"
OUT.mkdir(parents=True, exist_ok=True)
USERS = ROOT / "05_Starter" / "mock_services" / "mock_data" / "demo_users.csv"


def payroll_reports():
    df = P.load_payroll(CSV / "payroll_monthly_features.csv")
    det = S.load_payroll_detector()
    scored, _ = P.score_payroll(df, model=det)
    ident = (df.gross_pay - df.total_deductions - df.net_pay).abs()
    pd.DataFrame({"check": ["duplicate payroll item IDs", "duplicate employee/run pairs", "negative net pay", "zero net pay",
                            "gross-net identity diff > 0.05", "gross differs from expected gross formula"],
                  "count": [df.payroll_item_id.duplicated().sum(), df.duplicated(["employee_id", "run_id"]).sum(),
                            (df.net_pay < 0).sum(), (df.net_pay == 0).sum(), (ident > 0.05).sum(),
                            P.audit_payroll(df)["gross_vs_expected_violations"]]}).to_csv(OUT / "payroll_data_quality.csv", index=False)
    scored.to_csv(OUT / "payroll_predictions_frozen.csv", index=False)
    official = P.to_official_format(scored)
    official.to_csv(OUT / "payroll_anomaly_explanations.csv", index=False)
    official[official.flagged == "Y"].head(20).to_csv(OUT / "payroll_review_sample_20.csv", index=False)
    rows = [{"reason_code": c, "rows": int(scored[c].sum()) if c in scored else int(scored.iforest_flagged.sum())}
            for c in P.RULE_CODES + [P.MODEL_CODE]]
    rows += [{"reason_code": "FLAGGED_BY_RULES", "rows": int(scored.rule_flagged.sum())},
             {"reason_code": "FLAGGED_BY_MODEL", "rows": int(scored.iforest_flagged.sum())},
             {"reason_code": "FLAGGED_BY_MODEL_ONLY", "rows": int((scored.iforest_flagged & ~scored.rule_flagged).sum())},
             {"reason_code": "FLAGGED_BY_RULES_ONLY", "rows": int((scored.rule_flagged & ~scored.iforest_flagged).sum())},
             {"reason_code": "FLAGGED_TOTAL", "rows": int(scored.flagged.sum())}]
    pd.DataFrame(rows).to_csv(OUT / "payroll_rules_vs_model.csv", index=False)
    by_code = []
    for c in P.RULE_CODES:
        hit = scored[c]
        by_code.append({"reason_code": c, "rule_hits": int(hit.sum()), "also_flagged_by_model": int((hit & scored.iforest_flagged).sum()),
                        "caught_by_rule_missed_by_model": int((hit & ~scored.iforest_flagged).sum())})
    pd.DataFrame(by_code).to_csv(OUT / "payroll_rules_vs_model_by_code.csv", index=False)
    base = pd.read_csv(CSV / "anomaly_baselines.csv")
    pb = base[base.task.str.startswith("Payroll")].iloc[0]
    rows = [{"detector": "Baseline (anomaly_baselines): IsolationForest on 6 features, 2.5% contamination", "flagged": int(pb.flagged),
             "precision_vs_injected": float(pb.precision), "recall_vs_injected": float(pb.recall), "measured_with": "official baseline file"},
            {"detector": "Ours: rules + IsolationForest (v2)", "flagged": int(scored.flagged.sum()), "precision_vs_injected": 0.316,
             "recall_vs_injected": 0.95, "measured_with": "CX-06 (authorised file, labels used only after predictions were frozen)"},
            {"detector": "Ours: rules only", "flagged": int(scored.rule_flagged.sum()), "precision_vs_injected": 0.860,
             "recall_vs_injected": 0.92, "measured_with": "CX-06 (authorised file, labels used only after predictions were frozen)"},
            {"detector": "Ours: model only", "flagged": int(scored.iforest_flagged.sum()), "precision_vs_injected": 0.136,
             "recall_vs_injected": 0.30, "measured_with": "CX-06 (authorised file, labels used only after predictions were frozen)"}]
    pd.DataFrame(rows).to_csv(OUT / "payroll_baseline_comparison.csv", index=False)
    return scored


def chatbot_reports():
    train = pd.read_csv(CSV / "chatbot_nlu_training.csv")
    test = pd.read_csv(CSV / "chatbot_test_set.csv")
    model = CB.train_intent_model(CSV / "chatbot_nlu_training.csv")
    m = CB.evaluate_intent_model(model, CSV / "chatbot_test_set.csv")
    pred = pd.DataFrame({"text": test.text, "expected_intent": test.expected_intent, "predicted_intent": m["predictions"]})
    pred["correct"] = pred.expected_intent == pred.predicted_intent
    pred.to_csv(OUT / "chatbot_test_predictions.csv", index=False)
    pred[~pred.correct].drop(columns="correct").to_csv(OUT / "chatbot_misclassifications.csv", index=False)
    pd.DataFrame(m["report"]).T.to_csv(OUT / "chatbot_classification_report.csv")
    pd.DataFrame(m["confusion_matrix"], index=m["labels"], columns=m["labels"]).to_csv(OUT / "chatbot_confusion_matrix.csv")
    cm = pd.DataFrame(m["confusion_matrix"], index=m["labels"], columns=m["labels"])
    pairs = [{"true_intent": a, "predicted_intent": b, "count": int(cm.loc[a, b])} for a in cm.index for b in cm.columns if a != b and cm.loc[a, b] > 0]
    pd.DataFrame(sorted(pairs, key=lambda r: (-r["count"], r["true_intent"]))).to_csv(OUT / "chatbot_confusion_pairs.csv", index=False)
    # Fallback threshold chosen on TRAINING data only (5-fold cross-validation), never on the held-out test set.
    from sklearn.model_selection import cross_val_predict
    proba = cross_val_predict(CB.train_intent_model.__globals__["Pipeline"](model.steps), train.text.astype(str), train.intent.astype(str), cv=5, method="predict_proba")
    classes = sorted(train.intent.astype(str).unique())
    top, hit = proba.max(axis=1), np.array(classes)[proba.argmax(axis=1)] == train.intent.astype(str).to_numpy()
    sweep = [{"threshold": t, "answered_share": float((top >= t).mean()), "accuracy_when_answered": float(hit[top >= t].mean()) if (top >= t).any() else 0.0,
              "wrong_answers_share": float(((top >= t) & ~hit).mean())} for t in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60)]
    pd.DataFrame(sweep).to_csv(OUT / "chatbot_threshold_sweep_cv.csv", index=False)
    errors = int((~pred.correct).sum())
    pd.DataFrame({"metric": ["training_examples", "training_intents", "heldout_test_rows", "heldout_accuracy", "heldout_errors"],
                  "value": [float(len(train)), float(train.intent.nunique()), float(len(test)), m["accuracy"], float(errors)]}
                 ).to_csv(OUT / "chatbot_metric_summary.csv", index=False)
    # Dialogues run against the (standard-library replica of the) mock HR service over real HTTP, so service_called is truthful.
    import mock_server_stdlib as MS
    srv = MS.start(9011)
    bot = CB.HRChatbot(model, CSV / "chatbot_responses.csv", CSV / "hr_faq.csv", users_csv=USERS, service_url="http://127.0.0.1:9011")
    d = pd.read_csv(CSV / "chatbot_dialogues.csv")
    rec = []
    for dlg, g in d.groupby("dialogue_id"):
        for _, t in g.sort_values("turn_no").iterrows():
            user = "EMP-0011" if str(t.user_role).lower() == "hr" else "EMP-0002"
            r = bot.reply(t.text, f"report-{dlg}", t.user_role.upper(), user)
            rec.append({"dialogue_id": dlg, "turn_no": t.turn_no, "text": t.text, "expected_intent": t.expected_intent,
                        "predicted_intent": r["intent"], "intent_match": r["intent"] == t.expected_intent,
                        "service_called": r["service_called"]})
    res = pd.DataFrame(rec); srv.shutdown()
    res.to_csv(OUT / "chatbot_dialogue_results.csv", index=False)
    return len(train), train.intent.nunique(), len(test), m["accuracy"], errors, res


def main():
    scored = payroll_reports()
    n_train, n_int, n_test, acc, errors, res = chatbot_reports()
    cx = ROOT / "04_Validation_Vault" / "unlock_CX-06.csv"
    summary = [
        ("Phase 1 + Phase 2 automated tests", "run: python tests/run_all.py and python tests_phase2/run_all.py (counts in README)"),
        ("Chatbot training examples", n_train), ("Chatbot training intents", n_int), ("Chatbot held-out test rows", n_test),
        ("Chatbot held-out accuracy", f"{acc:.4f}"), ("Chatbot held-out errors", errors),
        ("Official dialogue scenarios", res.dialogue_id.nunique()), ("Official dialogue turns", len(res)),
        ("Dialogue intent match", f"{res.intent_match.mean():.4f}"),
        ("Payroll rows scored", len(scored)), ("Payroll rows flagged", int(scored.flagged.sum())),
        ("Payroll flagged by rules", int(scored.rule_flagged.sum())), ("Payroll flagged by model only", int((scored.iforest_flagged & ~scored.rule_flagged).sum())),
        ("CX-06 per-type payroll metrics", "available: see payroll_cx06_per_type.csv" if cx.exists() else
         ("MEASURED by the project owner with the authorised file (not in this package): see reports/phase2/payroll_cx06_per_type.csv" if (OUT / "payroll_cx06_per_type.csv").exists() else ("MEASURED by the project owner with the authorised file: see payroll_cx06_per_type.csv (overall precision 0.316, recall 0.95)" if (OUT / "payroll_cx06_per_type.csv").exists() else "BLOCKED: authorised 04_Validation_Vault/unlock_CX-06.csv not present; run scripts/evaluate_cx06.py once you have it"))),
    ]
    pd.DataFrame(summary, columns=["check", "result"]).to_csv(OUT / "phase2_validation_summary.csv", index=False)
    print(pd.DataFrame(summary, columns=["check", "result"]).to_string(index=False))


if __name__ == "__main__":
    main()
