#!/usr/bin/env python
"""Append the complete 38-stage classification workflow to every notebook in 05_Starter/notebooks/phase2.

    python scripts/build_phase2_workflow38.py              # build + execute every notebook
    python scripts/build_phase2_workflow38.py --no-exec    # build only
    python scripts/build_phase2_workflow38.py 07_payroll_data_audit.ipynb P2_T06_payroll_outlier_detector.ipynb

* Source of truth for the stage titles/items: docs/CLASSIFICATION_WORKFLOW_38.md
* Stage code: scripts/p2nb/wf38_payroll_*.py (payroll/anomaly notebooks) and wf38_chatbot_*.py (chatbot notebooks)
* Existing cells are never modified; cells added here carry metadata.wf38 and are rebuilt on every run.
* Outputs are produced by really executing the cells; the shipped models in models/ are never overwritten
  (demo artefacts go to reports/phase2/workflow38/).
"""
import pathlib, sys, time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "p2nb"))
import wf38, wf38_payroll_1 as P1, wf38_payroll_2 as P2, wf38_chatbot_1 as C1, wf38_chatbot_2 as C2   # noqa: E402

TRACKS = {
    "payroll": dict(label="Payroll / anomaly detection (binary classification: payslip error vs normal; rules + unsupervised Isolation Forest)",
                    setup=P1.SETUP, code={**P1.CODE, **P2.CODE}, notes={**P1.NOTES, **P2.NOTES}),
    "chatbot": dict(label="HR chatbot (33-class intent classification of one message; Logistic Regression on word + character TF-IDF)",
                    setup=C1.SETUP, code={**C1.CODE, **C2.CODE}, notes={**C1.NOTES, **C2.NOTES}),
}

# notebook -> (track, what this notebook is mainly about, {stage: specific remark})
NOTEBOOKS = {
    "07_payroll_data_audit.ipynb": ("payroll", "The original cells above audit the payroll file. Stages 4-7 (collection, loading, dictionary, quality audit) and 9-10 (leakage, cleaning) are this notebook's own job; the other stages apply the same workflow to the same data so the notebook is complete on its own.",
        {7: "Compare with the audit cells above: same counts, plus an issue log with a resolution for every finding."}),
    "08_payroll_eda_features.ipynb": ("payroll", "The original cells above explore the data and build features. Stages 8 (EDA), 11 (feature engineering) and 13 (preprocessing pipeline) are this notebook's own job.",
        {11: "The feature-set comparison shows the ratio features carry most of the signal."}),
    "09_payroll_rules_baselines.ipynb": ("payroll", "The original cells above build the transparent rules and baselines. Stages 14 (baselines), 15 (model selection) and 18 (rare positive class) are this notebook's own job.",
        {14: "Rules alone catch the structured errors; the table shows exactly which error types need the model."}),
    "10_payroll_iforest_train_score.ipynb": ("payroll", "The original cells above train and freeze the Isolation Forest. Stages 16-17 (training, time-aware CV), 19 (tuning), 21 (threshold) and 31 (packaging) are this notebook's own job.",
        {31: "The shipped models/payroll_detector_v2.joblib is not touched; the demo model goes to reports/phase2/workflow38/models_demo/."}),
    "11_payroll_error_type_evaluation.ipynb": ("payroll", "The original cells above evaluate per error type against the authorised CX-06 file. Stages 20 (evaluation), 24 (error analysis) and 27 (final test) are this notebook's own job.",
        {27: "CX-06 stays authorised and separate: if the vault file is missing the stage says BLOCKED instead of inventing labels."}),
    "12_anomaly_explanations_money_at_risk.ipynb": ("payroll", "The original cells above explain flags and rank review work. Stages 23 (explainability), 24 (error analysis) and 29 (prediction pipeline) are this notebook's own job.",
        {23: "Reason codes and money_at_risk are the explanation a clerk reads; the permutation importance explains the forest."}),
    "13_api_smoke_and_validation.ipynb": ("payroll", "The original cells above smoke-test the API contracts. Stages 29-34 (prediction pipeline, QA, packaging, deployment planning, deployment, monitoring) are this notebook's own job.",
        {33: "FastAPI checks run when FastAPI is installed; otherwise the stage says SKIPPED and tests the same code path through the service layer."}),
    "14_chatbot_nlu_workflow.ipynb": ("chatbot", "The original cells above run audit -> train -> evaluate for the chatbot. Stages 5-20 (loading to evaluation) are this notebook's own job.",
        {9: "Read this stage carefully: it explains why ordinary cross-validation says 0.99 while honest validation says about 0.64."}),
    "15_chatbot_faq_slots_permissions_ui.ipynb": ("chatbot", "The original cells above cover dialogue state, access control, FAQ answers and the UI. Stages 25 (responsible AI), 29 (prediction pipeline) and 30 (testing) are this notebook's own job.",
        {25: "The pay-privacy and permission checks run through the real HRChatbot, role by role."}),
    "16_chatbot_advanced_model_improvement.ipynb": ("chatbot", "The original cells above improve the NLU model. Stages 15-19 (model selection to tuning), 22 (calibration) and 23 (explainability) are this notebook's own job.",
        {19: "The tuned setting is adopted only if it wins consistently across grouped folds."}),
    "P2_MASTER_37_chatbot_classification_workflow.ipynb": ("chatbot", "The original 37-section chatbot workflow is above. This 38-stage section is the same workflow rebuilt from the project's checklist document, with every checklist item of every section listed and evidenced in code.",
        {27: "Compare this sealed-test evaluation with the master workflow's own test section above."}),
    "P2_T06_payroll_outlier_detector.ipynb": ("payroll", "The original task above builds the payroll outlier detector. Stages 14-17 are its core; the 38-stage section applies the full workflow end to end.", {}),
    "P2_T07_reason_codes_money_at_risk.ipynb": ("payroll", "The original task above adds reason codes and money at risk. Stages 23-24 are its core; the 38-stage section applies the full workflow end to end.", {}),
    "P2_T08_anomaly_api_service.ipynb": ("payroll", "The original task above builds the anomaly API service. Stages 29-33 are its core; the 38-stage section applies the full workflow end to end.", {}),
    "P2_T09_hr_intent_chatbot.ipynb": ("chatbot", "The original task above builds the HR intent chatbot. Stages 14-20 are its core; the 38-stage section applies the full workflow end to end.", {}),
    "P2_T10_slots_memory_permissions.ipynb": ("chatbot", "The original task above adds slots, memory and permissions. Stages 25 and 29 are its core; the 38-stage section applies the full workflow end to end.", {}),
}


def main(argv):
    do_exec = "--no-exec" not in argv
    names = [a for a in argv if not a.startswith("--")] or list(NOTEBOOKS)
    failures = []
    for name in names:
        track, focus, nb_notes = NOTEBOOKS[name]; t = TRACKS[track]; path = wf38.NB_DIR / name; t0 = time.time()
        cells = wf38.build_cells(name, t["label"], focus, t["setup"], t["code"], t["notes"], nb_notes)
        total = wf38.apply(path, cells); msg = f"{name:55s} track={track:8s} cells now={total:3d} (added {len(cells)})"
        if do_exec:
            n, errs = wf38.execute(path, verbose=False); msg += f" | executed {n} code cells in {time.time() - t0:.0f}s | errors={len(errs)}"; failures += errs
        print(msg, flush=True)
    if failures:
        print("\n".join(failures)); sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
