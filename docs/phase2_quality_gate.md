# Phase 2 Quality Gate and Submission Checklist

This checklist is evidence-driven. A checked item means the corresponding command/output was observed in the package validation run; it does not imply hidden evaluation data was available.

## Validation observed in this package environment
- Phase 1 suite: **32 passed** (`python tests/run_all.py`). Phase 2 suite: **49 passed** (`python tests_phase2/run_all.py`); `tests_phase2/test_api.py` needs FastAPI/httpx and must be run with `pytest -q tests_phase2` on a machine that has them.
- All 10 Phase 2 notebooks now have cell-level explanation blocks describing purpose, interpretation, reproducibility and leakage/safety checks. Existing executable code and stored outputs were preserved.
- The saved chatbot evaluation report records **81.06% (107/132)** for the word + character TF-IDF Logistic Regression pipeline. Re-run evaluation after any retraining or data changes before quoting the score.
- CX-06 was evaluated by the project owner with the authorised label file (not included in this package, never committed) against the frozen predictions: see `reports/phase2/payroll_cx06_per_type.csv` and `docs/PHASE2_STATUS_REVIEW.md`. Overall: precision 0.316, recall 0.95 (rules alone: precision 0.860, recall 0.92). The label file itself could not be inspected by the package author; the table is internally consistent (tp + fn = support, 95 tp / 301 flagged).

## Before submission
- [ ] Run all Phase 2 notebooks from a clean kernel, in order, on the target environment.
- [ ] Confirm generated outputs and evidence tables are fresh, not stale outputs from an earlier run.
- [x] Authorised `unlock_CX-06` evaluation run (see `reports/phase2/payroll_cx06_per_type.csv`).
- [ ] Confirm chatbot training/test files and 33 intent labels match the official task dataset.
- [ ] Review each intent's precision, recall, F1 and support; review actual confusion pairs.
- [ ] Add a separately collected, independently labelled test set to estimate performance on new employee messages.
- [ ] Run API tests and all 12 required dialogue scenarios; verify service-call behavior as well as response text.
- [ ] Verify role permissions and salary privacy using permitted and denied identities.
- [ ] Fill the Phase 2 report with actual metrics, evidence paths, limitations and next steps.
- [ ] Check Git status, repository link, requested tag and deliverable files before submitting.

## Chatbot model selection policy
1. Fit preprocessing only on training folds.
2. Select features and hyperparameters with stratified cross-validation on training data.
3. Do not use held-out test examples to create training examples or choose model settings.
4. Report accuracy, macro-F1, weighted-F1 and per-intent precision/recall/F1.
5. Tune confidence fallback on validation data and report its coverage/abstention rate; do not silently count abstentions as correct.
6. Evaluate robustness on new, independently labelled messages before making a generalisation claim.

## Interpretation
The supplied held-out set contains 132 examples. A requirement for 132/132 correct has not been established from the available task notes. Report actual results; do not manufacture 100% accuracy. High performance on this small test set alone is not sufficient evidence of production readiness.
